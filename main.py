from __future__ import annotations

import logging
import os
import threading
from pathlib import Path
from typing import Literal

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from huggingface_hub.errors import HfHubHTTPError
from httpx import HTTPError as HttpxError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


logging.basicConfig(level=os.getenv("ASTRA_LOG_LEVEL", "INFO"))
logger = logging.getLogger("astra")

MODEL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
MODEL_CACHE = Path(
    os.getenv(
        "ASTRA_MODEL_CACHE",
        str(Path.home() / ".cache" / "astra-ai" / "huggingface"),
    )
).expanduser()
OFFLINE_MODE = os.getenv("ASTRA_OFFLINE", "").strip().lower() in {"1", "true", "yes"}


def _read_limit(name: str, default: int, minimum: int, maximum: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        limit = int(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
    if not minimum <= limit <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}.")
    return limit


MAX_INPUT_CHARS = _read_limit("ASTRA_MAX_INPUT_CHARS", 4000, 100, 20000)
MAX_INPUT_TOKENS = _read_limit("ASTRA_MAX_INPUT_TOKENS", 2048, 128, 8192)
MAX_NEW_TOKENS = _read_limit("ASTRA_MAX_NEW_TOKENS", 384, 16, 1024)
MAX_MESSAGES = _read_limit("ASTRA_MAX_MESSAGES", 40, 1, 100)

model_state: dict[str, str | None] = {
    "status": "not_loaded",
    "device": None,
    "error": None,
}
tokenizer: PreTrainedTokenizerBase | None = None
model: PreTrainedModel | None = None
model_device: torch.device | None = None
load_lock = threading.Lock()
inference_lock = threading.Lock()

app = FastAPI(title="Astra AI", version="1.0.0")
allowed_origins = [
    origin.strip()
    for origin in os.getenv(
        "ASTRA_CORS_ORIGINS",
        "http://localhost:5173,http://127.0.0.1:5173",
    ).split(",")
    if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message content cannot be empty.")
        return value


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    messages: list[ChatMessage] = Field(min_length=1, max_length=MAX_MESSAGES)

    @model_validator(mode="after")
    def latest_message_must_be_from_user(self) -> ChatRequest:
        if self.messages[-1].role != "user":
            raise ValueError("The latest message must be from the user.")
        return self


class ChatResponse(BaseModel):
    response: str


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "model": MODEL_ID,
        "model_status": model_state["status"],
        "device": model_state["device"],
        "error": model_state["error"],
        "max_input_chars": MAX_INPUT_CHARS,
        "max_input_tokens": MAX_INPUT_TOKENS,
        "max_new_tokens": MAX_NEW_TOKENS,
        "max_messages": MAX_MESSAGES,
        "offline_mode": OFFLINE_MODE,
    }


def _get_model() -> tuple[PreTrainedModel, PreTrainedTokenizerBase, torch.device]:
    global model, model_device, tokenizer

    if model is not None and tokenizer is not None and model_device is not None:
        return model, tokenizer, model_device

    with load_lock:
        if model is not None and tokenizer is not None and model_device is not None:
            return model, tokenizer, model_device

        if model_state["status"] == "error":
            raise HTTPException(
                status_code=503,
                detail=(
                    "The local model failed to load. Check the backend log and available "
                    "memory, then restart the backend to try again."
                ),
            )

        model_state["status"] = "loading"
        model_state["error"] = None
        logger.info("Loading %s from the local Hugging Face cache or downloading it.", MODEL_ID)
        try:
            loaded_tokenizer = AutoTokenizer.from_pretrained(
                MODEL_ID,
                cache_dir=MODEL_CACHE,
                local_files_only=OFFLINE_MODE,
                token=False,
                trust_remote_code=False,
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            loaded_model = AutoModelForCausalLM.from_pretrained(
                MODEL_ID,
                cache_dir=MODEL_CACHE,
                local_files_only=OFFLINE_MODE,
                token=False,
                trust_remote_code=False,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True,
                use_safetensors=True,
            )
            loaded_model.to(device)
            loaded_model.eval()
        except (
            HfHubHTTPError,
            HttpxError,
            ImportError,
            MemoryError,
            OSError,
            RuntimeError,
            ValueError,
        ) as exc:
            logger.exception("Could not load the local model.")
            model_state["status"] = "error"
            model_state["error"] = f"{type(exc).__name__}: {exc}"
            raise HTTPException(
                status_code=503,
                detail=(
                    f"Could not load {MODEL_ID} ({type(exc).__name__}). "
                    "Check the backend log, internet access for the first download, "
                    "and available RAM/disk space."
                ),
            ) from exc

        tokenizer = loaded_tokenizer
        model = loaded_model
        model_device = device
        model_state["status"] = "ready"
        model_state["device"] = str(device)
        logger.info("Model loaded on %s.", device)
        return model, tokenizer, device


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    loaded_model, loaded_tokenizer, device = _get_model()
    conversation = [
        {"role": message.role, "content": message.content}
        for message in request.messages
    ]

    try:
        prompt = loaded_tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
        )
        loaded_tokenizer.truncation_side = "left"
        inputs = loaded_tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to(device)

        with inference_lock, torch.inference_mode():
            generated = loaded_model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=loaded_tokenizer.eos_token_id,
                eos_token_id=loaded_tokenizer.eos_token_id,
            )
        answer_tokens = generated[0, inputs["input_ids"].shape[-1] :]
        answer = loaded_tokenizer.decode(answer_tokens, skip_special_tokens=True).strip()
        if not answer:
            raise RuntimeError("The model returned an empty response.")
    except (RuntimeError, ValueError, OSError, MemoryError, IndexError) as exc:
        logger.exception("Local model inference failed.")
        raise HTTPException(
            status_code=500,
            detail=f"Local inference failed ({type(exc).__name__}): {exc}",
        ) from exc

    return ChatResponse(response=answer)
