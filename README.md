# Astra AI

**Your ideas. Infinite possibilities.** A locally running AI chat app with a space-inspired, glass-like interface. The browser talks only to your local FastAPI server; chat inference uses `HuggingFaceTB/SmolLM2-1.7B-Instruct` on your computer. There are no API keys, paid services, or Ollama dependency.

## Requirements

- Windows 10/11, 64-bit.
- Python **3.13 or 3.12**. Python 3.13 is recommended for the supplied PyTorch CPU wheel.
- Node.js LTS (includes npm).
- Internet access for the first dependency install and model download. Once cached, model inference can run offline.
- Plan for **at least 8 GB of available RAM** while generating; 16 GB of installed RAM is recommended. CPU inference is slower than a compatible GPU. The half-precision model cache needs roughly 4–5 GB of free disk space; Python packages and frontend dependencies need additional space.

The model is approximately 1.7 billion parameters. Astra loads it on the first chat request rather than blocking the web app at startup. The default install uses CPU-only PyTorch and half-precision weights to reduce memory use. If the machine cannot allocate enough memory, the backend reports a model-load error instead of returning a fake response. A compatible CUDA-enabled PyTorch build can use an NVIDIA GPU; its installation must match the installed driver and Python version (see [PyTorch's Windows install selector](https://pytorch.org/get-started/locally/)). The tested setup script intentionally selects the CPU build so that GPU support is never assumed.

## Setup (Windows PowerShell)

Open PowerShell in this project folder. Check the tools first:

```powershell
py -0p
node --version
npm.cmd --version
```

If Python 3.13/3.12 is missing, install 64-bit Python from [python.org](https://www.python.org/downloads/windows/) and enable the Python Launcher. If Node is missing, install the current LTS version from [nodejs.org](https://nodejs.org/). Close and reopen PowerShell after installation, then rerun the checks.

Run the setup script. It checks for Python 3.13/3.12 and Node/npm, creates `.venv`, installs CPU PyTorch and backend dependencies, and installs the frontend packages. If PowerShell blocks local scripts, allow scripts for this terminal only (this does not change the machine-wide policy), then run setup. `npm.cmd` is used so the script also works when the policy blocks `npm.ps1`.

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\setup.ps1
```

The first install downloads Python and JavaScript packages. PyTorch is a large download. To set up manually instead:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r .\backend\requirements.txt
Set-Location .\frontend
npm.cmd install
Set-Location ..
```

## Run locally

Keep both PowerShell terminals open.

**Terminal 1 — FastAPI** (from the project root):

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

**Terminal 2 — Vite**:

```powershell
Set-Location .\frontend
npm.cmd run dev
```

Open **http://127.0.0.1:5173**. The Vite `/api` proxy removes `/api` and forwards the request to FastAPI on `127.0.0.1:8000`.

## Test the local endpoints

Health does not load or download the model:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
```

Run the request-validation tests without downloading the model:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s backend
```

Send a real inference request (the first one downloads and loads the model):

```powershell
$body = @{
    messages = @(
        @{ role = "user"; content = "In one sentence, what makes the night sky look dark?" }
    )
} | ConvertTo-Json -Depth 5

Invoke-RestMethod -Uri http://127.0.0.1:8000/chat -Method Post -ContentType "application/json" -Body $body
```

Through the frontend proxy, send the same `$body` to `http://127.0.0.1:5173/api/chat`. Invalid input is rejected with HTTP 422; model loading failures return 503, and inference failures return 500 with a useful error detail. In the UI, the first message triggers model loading and the status appears under **Settings & status**.

## Model files and offline use

Hugging Face model files are downloaded automatically on the first chat request and cached under `~\.cache\astra-ai\huggingface`. Subsequent starts reuse the cache. To use only already-cached files, set `ASTRA_OFFLINE=1` before starting the backend:

```powershell
$env:ASTRA_OFFLINE = "1"
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Optional backend settings:

```powershell
$env:ASTRA_MAX_INPUT_CHARS = "4000"
$env:ASTRA_MAX_INPUT_TOKENS = "2048"
$env:ASTRA_MAX_NEW_TOKENS = "384"
$env:ASTRA_MAX_MESSAGES = "40"
$env:ASTRA_MODEL_CACHE = "D:\AstraModelCache"
```

Values are validated at startup. Chat history is saved in this browser's local storage; the model cache is separate. Neither is committed to this project.

## Netlify static preview

The public static interface is deployed at [astra-ai-ui-preview.netlify.app](https://astra-ai-ui-preview.netlify.app). Netlify serves only the UI; it does not host the FastAPI backend or SmolLM2 model. The preview disables chat controls and directs visitors to run Astra locally for AI replies. The deployment is configured by the root `netlify.toml`.
