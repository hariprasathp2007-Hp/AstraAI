import unittest

from fastapi.testclient import TestClient

from backend.main import app


class ChatApiValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def test_health_does_not_load_model(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "ok")
        self.assertEqual(response.json()["model_status"], "not_loaded")

    def test_empty_message_list_is_rejected(self) -> None:
        response = self.client.post("/chat", json={"messages": []})

        self.assertEqual(response.status_code, 422)

    def test_blank_message_is_rejected(self) -> None:
        response = self.client.post(
            "/chat",
            json={"messages": [{"role": "user", "content": "   "}]},
        )

        self.assertEqual(response.status_code, 422)

    def test_latest_assistant_message_is_rejected(self) -> None:
        response = self.client.post(
            "/chat",
            json={"messages": [{"role": "assistant", "content": "Hello"}]},
        )

        self.assertEqual(response.status_code, 422)

    def test_malformed_json_is_rejected(self) -> None:
        response = self.client.post(
            "/chat",
            content="{",
            headers={"Content-Type": "application/json"},
        )

        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
