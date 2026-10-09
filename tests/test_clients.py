"""HTTP plumbing of the real clients, checked against a local mock server."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from research_brain.config import LLMProfile
from research_brain.llm import make_client

REQUESTS: list[dict] = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _reply(self, payload):
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/tags":
            self._reply({"models": [{"name": "qwen2.5:7b"}]})
        else:
            self._reply({"data": [{"id": "local-model"}]})

    def do_POST(self):
        data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        REQUESTS.append({"path": self.path, "headers": {k.lower(): v for k, v in self.headers.items()}, "json": data})
        if self.path == "/api/chat":
            self._reply({"message": {"content": "<think>hmm</think>ollama-ok"}})
        elif self.path == "/api/embed":
            self._reply({"embeddings": [[0.1, 0.2] for _ in data["input"]]})
        elif self.path.endswith("/chat/completions"):
            self._reply({"choices": [{"message": {"content": "openai-ok"}}]})
        elif self.path.endswith("/embeddings"):
            self._reply({"data": [{"embedding": [1.0, 0.0]} for _ in data["input"]]})
        elif self.path == "/v1/messages":
            self._reply({"content": [{"type": "text", "text": "claude-ok"}]})


@pytest.fixture(scope="module")
def server():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_ollama_chat_embed_and_tags(server):
    c = make_client(LLMProfile(provider="ollama", model="qwen2.5:7b", base_url=server, think=False))
    assert c.chat("sys", "hi", json_mode=True) == "<think>hmm</think>ollama-ok"
    req = REQUESTS[-1]["json"]
    assert req["format"] == "json" and req["think"] is False and req["options"]["num_ctx"] == 16384
    assert req["messages"][0] == {"role": "system", "content": "sys"}
    assert c.embed(["a", "b"]) == [[0.1, 0.2], [0.1, 0.2]]
    assert c.list_models() == ["qwen2.5:7b"]


def test_openai_compatible_with_key(server, monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    c = make_client(LLMProfile(provider="deepseek", model="deepseek-chat", base_url=server))
    assert c.chat("sys", "hi", json_mode=True) == "openai-ok"
    req = REQUESTS[-1]
    assert req["path"] == "/chat/completions" and req["headers"]["authorization"] == "Bearer sk-test"
    assert req["json"]["response_format"] == {"type": "json_object"}
    lm = make_client(LLMProfile(provider="lmstudio", model="local-model", base_url=server + "/v1"))
    lm.chat("s", "u", json_mode=True)
    assert "response_format" not in REQUESTS[-1]["json"]  # not sent to local servers by default


def test_anthropic_messages_api(server, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    c = make_client(LLMProfile(provider="anthropic", model="claude-sonnet-5-5", base_url=server))
    assert c.chat("system prompt", "hello") == "claude-ok"
    req = REQUESTS[-1]
    assert req["path"] == "/v1/messages"
    assert req["headers"]["x-api-key"] == "sk-ant-test" and req["headers"]["anthropic-version"] == "2023-06-01"
    assert req["json"]["system"] == "system prompt" and req["json"]["messages"][0]["role"] == "user"


def test_missing_api_key_is_reported(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = make_client(LLMProfile(provider="anthropic", model="x"))
    with pytest.raises(Exception, match="ANTHROPIC_API_KEY"):
        c.chat("s", "u")
