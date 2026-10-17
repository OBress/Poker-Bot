"""LLM client tests with a fake backend (no network)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pokerbot.llm import LLMClient, LLMSettings, extract_code


class FakeBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **request):
        self.calls += 1
        text = f"reply {self.calls} to {request['messages'][-1]['content']}"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            model=request["model"],
        )


def make(tmp_path: Path) -> tuple[LLMClient, FakeBackend]:
    backend = FakeBackend()
    settings = LLMSettings(api_key="test", cache_dir=tmp_path)
    return LLMClient(settings, client=backend), backend


def test_cache_hits_skip_the_backend(tmp_path: Path) -> None:
    client, backend = make(tmp_path)
    msgs = [{"role": "user", "content": "hi"}]
    first = client.complete(msgs)
    second = client.complete(msgs)
    assert backend.calls == 1
    assert second.cached and second.text == first.text
    assert client.usage.prompt_tokens == 10 and client.usage.cached_calls == 1


def test_sample_index_gives_distinct_cache_entries(tmp_path: Path) -> None:
    client, backend = make(tmp_path)
    msgs = [{"role": "user", "content": "hi"}]
    a = client.complete(msgs, sample=0)
    b = client.complete(msgs, sample=1)
    assert backend.calls == 2 and a.text != b.text


def test_complete_many_preserves_order(tmp_path: Path) -> None:
    client, _ = make(tmp_path)
    prompts = [[{"role": "user", "content": str(i)}] for i in range(10)]
    out = client.complete_many(prompts)
    assert [c.text.split(" to ")[1] for c in out] == [str(i) for i in range(10)]


def test_extract_code_picks_longest_python_block() -> None:
    text = "intro\n```python\nx = 1\n```\nmore\n```python\ndef f():\n    return 2\n```\n"
    assert extract_code(text) == "def f():\n    return 2\n"
    assert extract_code("no code here") is None
