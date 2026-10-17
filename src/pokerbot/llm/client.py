"""Client for the Luminal OpenAI-compatible endpoint (DeepSeek V4.1 Flash).

Used only offline by ``evolve/`` to write code. Adds a disk cache keyed by the full request
(so re-running an evolution is free and reproducible), bounded concurrency, and token logging.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

Message = dict[str, str]

DEFAULT_BASE_URL = "https://flash.luminal.cloud/v1"
DEFAULT_MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"


@dataclass(frozen=True)
class LLMSettings:
    api_key: str
    base_url: str = DEFAULT_BASE_URL
    model: str = DEFAULT_MODEL
    max_concurrency: int = 8
    timeout_s: float = 180.0
    cache_dir: Path = Path(".llm_cache")

    @staticmethod
    def from_env() -> LLMSettings:
        load_dotenv()
        key = os.environ.get("LUMINAL_API_KEY")
        if not key:
            raise RuntimeError("LUMINAL_API_KEY is not set (put it in .env)")
        return LLMSettings(
            api_key=key,
            base_url=os.environ.get("LUMINAL_BASE_URL", DEFAULT_BASE_URL),
            model=os.environ.get("LUMINAL_MODEL", DEFAULT_MODEL),
        )


@dataclass(frozen=True)
class Completion:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    cached: bool
    seconds: float


@dataclass
class Usage:
    calls: int = 0
    cached_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, c: Completion) -> None:
        with self._lock:
            self.calls += 1
            self.cached_calls += c.cached
            if not c.cached:
                self.prompt_tokens += c.prompt_tokens
                self.completion_tokens += c.completion_tokens


class LLMClient:
    def __init__(self, settings: LLMSettings, client: Any | None = None, cache: bool = True):
        self.settings = settings
        if client is None:
            from openai import OpenAI

            client = OpenAI(
                base_url=settings.base_url,
                api_key=settings.api_key,
                timeout=settings.timeout_s,
                max_retries=5,
            )
        self._client = client
        self._cache = cache
        self._limit = threading.Semaphore(settings.max_concurrency)
        self.usage = Usage()
        if cache:
            settings.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def from_env(cache: bool = True) -> LLMClient:
        return LLMClient(LLMSettings.from_env(), cache=cache)

    def complete(
        self,
        messages: Sequence[Message],
        temperature: float = 0.8,
        max_tokens: int = 4096,
        sample: int = 0,
    ) -> Completion:
        """One completion. ``sample`` distinguishes repeated draws of the same prompt in the
        cache, so asking for samples 0..k-1 gives k different (but reproducible) answers."""
        request = {
            "model": self.settings.model,
            "messages": list(messages),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        key = hashlib.sha256(
            json.dumps({**request, "sample": sample}, sort_keys=True).encode()
        ).hexdigest()
        path = self.settings.cache_dir / f"{key}.json"
        if self._cache and path.exists():
            data = json.loads(path.read_text(encoding="utf-8"))
            completion = Completion(**{**data, "cached": True})
            self.usage.add(completion)
            return completion

        with self._limit:
            start = time.perf_counter()
            response = self._client.chat.completions.create(**request)
            seconds = time.perf_counter() - start
        usage = getattr(response, "usage", None)
        completion = Completion(
            text=response.choices[0].message.content or "",
            model=getattr(response, "model", None) or self.settings.model,
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cached=False,
            seconds=seconds,
        )
        if self._cache:
            data = {k: v for k, v in completion.__dict__.items() if k != "cached"}
            path.write_text(json.dumps(data), encoding="utf-8")
            with (self.settings.cache_dir / "usage.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "time": time.time(), **data, "text": None}) + "\n")
        self.usage.add(completion)
        return completion

    def complete_many(
        self,
        prompts: Sequence[Sequence[Message]],
        temperature: float = 0.8,
        max_tokens: int = 4096,
    ) -> list[Completion]:
        """Run many prompts concurrently (bounded by ``max_concurrency``), in order."""
        with ThreadPoolExecutor(self.settings.max_concurrency) as pool:
            futures = [
                pool.submit(self.complete, p, temperature, max_tokens, sample=i)
                for i, p in enumerate(prompts)
            ]
            return [f.result() for f in futures]


_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str | None:
    """The longest fenced Python block in a response, or None."""
    blocks = _CODE_BLOCK.findall(text)
    return max(blocks, key=len).strip() + "\n" if blocks else None
