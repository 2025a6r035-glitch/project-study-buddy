"""LLM + embedding clients. Gemini is used when a key is present;
a local hashing embedder is the offline fallback."""
from __future__ import annotations

import json
import re
import time
import zlib

import numpy as np

from . import config


class LLMError(RuntimeError):
    """Raised for any problem talking to the model."""


def extract_json(text: str):
    """Parse JSON from a model reply, tolerating code fences and stray prose."""
    text = (text or "").strip()
    fence = re.search(r"```(?:json)?\s*(.*?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}|\[.*\]", text, re.S)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass
    raise LLMError("The model did not return valid JSON.")


def _require_genai():
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise LLMError("Package 'google-genai' is not installed. Run: pip install google-genai") from exc
    return genai, types


class GeminiLLM:
    """Text generation via the Gemini API."""

    def __init__(self, api_key: str | None = None, model: str | None = None):
        key = (api_key or config.GEMINI_API_KEY).strip()
        if not key:
            raise LLMError("No Gemini API key found. Put GEMINI_API_KEY in your .env file.")
        genai, types = _require_genai()
        self._types = types
        self.client = genai.Client(api_key=key)
        self.model = model or config.GEN_MODEL

    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False,
                 temperature: float = 0.4) -> str:
        cfg = self._types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            response_mime_type="application/json" if json_mode else None,
        )
        last: Exception | None = None
        for attempt in range(3):
            try:
                resp = self.client.models.generate_content(model=self.model, contents=prompt, config=cfg)
                if resp.text:
                    return resp.text
                last = LLMError("Empty response from model")
            except Exception as exc:  # network, quota, bad model name...
                last = exc
            if attempt < 2:
                time.sleep(1.5 * (attempt + 1))
        raise LLMError(f"Gemini request failed: {last}")

    def generate_json(self, prompt: str, system: str | None = None):
        text = self.generate(prompt, system, json_mode=True, temperature=0.3)
        try:
            return extract_json(text)
        except LLMError:
            text = self.generate(prompt + "\n\nReturn ONLY valid JSON and nothing else.",
                                 system, json_mode=True, temperature=0.1)
            return extract_json(text)


class GeminiEmbedder:
    name = "gemini"

    def __init__(self, api_key: str | None = None, model: str | None = None):
        key = (api_key or config.GEMINI_API_KEY).strip()
        if not key:
            raise LLMError("No Gemini API key found.")
        genai, types = _require_genai()
        self._types = types
        self.client = genai.Client(api_key=key)
        self.model = model or config.EMBED_MODEL

    def embed(self, texts: list[str], task: str = "RETRIEVAL_DOCUMENT") -> np.ndarray:
        out: list[list[float]] = []
        for i in range(0, len(texts), 50):
            batch = texts[i:i + 50]
            last: Exception | None = None
            for attempt in range(3):
                try:
                    res = self.client.models.embed_content(
                        model=self.model, contents=batch,
                        config=self._types.EmbedContentConfig(task_type=task))
                    out.extend(list(e.values) for e in res.embeddings)
                    break
                except Exception as exc:
                    last = exc
                    if attempt < 2:
                        time.sleep(1.5 * (attempt + 1))
            else:
                raise LLMError(f"Embedding request failed: {last}")
        return np.asarray(out, dtype="float32")


_STOP = set("a an and are as at be by for from has have in is it its of on or that the this to was were what when where which who why with how do does did can i you we they".split())


class LocalEmbedder:
    """Offline hashing bag-of-words embedder (keyword-style retrieval, no API needed)."""
    name = "local"
    DIM = 768

    def embed(self, texts: list[str], task: str | None = None) -> np.ndarray:
        out = np.zeros((len(texts), self.DIM), dtype="float32")
        for i, text in enumerate(texts):
            toks = [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOP]
            for tok in toks:
                out[i, zlib.crc32(tok.encode()) % self.DIM] += 1.0
            for a, b in zip(toks, toks[1:]):
                out[i, zlib.crc32(f"{a} {b}".encode()) % self.DIM] += 0.5
        return np.log1p(out)


def make_embedder(api_key: str | None = None):
    key = (api_key or config.GEMINI_API_KEY).strip()
    if config.EMBEDDINGS == "local" or not key:
        return LocalEmbedder()
    try:
        return GeminiEmbedder(api_key=key)
    except LLMError:
        return LocalEmbedder()
