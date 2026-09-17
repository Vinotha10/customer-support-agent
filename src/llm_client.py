"""
Minimal provider-agnostic chat wrapper. Given the Anthropic billing issue
and the Gemini model/quota churn both hit during Day 3, the actual reply
+ escalation agent (reply_agent.py) should not be hard-coded to one
provider -- whichever one is actually working today should just work via
an environment variable, no code changes.

Usage:
    export LLM_PROVIDER=anthropic   (or) gemini   (or) ollama
    export ANTHROPIC_API_KEY=...    (if provider=anthropic)
    export GEMINI_API_KEY=...       (if provider=gemini)
    (ollama needs no API key -- runs a local server, see ollama.com)
    (or put these in a .env file -- see .env.example)

    from llm_client import chat, pacing_seconds
    reply_text = chat("some prompt")

Model defaults are the same ones already confirmed working earlier in this
project (Haiku for Anthropic, gemini-3.5-flash-lite for Gemini) -- override
with LLM_MODEL env var if needed.

Two different failure modes are handled differently, on purpose:
  - PER-MINUTE rate limits / transient server errors (503, etc.) -> retry
    the SAME model with exponential backoff. These resolve within seconds
    to a couple minutes.
  - DAILY quota exhaustion (Gemini's "GenerateRequestsPerDayPerProjectPerModel"
    error) -> backoff is pointless, a daily quota will not reset in the next
    60 seconds. Instead, automatically fall over to the NEXT model in
    _GEMINI_FALLBACK_MODELS, since Google tracks quota per-model, so a
    genuinely different model has its own separate, untouched allowance.
    This was a real, repeated problem in this project -- four separate
    Gemini quota/model interruptions in one day -- so it's handled
    structurally here instead of requiring a manual model swap each time.
"""
import os
import time

from dotenv import load_dotenv

load_dotenv(override=True)  # .env always wins, even over a stale $env: var set earlier in this shell session

PROVIDER = os.environ.get("LLM_PROVIDER", "").lower()

_ANTHROPIC_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_OLLAMA_DEFAULT_MODEL = "llama3.2:3b"  # small enough to run on CPU-only laptops at reasonable speed

# Confirmed available on this project's key via list_gemini_models.py.
# Tried in order; on daily-quota exhaustion of one, the next is used for
# the rest of the run (and stays "sticky" -- see _current_gemini_model).
_GEMINI_FALLBACK_MODELS = [
    "gemini-3.5-flash-lite",
    "gemini-2.5-flash",
    "gemini-flash-lite-latest",
    "gemini-2.5-pro",
]
_current_gemini_model_idx = 0  # module-level so a fallback "sticks" for the rest of the run

# Conservative default gaps between calls in a batch loop. Gemini's free
# tier hit a 15 requests/minute ceiling on gemini-3.5-flash-lite in
# practice -- 4.5s keeps well under that (13.3/min). Anthropic's paid tier
# has no comparable free-tier ceiling for this volume, so a light pacing
# gap is enough just to be a good API citizen.
_PACING_SECONDS = {"anthropic": 0.5, "gemini": 4.5, "ollama": 0.1}  # ollama is local, no external rate limit

_TRANSIENT_SIGNALS = (
    "503", "UNAVAILABLE", "high demand", "overloaded",  # transient server-side errors
    "502", "500", "timeout", "ServerError",  # other transient network/server issues
)
_PER_MINUTE_QUOTA_SIGNALS = ("PerMinute", "RequestsPerMinute")
_DAILY_QUOTA_SIGNALS = ("PerDay", "RequestsPerDay")


def pacing_seconds() -> float:
    return _PACING_SECONDS.get(PROVIDER, 3.0)


def _anthropic_chat(prompt: str, model: str = None) -> str:
    import anthropic
    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model or os.environ.get("LLM_MODEL", _ANTHROPIC_DEFAULT_MODEL),
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.content[0].text.strip()


def _gemini_chat(prompt: str, model: str = None) -> str:
    from google import genai
    client = genai.Client()
    # Deliberately does NOT check an LLM_MODEL env var here (unlike the
    # Anthropic path) -- that was a real bug: if LLM_MODEL happened to be
    # set, it silently overrode every fallback switch below, so the
    # printed "switching to X" messages were cosmetic while every actual
    # API call kept hitting the same already-exhausted model. Only an
    # explicit `model` argument (a deliberate one-off override) or the
    # fallback list is used.
    use_model = model or _GEMINI_FALLBACK_MODELS[_current_gemini_model_idx]
    resp = client.models.generate_content(model=use_model, contents=prompt)
    return (resp.text or "").strip()


def _ollama_chat(prompt: str, model: str = None) -> str:
    # No API key, no quota, no deprecation risk -- runs entirely on your
    # own machine via Ollama's local REST server (default port 11434).
    # Install: https://ollama.com/download, then `ollama pull llama3.2:3b`
    import requests
    resp = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model or os.environ.get("LLM_MODEL", _OLLAMA_DEFAULT_MODEL),
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,  # local CPU inference can be slow on the first call while the model loads into memory
    )
    if resp.status_code == 404:
        raise SystemExit(
            f"Ollama model not found. Run: ollama pull {model or _OLLAMA_DEFAULT_MODEL}\n"
            f"Also confirm Ollama is running (it should start automatically after install)."
        )
    resp.raise_for_status()
    return resp.json()["response"].strip()


_MODEL_UNAVAILABLE_SIGNALS = ("404", "NOT_FOUND", "no longer available")


def _classify_error(msg: str) -> str:
    if any(sig in msg for sig in _MODEL_UNAVAILABLE_SIGNALS):
        return "model_unavailable"
    if any(sig in msg for sig in _DAILY_QUOTA_SIGNALS):
        return "daily_quota"
    if any(sig in msg for sig in _PER_MINUTE_QUOTA_SIGNALS) or "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate_limit" in msg:
        return "per_minute"
    if any(sig in msg for sig in _TRANSIENT_SIGNALS):
        return "transient"
    return "other"


def _advance_gemini_fallback(reason: str) -> bool:
    """Returns True if it switched to a new model, False if all are exhausted."""
    global _current_gemini_model_idx
    if _current_gemini_model_idx < len(_GEMINI_FALLBACK_MODELS) - 1:
        _current_gemini_model_idx += 1
        new_model = _GEMINI_FALLBACK_MODELS[_current_gemini_model_idx]
        print(f"[llm_client] {reason} on previous model -- switching to '{new_model}' for the rest of this run")
        return True
    print(f"[llm_client] {reason} on ALL fallback models ({_GEMINI_FALLBACK_MODELS}). "
          f"Wait for quota reset / model availability, or switch LLM_PROVIDER=anthropic "
          f"in your .env to keep going today.")
    return False


def chat(prompt: str, model: str = None, max_retries: int = 4) -> str:
    if PROVIDER == "anthropic":
        fn = _anthropic_chat
    elif PROVIDER == "gemini":
        fn = _gemini_chat
    elif PROVIDER == "ollama":
        fn = _ollama_chat
    else:
        raise SystemExit(
            "Set LLM_PROVIDER to 'anthropic', 'gemini', or 'ollama' (env var or .env file) before running. "
            f"Currently set to: '{PROVIDER or '(not set)'}'"
        )

    for attempt in range(max_retries):
        try:
            return fn(prompt, model)
        except Exception as e:
            msg = str(e)
            kind = _classify_error(msg)

            if kind in ("daily_quota", "model_unavailable") and PROVIDER == "gemini" and model is None:
                # backoff is pointless for a whole-day quota OR a model that
                # no longer exists -- switch to the next fallback model instead
                reason = "daily quota exhausted" if kind == "daily_quota" else "model unavailable/deprecated"
                if _advance_gemini_fallback(reason):
                    continue  # retry immediately with the new model, no sleep needed
                raise


            if kind in ("per_minute", "transient") and attempt < max_retries - 1:
                wait = pacing_seconds() * (2 ** (attempt + 1))
                print(f"[llm_client] {kind} error (attempt {attempt + 1}/{max_retries}), waiting {wait:.0f}s...")
                time.sleep(wait)
                continue

            raise  # unrecognized error, or retries exhausted -- let the caller see it


if __name__ == "__main__":
    print(f"[llm_client] provider={PROVIDER or '(not set)'}")
    print(chat("Reply with exactly the word: ok"))
