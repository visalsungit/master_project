import re
import json
import os
import sys
import subprocess
from typing import Any, Dict, List, Optional, Tuple

import requests
from pymongo import MongoClient

"""
ollama_chat.main
-----------------
Small CLI utility that combines Ollama LLM responses with MongoDB data
for simple banking-related Q&A (FX, fixed deposits, savings). This module
is intentionally lightweight and resilient when Ollama or MongoDB are
unavailable.

The changes made below aim to improve readability (type hints and
docstrings) without changing runtime behaviour.
"""


# ===============================
# Ollama Configuration
# ===============================
# Allow overrides without editing the script:
#   set OLLAMA_BASE_URL=http://127.0.0.1:11434
#   set OLLAMA_MODEL=qwen2.5:3b
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")



# ===============================
# MongoDB Configuration
# ===============================
MONGO_URI = "mongodb://localhost:27017"
DB_NAME = "banking_db"

COLL_BANK = "bank_code"
COLL_EXCHANGE = "exchange_rates"
COLL_FIXED = "fixed_deposits"
COLL_SAVINGS = "savings_accounts"
COLL_LOAN = "loan"
COLL_CREDIT_CARDS = "credit_cards"

# Optimized MongoDB connection with connection pooling
mongo_client = MongoClient(
    MONGO_URI,
    maxPoolSize=50,  # Max connections in pool
    minPoolSize=10,  # Min connections to maintain
    maxIdleTimeMS=45000,  # Close idle connections after 45s
    serverSelectionTimeoutMS=5000,  # Fail fast if server unavailable
)
db = mongo_client[DB_NAME]


# ===============================
# Helpers
# ===============================
_BANK_CACHE: Optional[List[Dict[str, Any]]] = None
_BANK_CODE_LOOKUP_CACHE: Dict[str, List[str]] = {}  # Cache for bank code extractions


ALLOWED_INTENTS = {
    "FX",
    "FIXED_DEPOSIT",
    "SAVINGS",
    "INTEREST",
    "LOAN",
    "CREDIT_CARD",
    "BANK",
    "GENERAL",
}


def _get_bank_master() -> List[Dict[str, Any]]:
    """Load bank master data once (best-effort).

    Expected fields: bank_code, bank_name.
    """
    global _BANK_CACHE
    if _BANK_CACHE is not None:
        return _BANK_CACHE
    try:
        _BANK_CACHE = list(db[COLL_BANK].find({"is_active": {"$ne": False}}, {"_id": 0}).limit(5000))
    except Exception:
        _BANK_CACHE = []
    return _BANK_CACHE


def extract_bank_codes_from_text(text: str) -> List[str]:
    """Return matched bank_codes based on substring match against bank_name.

    This is intentionally simple (Approach B). It works well for small, known bank lists.
    Uses caching to avoid redundant processing.
    """
    # Check cache first
    cache_key = (text or "").lower().strip()
    if cache_key in _BANK_CODE_LOOKUP_CACHE:
        return _BANK_CODE_LOOKUP_CACHE[cache_key].copy()
    
    t = cache_key
    if not t:
        return []
    
    user_tokens = set(re.findall(r"\b[a-z0-9]+\b", t))

    # Words that shouldn't be used to identify a bank.
    stop = {
        "bank",
        "plc",
        "co",
        "company",
        "ltd",
        "limited",
        "cambodia",
        "kh",
        "commercial",
        "of",
        "the",
        "and",
        # Common query words that should not map to bank codes.
        "can",
        "could",
        "would",
        "should",
        "you",
        "me",
        "my",
        "your",
        "please",
        "kindly",
        "who",
        "which",
        "what",
        "where",
        "when",
        "why",
        "how",
        "has",
        "have",
        "with",
        "without",
        "from",
        "like",
        "different",
        "specify",
        "specific",
        "list",
        "show",
        "give",
        "need",
        "want",
        "find",
        "unsecured",
        "secured",
        "collateral",
        "salary",
        "personal",
        "loan",
    }

    hits: List[str] = []
    for b in _get_bank_master():
        name = (b.get("bank_name") or "").strip()
        code = (b.get("bank_code") or "").strip()
        if not name or not code:
            continue

        name_l = name.lower()
        code_l = code.lower()

        # Full-string match (fast path)
        if name_l and name_l in t:
            hits.append(code)
            continue

        # Code match
        if code_l and code_l in t:
            hits.append(code)
            continue

        # Token/prefix match on bank_code (e.g. user types "ppc" for "ppcbbank").
        # Strip common suffixes and compare short prefixes.
        code_token = code_l
        if code_token.endswith("bank"):
            code_token = code_token[: -len("bank")]
        # Accept 3-5 char prefixes as identifiers, but only when the prefix token
        # is not a common stopword (to avoid false positives like "can" -> CANADIABANK).
        for n in (5, 4, 3):
            prefix = code_token[:n] if len(code_token) >= n else ""
            if prefix and prefix in user_tokens and prefix not in stop:
                hits.append(code)
                break
        else:
            # no break
            pass
        if hits and hits[-1] == code:
            continue

        # Token-based match: allow "aba" to match "ABA Bank".
        bank_tokens = [tok for tok in re.findall(r"\b[a-z0-9]+\b", name_l) if tok and tok not in stop]
        if not bank_tokens:
            continue

        # Acronym match: "Phnom Penh Commercial Bank" -> "ppcb"; user may type "ppc".
        acronym = "".join([tok[0] for tok in bank_tokens if tok])
        if acronym:
            if acronym in user_tokens:
                hits.append(code)
                continue
            if len(acronym) >= 3 and acronym[:3] in user_tokens:
                hits.append(code)
                continue

        # Require at least one meaningful token overlap.
        overlap = [tok for tok in bank_tokens if tok in user_tokens]
        if not overlap:
            continue

        # Prefer matches on the first token or any token length>=3.
        if bank_tokens[0] in overlap or any(len(tok) >= 3 for tok in overlap):
            hits.append(code)

    # de-dup while keeping order
    deduped: List[str] = []
    for c in hits:
        if c not in deduped:
            deduped.append(c)
    
    # Store in cache (limit cache size to prevent memory issues)
    if len(_BANK_CODE_LOOKUP_CACHE) < 1000:
        _BANK_CODE_LOOKUP_CACHE[cache_key] = deduped.copy()
    
    return deduped


def parse_currency_hint(text: str) -> Optional[str]:
    t = (text or "").lower()
    if "usd" in t or "$" in t:
        return "USD"
    if any(k in t for k in ["khr", "riel", "riels"]):
        return "KHR"
    return None


def split_user_question(text: str) -> List[str]:
    """Split a single user message into multiple sub-questions (Approach B).

    We split on common conjunctions and separators. This is heuristic.
    """
    raw = (text or "").strip()
    if not raw:
        return []

    # Split on question marks first to respect multi-question prompts.
    chunks: List[str] = []
    for part in re.split(r"\?+", raw):
        part = part.strip()
        if part:
            chunks.append(part)

    out: List[str] = []
    for c in chunks:
        # Then split on conjunction-like separators.
        pieces = re.split(r"\s*(?:\band\b|\balso\b|\bplus\b|&|;|\n)\s*", c, flags=re.IGNORECASE)
        for p in pieces:
            p = p.strip(" \t\r\n.,")
            if p:
                out.append(p)
    return out or [raw]


def _is_fx_roundtrip_text(text: str) -> bool:
    t = (text or "").lower()
    if "usd" not in t or "khr" not in t:
        return False
    # Look for explicit sequencing / reversal language.
    return any(k in t for k in [
        "then",
        "after",
        "buy usd",
        "buy back",
        "convert back",
        "exchange back",
        "again to usd",
        "back to usd",
    ]) and ("to khr" in t or "in khr" in t) and ("to usd" in t or "buy usd" in t or "back to usd" in t)


def _llm_plan_tasks(user_text: str, max_banks: int = 40) -> List[Dict[str, Any]]:
    """Use the LLM to identify which table(s) to query and which bank(s) apply.

    Output schema (JSON):
      {"tasks": [{"intent": "LOAN", "question": "...", "bank_codes": ["ABABANK"], "currency": "USD"}, ...]}

    If parsing fails, caller should fall back to heuristic routing.
    """
    banks = _get_bank_master()
    bank_hint = [{"bank_code": b.get("bank_code"), "bank_name": b.get("bank_name")} for b in banks[:max_banks]]

    schema = {
        "tasks": [
            {
                "intent": "FX | FIXED_DEPOSIT | SAVINGS | INTEREST | LOAN | CREDIT_CARD | BANK | GENERAL",
                "question": "string",
                "bank_codes": "optional list of strings",
                "currency": "optional (USD|KHR)",
            }
        ]
    }

    prompt = f"""
You are a query router for a small banking CLI.

Goal:
- Read the user's input.
- Split it into 1+ tasks if it contains multiple questions.
- For each task, choose ONE intent from: {sorted(ALLOWED_INTENTS)}
- If the user mentions a bank, map it to bank_codes using the bank master list.

Collections meaning:
- FX -> exchange_rates
- FIXED_DEPOSIT -> fixed_deposits
- SAVINGS/INTEREST -> savings_accounts
- LOAN -> loan
- CREDIT_CARD -> credit_cards
- BANK -> bank_code master

Bank master (bank_code -> bank_name) sample:
{json.dumps(bank_hint, ensure_ascii=False)}

Rules:
- Return ONLY valid JSON. No markdown, no commentary.
- Output must match this schema (example structure):
{json.dumps(schema)}
- If the question is NOT about banking products, rates, or financial data (e.g., greetings, general knowledge, math, unrelated topics), set intent to GENERAL and omit bank_codes.
- If unsure, set intent to GENERAL.
- If no bank is mentioned, omit bank_codes.
- If the user describes a sequence like "USD to KHR then buy USD" (round-trip FX), keep it as ONE task with intent FX (do not split).
- CRITICAL: The "question" field MUST be an EXACT copy of the user's words (verbatim substring). NEVER paraphrase, summarize, or rewrite the user's question. Copy it word-for-word.
- CRITICAL: Map bank names accurately using the provided bank master list. If you're unsure about a bank, omit bank_codes rather than guessing.

User input:
{user_text}
"""

    # Use a more structured approach with lower temperature for JSON parsing
    raw = ollama_chat(
        [
            {"role": "system", "content": "You are a precise JSON generator. Return ONLY valid JSON with no additional text. Temperature should be low for accuracy."},
            {"role": "user", "content": prompt},
        ]
    )
    # Sometimes models wrap JSON in text; extract the first JSON object.
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        raise ValueError("Router did not return JSON")
    obj = json.loads(m.group(0))
    tasks = obj.get("tasks") if isinstance(obj, dict) else None
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("Router returned no tasks")

    cleaned: List[Dict[str, Any]] = []
    for t in tasks:
        if not isinstance(t, dict):
            continue
        intent = str(t.get("intent") or "GENERAL").strip().upper()
        if intent not in ALLOWED_INTENTS:
            intent = "GENERAL"
        q = str(t.get("question") or user_text).strip()
        if not q:
            q = user_text
        bank_codes = t.get("bank_codes")
        if isinstance(bank_codes, list):
            bank_codes = [str(x).strip().upper() for x in bank_codes if isinstance(x, (str, int))]
            bank_codes = [x for x in bank_codes if x]
        else:
            bank_codes = None
        ccy = t.get("currency")
        if isinstance(ccy, str):
            ccy = ccy.strip().upper()
            if ccy not in {"USD", "KHR"}:
                ccy = None
        else:
            ccy = None

        cleaned.append({"intent": intent, "question": q, "bank_codes": bank_codes, "currency": ccy})

    # De-dup tasks while preserving order.
    deduped: List[Dict[str, Any]] = []
    seen: set = set()
    for t in cleaned:
        key = (t.get("intent"), t.get("question"), tuple(t.get("bank_codes") or []), t.get("currency"))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(t)
    return deduped

def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    parts: List[str] = []
    for m in messages:
        role = (m.get("role") or "user").strip().lower()
        content = (m.get("content") or "").strip()
        if not content:
            continue
        if role == "system":
            parts.append(f"System: {content}")
        elif role == "assistant":
            parts.append(f"Assistant: {content}")
        else:
            parts.append(f"User: {content}")
    parts.append("Assistant:")
    return "\n\n".join(parts)


def _probe_endpoints(base_url: str) -> Dict[str, Any]:
    """Best-effort probes to help users diagnose wrong base URLs/ports."""
    probes = [
        ("GET", f"{base_url}/api/version"),
        ("GET", f"{base_url}/api/tags"),
        ("GET", f"{base_url}/v1/models"),
        ("GET", f"{base_url}/api/v1/settings"),
    ]
    results: Dict[str, Any] = {"base_url": base_url, "probes": []}
    for method, url in probes:
        try:
            r = requests.request(method, url, timeout=5)
            entry: Dict[str, Any] = {"method": method, "url": url, "status": r.status_code}
            ctype = (r.headers.get("content-type") or "").split(";")[0].strip()
            if ctype:
                entry["content_type"] = ctype
            # include a tiny hint only; don't dump whole HTML pages
            hint = ""
            if ctype == "application/json":
                try:
                    j = r.json()
                    if isinstance(j, dict):
                        hint = ",".join(sorted([k for k in j.keys()])[:8])
                except Exception:
                    pass
            else:
                hint = (r.text or "").strip().replace("\n", " ")[:120]
            if hint:
                entry["hint"] = hint
            results["probes"].append(entry)
        except Exception as e:
            results["probes"].append({"method": method, "url": url, "error": str(e)})
    return results


def _format_plain_ranked_table(rows: List[Dict[str, Any]], title: str, limit: int = 5) -> str:
    if not rows:
        return f"{title}: Data not available."

    lines: List[str] = [title]
    for i, r in enumerate(rows[:limit], start=1):
        bank = r.get("bank") or r.get("bank_name") or "UNKNOWN"
        product = r.get("product") or r.get("product_name") or ""
        currency = r.get("currency") or ""
        term = r.get("term_months")
        payment = r.get("interest_payment")
        rate = r.get("annual_rate_pct")
        total = r.get("total")
        interest = r.get("interest")
        channel = r.get("channel_key")
        bits = [
            f"{i}. {bank}",
            f"{product}" if product else None,
            f"{currency}" if currency else None,
            f"{term} months" if term is not None else None,
            f"{payment}" if payment else None,
            f"rate={rate}%" if rate is not None else None,
            f"interest={interest}" if interest is not None else None,
            f"total={total}" if total is not None else None,
            f"channel={channel}" if channel else None,
        ]
        lines.append(" - ".join([b for b in bits if b]))
    return "\n".join(lines)


def ollama_chat(messages: List[Dict[str, str]]) -> str:
    # Convert message list to a single prompt compatible with Ollama's /api/generate
    prompt = _messages_to_prompt(messages)

    model_name = MODEL
    stream_enabled = os.getenv("OLLAMA_STREAM", "false").strip().lower() in {"1", "true", "yes", "y"}
    timeout_seconds = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "180").strip() or "180")

    # If the user didn't specify a model, prefer the first installed model.
    # This avoids hard-coded defaults breaking when not pulled locally.
    if "OLLAMA_MODEL" not in os.environ:
        try:
            t = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=min(10, timeout_seconds))
            if t.ok:
                tags = t.json()
                models = tags.get("models") if isinstance(tags, dict) else None
                if isinstance(models, list) and models:
                    first_model = models[0].get("model") or models[0].get("name")
                    if isinstance(first_model, str) and first_model.strip():
                        model_name = first_model.strip()
        except Exception:
            pass

    def _post(url: str, payload: Dict[str, Any]) -> requests.Response:
        return requests.post(url, json=payload, timeout=timeout_seconds)

    def _is_model_not_found(resp: requests.Response) -> bool:
        # Ollama sometimes returns HTTP 404 with a JSON body like:
        #   {"error": "model 'xxx' not found"}
        # This is NOT an endpoint problem.
        try:
            j = resp.json()
        except Exception:
            return False
        if not isinstance(j, dict):
            return False
        err = j.get("error")
        if not isinstance(err, str):
            return False
        low = err.lower()
        return ("not found" in low) and ("model" in low)

    def _call_generate(mname: str) -> requests.Response:
        payload = {
            "model": mname,
            "prompt": prompt,
            "temperature": 0.3,  # Increased from 0.2 for better conversational flow
            "max_tokens": 1024,  # Increased from 256 for comprehensive answers
            # If true, Ollama returns newline-delimited JSON objects.
            # If false, Ollama returns a single JSON object.
            "stream": stream_enabled,
        }
        return _post(f"{OLLAMA_BASE_URL}/api/generate", payload)

    def _call_chat(mname: str) -> requests.Response:
        payload = {
            "model": mname,
            "messages": messages,
            "stream": stream_enabled,
            "options": {
                "temperature": 0.3,  # Increased for better responses
                "num_predict": 1024,  # Limit response length
            },
        }
        return _post(f"{OLLAMA_BASE_URL}/api/chat", payload)

    def _call_openai_chat(mname: str) -> requests.Response:
        payload = {
            "model": mname,
            "messages": messages,
            "temperature": 0.3,  # Increased for better responses
            "max_tokens": 1024,  # Increased for comprehensive answers
            "stream": stream_enabled,
        }
        return _post(f"{OLLAMA_BASE_URL}/v1/chat/completions", payload)

    r = _call_generate(model_name)

    # If /api/generate is not available (404), try alternate endpoints.
    # BUT: Ollama may also return 404 when the model isn't installed.
    if r.status_code == 404 and not _is_model_not_found(r):
        # Some setups only expose /api/chat, or OpenAI-compatible /v1/chat/completions.
        r2 = _call_chat(model_name)
        if r2.status_code != 404 or _is_model_not_found(r2):
            r = r2
        else:
            r = _call_openai_chat(model_name)
    # If model not found, try to pick the first available model from /api/tags and retry once
    try:
        data = r.json()
    except Exception:
        data = None

    if isinstance(data, dict) and data.get("error") and "not found" in str(data.get("error")).lower():
        try:
            t = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
            t.raise_for_status()
            tags = t.json()
            models = tags.get("models") if isinstance(tags, dict) else None
            if isinstance(models, list) and models:
                first_model = models[0].get("model") or models[0].get("name")
                if first_model:
                    model_name = first_model
                    # Retry against the same endpoint family we already landed on.
                    if "/v1/chat/completions" in r.url:
                        r = _call_openai_chat(model_name)
                    elif r.url.endswith("/api/chat"):
                        r = _call_chat(model_name)
                    else:
                        r = _call_generate(model_name)
        except Exception:
            pass

    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        if r.status_code == 404:
            if _is_model_not_found(r):
                try:
                    err = r.json().get("error")
                except Exception:
                    err = None
                msg = err if isinstance(err, str) and err else "Requested model not found"
                raise RuntimeError(
                    msg
                    + ". Set OLLAMA_MODEL to an installed model (see /api/tags), or run `ollama pull <model>`."
                ) from e

            diag = _probe_endpoints(OLLAMA_BASE_URL)
            snippet = (r.text or "").strip().replace("\n", " ")[:200]
            raise RuntimeError(
                "Ollama endpoint not found (404). "
                "This usually means OLLAMA_BASE_URL is pointing to a service that is not Ollama, "
                "or Ollama is not running on that port. "
                f"URL={r.url}. Response='{snippet}'. Diagnostics: {json.dumps(diag, indent=2)}"
            ) from e
        raise

    # Try to parse JSON (non-stream mode returns a single JSON object)
    try:
        data = r.json()
    except Exception:
        data = None

    # Ollama /api/generate canonical response: {"response": "...", "done": true, ...}
    if isinstance(data, dict):
        if isinstance(data.get("error"), str) and data.get("error"):
            raise RuntimeError(data["error"])

        # OpenAI-compatible response
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                msg = first.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str) and content.strip():
                        return content.strip()

        resp = data.get("response")
        if isinstance(resp, str) and resp.strip():
            return resp.strip()

        # Be resilient to other common shapes
        msg = data.get("message")
        if isinstance(msg, dict):
            content = msg.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()

        if isinstance(choices, list) and choices:
            first = choices[0]
            if isinstance(first, dict):
                text = first.get("text") or first.get("content")
                if isinstance(text, str) and text.strip():
                    return text.strip()
            if isinstance(first, str) and first.strip():
                return first.strip()

        for key in ("text", "result", "output"):
            v = data.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()

        res = data.get("result")
        if isinstance(res, dict):
            out = res.get("output") or res.get("text")
            if isinstance(out, str) and out.strip():
                return out.strip()

    # Stream mode (or some proxy setups) returns newline-delimited JSON.
    # When that happens, requests.json() fails and r.text is many JSON objects.
    body = (r.text or "").strip()
    if body:
        parts: List[str] = []
        saw_json = False
        for line in body.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                saw_json = True
            except Exception:
                continue

            if isinstance(obj, dict):
                chunk = obj.get("response")
                if isinstance(chunk, str) and chunk:
                    parts.append(chunk)
                # Some APIs use {"message": {"content": "..."}}
                msg = obj.get("message")
                if isinstance(msg, dict):
                    content = msg.get("content")
                    if isinstance(content, str) and content:
                        parts.append(content)

                if obj.get("done") is True:
                    break

        if saw_json and parts:
            return "".join(parts).strip()

        # If it wasn't NDJSON (or couldn't be parsed), return raw text.
        return body

    raise ValueError("Unexpected Ollama response format")



def stop_ollama() -> None:
    """Best-effort stop.

    On Windows, uses taskkill. On Unix-like systems, uses pkill.
    If the command is missing/fails, we just continue.
    """
    print("\nStopping Ollama...")
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/IM", "ollama.exe", "/F"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        else:
            subprocess.run(
                ["pkill", "-f", "ollama"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
    except Exception:
        pass
    print("Ollama stop requested.")


def serialize_for_json(obj: Any) -> Any:
    """Convert datetime/ObjectId-like values to JSON-safe values."""
    if isinstance(obj, list):
        return [serialize_for_json(x) for x in obj]
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if hasattr(v, "isoformat"):  # datetime
                out[k] = v.isoformat()
            elif hasattr(v, "binary") and hasattr(v, "generation_time"):
                # bson ObjectId-like
                out[k] = str(v)
            else:
                out[k] = serialize_for_json(v)
        return out
    return obj


def parse_amount_currency(text: str) -> Tuple[float, str]:
    """
    Extract amount + currency from question.
    Examples:
      - "I have 1000 USD" -> (1000, "USD")
      - "convert 2,500 usd to khr" -> (2500, "USD")
    Defaults: (1000, "USD")
    """
    t = (text or "").lower().replace(",", "")

    # Prefer explicit currency patterns.
    # "$1000" or "1000 usd" or "usd 1000"
    m = re.search(r"\$\s*(\d+(?:\.\d+)?)", t)
    if m:
        return float(m.group(1)), "USD"

    m = re.search(r"(\d+(?:\.\d+)?)\s*(usd|khr|riel|riels)\b", t)
    if m:
        amt = float(m.group(1))
        c_raw = m.group(2)
        c = c_raw.upper()
        if c in ("RIEL", "RIELS"):
            c = "KHR"
        return amt, c

    m = re.search(r"\b(usd|khr|riel|riels)\b\s*(\d+(?:\.\d+)?)", t)
    if m:
        c_raw = m.group(1)
        amt = float(m.group(2))
        c = c_raw.upper()
        if c in ("RIEL", "RIELS"):
            c = "KHR"
        return amt, c

    # If currency isn't explicit, avoid treating tenor (e.g. "3 years") as amount.
    # Only accept a bare number as an amount if it is NOT directly tied to a time unit.
    m = re.search(r"\b(\d+(?:\.\d+)?)\b(?!\s*(?:year|years|yr|month|months|mo)\b)", t)
    if m:
        try:
            return float(m.group(1)), "USD"
        except Exception:
            pass

    return 1000.0, "USD"


def parse_term_months(text: str, default_months: int = 12) -> int:
    t = text.lower()
    # explicit months
    m = re.search(r"(\d+)\s*(month|months|mo)\b", t)
    if m:
        return int(m.group(1))
    # years
    y = re.search(r"(\d+)\s*(year|years|yr)\b", t)
    if y:
        return int(y.group(1)) * 12
    if "one year" in t or "1 year" in t:
        return 12
    if "six month" in t or "6 month" in t:
        return 6
    return default_months


def parse_interest_payment(text: str) -> str:
    """Return either 'monthly' or 'at_maturity'."""
    t = text.lower()
    if "monthly" in t:
        return "monthly"
    if "maturity" in t or "at maturity" in t:
        return "at_maturity"
    # safest default for fixed deposits is usually at maturity, but we still treat missing as a default
    return "at_maturity"


def parse_fd_channel_keys(text: str) -> Optional[List[str]]:
    """Map user words to MongoDB rate-channel keys."""
    t = text.lower()
    keys: List[str] = []
    if any(w in t for w in ["mobile", "app", "aba mobile", "digital"]):
        keys.append("aba_mobile")
    if any(w in t for w in ["branch", "over the counter", "counter", "otc"]):
        keys.append("over_the_counter")
    return keys or None


def classify_intents(text: str) -> List[str]:
    """Return one or more intents inferred from the text.

    This is a heuristic router (Approach B). Downstream code executes one task per intent.
    """
    t = (text or "").lower()
    intents: List[str] = []

    # Loans / credit cards
    # Treat vehicle terms as loan intents even if the user omits the word "loan".
    if any(
        w in t
        for w in [
            "loan",
            "mortgage",
            "home loan",
            "housing",
            "car loan",
            "car financing",
            "auto loan",
            "vehicle loan",
            "motorbike",
            "motorcycle",
            "moto",
            "bike financing",
            "hire purchase",
            "personal loan",
        ]
    ):
        intents.append("LOAN")
    if any(w in t for w in ["credit card", "creditcard", "card", "visa", "mastercard", "platinum", "gold"]):
        intents.append("CREDIT_CARD")

    # Deposits / savings
    if any(w in t for w in ["fixed", "term deposit", "fd", "maturity"]):
        intents.append("FIXED_DEPOSIT")
    if any(w in t for w in ["saving", "savings", "saver"]):
        intents.append("SAVINGS")
    if "interest" in t and "FIXED_DEPOSIT" not in intents and "SAVINGS" not in intents:
        intents.append("INTEREST")

    # FX
    if any(w in t for w in ["exchange", "convert", "fx", "rate", "buy", "sell"]):
        # Avoid incorrectly classifying "interest rate" as FX.
        if "interest" not in t and "loan" not in t:
            intents.append("FX")

    # Bank metadata queries
    if any(w in t for w in ["bank list", "which banks", "bank code", "bankcode"]):
        intents.append("BANK")

    # De-dup preserving order
    deduped: List[str] = []
    for it in intents:
        if it not in deduped:
            deduped.append(it)
    return deduped or ["GENERAL"]


def _format_plain_list(rows: List[Dict[str, Any]], title: str, keys: List[str], limit: int = 5) -> str:
    if not rows:
        return f"{title}: Data not available."
    lines = [title]
    for i, r in enumerate(rows[:limit], start=1):
        bits: List[str] = [f"{i}."]
        for k in keys:
            v = r.get(k)
            if v is None or v == "":
                continue
            bits.append(f"{k}={v}")
        lines.append(" ".join(bits))
    return "\n".join(lines)



# ===============================
# Mongo Queries
# ===============================
def get_latest_rates(currency_pair: str = "USD-KHR", rate_type: str = "counter") -> List[Dict]:
    """Return latest FX rate per bank for a currency pair.

    Supports two schemas:
    1) Legacy: {currency_pair, rate_type, buy_rate, sell_rate, scraped_at}
    2) Current: {rate_type, status, as_of, base_currency, rates: [{pair:"USD/KHR", buy, sell}, ...]}
    """

    # --- Schema 1 (legacy) ---
    pipeline_legacy = [
        {"$match": {"currency_pair": currency_pair, "rate_type": rate_type}},
        {"$sort": {"scraped_at": -1}},
        {
            "$group": {
                "_id": {"$ifNull": ["$bank_code", "$bank_name"]},
                "bank_code": {"$first": "$bank_code"},
                "bank_name": {"$first": "$bank_name"},
                "buy_rate": {"$first": "$buy_rate"},
                "sell_rate": {"$first": "$sell_rate"},
                "as_of": {"$first": "$scraped_at"},
            }
        },
        {"$project": {"_id": 0, "bank_code": 1, "bank_name": 1, "buy_rate": 1, "sell_rate": 1, "as_of": 1}},
    ]

    try:
        legacy = list(db[COLL_EXCHANGE].aggregate(pipeline_legacy))
    except Exception:
        legacy = []
    if legacy:
        return legacy

    # --- Schema 2 (current) ---
    # Convert "USD-KHR" -> "USD/KHR".
    pair_slash = currency_pair.replace("-", "/")

    pipeline_current = [
        {
            "$match": {
                "rate_type": {"$in": [rate_type, "spot_exchange_rate", "counter", "branch", "official_branch_rate"]},
                "status": {"$in": ["ACTIVE", "active", True]},
                "rates": {"$type": "array"},
            }
        },
        {"$sort": {"as_of": -1}},
        {
            "$addFields": {
                "_pair": {
                    "$arrayElemAt": [
                        {
                            "$filter": {
                                "input": "$rates",
                                "as": "r",
                                "cond": {"$eq": ["$$r.pair", pair_slash]},
                            }
                        },
                        0,
                    ]
                }
            }
        },
        # Drop docs where the requested pair isn't present.
        {"$match": {"_pair": {"$ne": None}}},
        {
            "$group": {
                "_id": {"$ifNull": ["$bank_code", "$bank_name"]},
                "bank_code": {"$first": "$bank_code"},
                "bank_name": {"$first": "$bank_name"},
                "buy_rate": {"$first": "$_pair.buy"},
                "sell_rate": {"$first": "$_pair.sell"},
                "as_of": {"$first": "$as_of"},
                "rate_type": {"$first": "$rate_type"},
                "source_channel": {"$first": "$source_channel"},
            }
        },
        {"$project": {"_id": 0, "bank_code": 1, "bank_name": 1, "buy_rate": 1, "sell_rate": 1, "as_of": 1, "rate_type": 1, "source_channel": 1}},
    ]

    return list(db[COLL_EXCHANGE].aggregate(pipeline_current))


def get_savings_accounts(
    bank_codes: Optional[List[str]] = None,
    currency: Optional[str] = None,
    limit: int = 50,
) -> List[Dict]:
    """Fetch savings/CASA products.

    Notes:
    - The `savings_accounts` collection may not have a top-level `currency` field.
      Many docs express currency inside `interest_rates` or `interest_rate` maps.
    - For savings listing, currency is usually optional.
    """
    query: Dict[str, Any] = {}

    if bank_codes:
        query["bank_code"] = {"$in": [c.strip().upper() for c in bank_codes if c and str(c).strip()]}

    if currency:
        ccy = currency.strip().upper()
        # Match either legacy top-level currency or presence of currency key in interest maps.
        query["$or"] = [
            {"currency": {"$in": [ccy, ccy.lower()]}},
            {f"interest_rates.{ccy}": {"$exists": True}},
            {f"interest_rate.{ccy}": {"$exists": True}},
            {f"interest_rate.{ccy.lower()}": {"$exists": True}},
        ]

    cursor = db[COLL_SAVINGS].find(query, {"_id": 0}).limit(limit)
    return list(cursor)


def get_loans(
    bank_codes: Optional[List[str]] = None,
    bank_code_prefixes: Optional[List[str]] = None,
    currency: Optional[str] = None,
    limit: int = 50,
) -> List[Dict]:
    """Fetch loan products, optionally filtered by bank_code and currency."""
    # Do not over-filter by product_type; different sources use values like
    # "loan", "home_loan", "mortgage_loan", etc. Treat the collection as loans.
    query: Dict[str, Any] = {}

    bank_or: List[Dict[str, Any]] = []
    if bank_codes:
        bank_or.append({"bank_code": {"$in": bank_codes}})
    if bank_code_prefixes:
        # Prefix fallback for inconsistent bank_code variants (e.g. PPCBBANK vs PPCBANK).
        # Use anchored regex so it's still index-friendly if an index exists on bank_code.
        for p in bank_code_prefixes:
            p = (p or "").strip().upper()
            if p:
                bank_or.append({"bank_code": {"$regex": f"^{re.escape(p)}", "$options": ""}})
    if bank_or:
        query["$or"] = bank_or
    if currency:
        query["currency"] = {"$in": [currency.upper(), currency.lower()]}
    cursor = db[COLL_LOAN].find(query, {"_id": 0}).limit(limit)
    return list(cursor)


def get_credit_cards(
    bank_codes: Optional[List[str]] = None,
    limit: int = 50,
) -> List[Dict]:
    """Fetch credit card products, optionally filtered by bank_code."""
    query: Dict[str, Any] = {}
    if bank_codes:
        query["bank_code"] = {"$in": bank_codes}
    cursor = db[COLL_CREDIT_CARDS].find(query, {"_id": 0}).limit(limit)
    return list(cursor)


def answer_loan_question(user_question: str, bank_codes: Optional[List[str]] = None, currency: Optional[str] = None) -> str:
    bank_codes = bank_codes or extract_bank_codes_from_text(user_question)
    currency = currency or parse_currency_hint(user_question)

    def _norm_text(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (dict, list)):
            try:
                return json.dumps(v, ensure_ascii=False, default=str)
            except Exception:
                return str(v)
        return str(v)

    def _loan_field(doc: Dict[str, Any], *keys: str) -> str:
        for k in keys:
            if k in doc and doc.get(k) not in (None, ""):
                return _norm_text(doc.get(k)).strip()
        return ""

    def _loan_haystack(doc: Dict[str, Any]) -> str:
        # Support both schemas:
        # - loan_name/loan_category/... (older)
        # - product_name/product_category/product_subcategory/... (newer)
        parts = [
            _loan_field(doc, "bank_name", "bank"),
            _loan_field(doc, "bank_code"),
            _loan_field(doc, "loan_name", "product_name"),
            _loan_field(doc, "loan_category", "product_category"),
            _loan_field(doc, "product_subcategory", "loan_subcategory"),
            _loan_field(doc, "description"),
        ]
        return " ".join([p for p in parts if p]).lower()

    def _is_unsecured_loan(doc: Dict[str, Any]) -> bool:
        sub = _loan_field(doc, "product_subcategory", "loan_subcategory").lower()
        if sub in {"unsecured", "no collateral", "without collateral", "non-collateral", "non collateral"}:
            return True
        hay = _loan_haystack(doc)
        return any(
            k in hay
            for k in [
                "unsecured",
                "no collateral",
                "without collateral",
                "non-collateral",
                "non collateral",
            ]
        )

    def _is_salary_loan(doc: Dict[str, Any]) -> bool:
        hay = _loan_haystack(doc)
        return "salary" in hay

    def _is_housing_loan(doc: Dict[str, Any]) -> bool:
        hay = _loan_haystack(doc)
        # Keep it simple and grounded in text fields.
        return any(k in hay for k in ["housing loan", "housing", "home loan", "home", "mortgage"])

    def _extract_product_keywords(text: str) -> List[str]:
        t = (text or "").lower()
        # Normalize some variants.
        mapping = {
            "motor bike": "motorbike",
            "motor-cycle": "motorcycle",
        }
        for a, b in mapping.items():
            t = t.replace(a, b)

        keys: List[str] = []
        # Vehicle-related products
        if any(k in t for k in ["motorbike", "motorcycle", "bike loan", "motor loan"]):
            keys.extend(["motorbike", "motorcycle"])
        # Only include the short token "moto" when the user actually typed it
        # (prevents accidental matching against unrelated words like "automotive").
        if re.search(r"\bmoto\b", t):
            keys.append("moto")
        if any(k in t for k in ["car loan", "auto loan", "vehicle loan", "car financing"]):
            keys.extend(["car", "auto", "vehicle"])

        # Digital loans
        if "digital loan" in t:
            keys.append("digital")
            keys.append("digital loan")
        elif "digital" in t and "loan" in t:
            keys.append("digital")

        # Secured/unsecured / collateral cues (common in product_subcategory).
        if re.search(r"\bunsecured\b", t):
            keys.append("unsecured")
        if re.search(r"\bsecured\b", t):
            keys.append("secured")
        if re.search(r"\bcollateral\b", t):
            keys.append("collateral")
        # Common loan categories
        if any(k in t for k in ["salary loan", "salary"]):
            keys.append("salary")
        if "personal" in t:
            keys.append("personal")
        if any(k in t for k in ["housing", "home", "mortgage"]):
            keys.extend(["housing", "home", "mortgage"])

        # De-dup preserving order
        out: List[str] = []
        for k in keys:
            if k not in out:
                out.append(k)
        return out

    def _loan_product_blob(doc: Dict[str, Any]) -> str:
        return " ".join(
            [
                _loan_field(doc, "product_name", "loan_name"),
                _loan_field(doc, "product_category", "loan_category"),
                _loan_field(doc, "product_subcategory", "loan_subcategory"),
                _loan_field(doc, "product_slug"),
            ]
        ).lower()

    def _infer_product_query_tokens(question: str) -> List[str]:
        t = (question or "").lower()
        toks = re.findall(r"\b[a-z0-9]+\b", t)
        # Remove very common words to avoid accidental matches.
        stop = {
            "a",
            "an",
            "the",
            "and",
            "or",
            "to",
            "of",
            "for",
            "in",
            "on",
            "at",
            "with",
            "without",
            "about",
            "me",
            "you",
            "please",
            "can",
            "could",
            "would",
            "help",
            "tell",
            "brief",
            "summarize",
            "summary",
            "overview",
            "explain",
            "detail",
            "details",
            "product",
            "products",
            "loan",
            "loans",
            "bank",
            "banks",
            "do",
            "does",
            "have",
            "has",
            "list",
            "show",
            "all",
            "more",
        }
        out: List[str] = []
        for tok in toks:
            if tok in stop:
                continue
            # Skip very short tokens (often noise)
            if len(tok) <= 2:
                continue
            if tok not in out:
                out.append(tok)
        return out

    def _best_match_by_tokens(docs: List[Dict[str, Any]], query_tokens: List[str]) -> Tuple[Optional[Dict[str, Any]], int]:
        if not docs or not query_tokens:
            return None, 0

        def score(doc: Dict[str, Any]) -> int:
            # Heavier weight for product-name/slug matches (more reliable than category text).
            name_blob = " ".join(
                [
                    _loan_field(doc, "product_name", "loan_name"),
                    _loan_field(doc, "product_slug"),
                ]
            ).lower()
            meta_blob = " ".join(
                [
                    _loan_field(doc, "product_category", "loan_category"),
                    _loan_field(doc, "product_subcategory", "loan_subcategory"),
                ]
            ).lower()

            s = 0
            for tok in query_tokens:
                if re.search(rf"\b{re.escape(tok)}\b", name_blob):
                    s += 3
                elif tok in name_blob:
                    s += 2
                elif re.search(rf"\b{re.escape(tok)}\b", meta_blob):
                    s += 1
                elif tok in meta_blob:
                    s += 1

            # Strong boost if the full product name appears as a substring in the question.
            pname = _loan_field(doc, "product_name", "loan_name").lower()
            if pname and pname in (user_question or "").lower():
                s += 5
            return s

        best_doc: Optional[Dict[str, Any]] = None
        best_score = 0
        for d in docs:
            s = score(d)
            if s > best_score:
                best_score = s
                best_doc = d
        return best_doc, best_score

    def _deterministic_loan_summary(doc: Dict[str, Any]) -> str:
        bank_name = _loan_field(doc, "bank_name", "bank") or "UNKNOWN"
        bank_code = _loan_field(doc, "bank_code") or ""
        pname = _loan_field(doc, "product_name", "loan_name") or "(unnamed product)"
        ptype = _loan_field(doc, "product_type") or _loan_field(doc, "loan_type")
        pcat = _loan_field(doc, "product_category", "loan_category")
        psub = _loan_field(doc, "product_subcategory", "loan_subcategory")
        target = _loan_field(doc, "target_customer")

        loan_currency = doc.get("loan_currency") or doc.get("loan_currencies")
        if isinstance(loan_currency, list):
            loan_currency = [str(x).upper() for x in loan_currency if x]
        loan_amount = doc.get("loan_amount")
        max_usd = None
        if isinstance(loan_amount, dict):
            max_usd = loan_amount.get("maximum_usd") or loan_amount.get("max_usd")

        collateral = _loan_field(doc, "collateral")
        guarantor = _loan_field(doc, "guarantor")
        approval = doc.get("approval")
        approval_time = None
        if isinstance(approval, dict):
            approval_time = approval.get("timeframe") or approval.get("type")
        application = doc.get("application")
        app_channel = None
        app_avail = None
        if isinstance(application, dict):
            app_channel = application.get("channel") or application.get("method")
            app_avail = application.get("availability")

        key_benefits = doc.get("key_benefits")
        kb_list: List[str] = []
        if isinstance(key_benefits, list):
            for b in key_benefits:
                if isinstance(b, str) and b.strip():
                    kb_list.append(b.strip())

        adv = doc.get("advantages")
        adv_list: List[str] = []
        if isinstance(adv, list):
            for a in adv:
                if isinstance(a, str) and a.strip():
                    adv_list.append(a.strip())

        req = doc.get("requirements")
        req_count = len(req) if isinstance(req, list) else (len(req.keys()) if isinstance(req, dict) else None)

        lines: List[str] = []
        lines.append("Loan product (from MongoDB):")
        lines.append(f"- Bank: {bank_name}" + (f" ({bank_code})" if bank_code else ""))
        lines.append(f"- Product: {pname}")
        if ptype:
            lines.append(f"- Type: {ptype}")
        if pcat:
            lines.append(f"- Category: {pcat}")
        if psub:
            lines.append(f"- Subcategory: {psub}")
        if target:
            lines.append(f"- Target customer: {target}")
        if loan_currency:
            lines.append(f"- Currency: {', '.join(loan_currency) if isinstance(loan_currency, list) else loan_currency}")
        if max_usd is not None:
            lines.append(f"- Max amount (USD): {max_usd}")
        if collateral:
            lines.append(f"- Collateral: {collateral}")
        if guarantor:
            lines.append(f"- Guarantor: {guarantor}")
        if approval_time:
            lines.append(f"- Approval: {approval_time}")
        if app_channel:
            lines.append(f"- Application channel: {app_channel}")
        if app_avail:
            lines.append(f"- Availability: {app_avail}")
        if kb_list:
            lines.append("- Key benefits: " + "; ".join(kb_list[:6]))
        if adv_list:
            lines.append("- Advantages: " + "; ".join(adv_list[:3]))
        if req_count is not None:
            lines.append(f"- Requirements: {req_count} item(s)")
        updated = _loan_field(doc, "updated_at", "updated_date")
        if updated:
            lines.append(f"- Updated: {updated}")
        return "\n".join(lines)

    def _deterministic_loan_full_details(doc: Dict[str, Any]) -> str:
        """Return a friendly full-details view for the matched loan document.

        Goal: show *all* fields available in the MongoDB document, but formatted
        for humans (not raw JSON). Raw JSON can be requested explicitly.
        """
        bank_name = _loan_field(doc, "bank_name", "bank") or "UNKNOWN"
        bank_code = _loan_field(doc, "bank_code") or ""
        pname = _loan_field(doc, "product_name", "loan_name") or "(unnamed product)"

        # We already fetch with {"_id": 0}, but keep this defensive.
        doc_out = {k: v for k, v in doc.items() if k != "_id"}

        def _friendly_value(v: Any) -> str:
            if v is None:
                return "Data not available"
            if isinstance(v, bool):
                return "Yes" if v else "No"
            if isinstance(v, (int, float)):
                return str(v)
            if isinstance(v, str):
                s = v.strip()
                return s if s else "Data not available"
            return ""

        def _render(obj: Any, indent: str = "", max_items: int = 200) -> List[str]:
            lines: List[str] = []
            if isinstance(obj, dict):
                for k in sorted(obj.keys(), key=lambda x: str(x)):
                    v = obj.get(k)
                    if isinstance(v, (dict, list)):
                        lines.append(f"{indent}- {k}:")
                        lines.extend(_render(v, indent + "  ", max_items=max_items))
                    else:
                        lines.append(f"{indent}- {k}: {_friendly_value(v) or _norm_text(v)}")
                return lines

            if isinstance(obj, list):
                if not obj:
                    return [f"{indent}- (empty)"]
                shown = 0
                for item in obj:
                    if shown >= max_items:
                        lines.append(f"{indent}- … ({len(obj) - shown} more item(s))")
                        break
                    if isinstance(item, dict):
                        lines.append(f"{indent}- item:")
                        lines.extend(_render(item, indent + "  ", max_items=max_items))
                    elif isinstance(item, list):
                        lines.append(f"{indent}- item:")
                        lines.extend(_render(item, indent + "  ", max_items=max_items))
                    else:
                        lines.append(f"{indent}- {_friendly_value(item) or _norm_text(item)}")
                    shown += 1
                return lines

            # Primitive fallback
            return [f"{indent}- {_friendly_value(obj) or _norm_text(obj)}"]

        used_keys = {
            "bank_name",
            "bank",
            "bank_code",
            "product_name",
            "loan_name",
            "product_type",
            "loan_type",
            "product_category",
            "loan_category",
            "product_subcategory",
            "loan_subcategory",
            "product_slug",
            "created_at",
            "updated_at",
            "updated_date",
            "created_date",
            "description",
            "loan_terms",
            "eligibility",
            "application",
            "metadata",
        }

        ptype = _loan_field(doc, "product_type", "loan_type")
        pcat = _loan_field(doc, "product_category", "loan_category")
        psub = _loan_field(doc, "product_subcategory", "loan_subcategory")
        pslug = _loan_field(doc, "product_slug")
        created = _loan_field(doc, "created_at", "created_date")
        updated = _loan_field(doc, "updated_at", "updated_date")

        lines: List[str] = []
        lines.append("Loan product (full details from MongoDB):")
        lines.append(f"- Bank: {bank_name}" + (f" ({bank_code})" if bank_code else ""))
        lines.append(f"- Product: {pname}")
        if ptype:
            lines.append(f"- Type: {ptype}")
        if pcat:
            lines.append(f"- Category: {pcat}")
        if psub:
            lines.append(f"- Subcategory: {psub}")
        if pslug:
            lines.append(f"- Slug: {pslug}")
        if created:
            lines.append(f"- Created: {created}")
        if updated:
            lines.append(f"- Updated: {updated}")

        # Primary structured sections (if present)
        for section_key, section_title in [
            ("description", "Description"),
            ("loan_terms", "Loan Terms"),
            ("eligibility", "Eligibility"),
            ("application", "Application"),
            ("metadata", "Metadata"),
        ]:
            if section_key in doc_out and doc_out.get(section_key) not in (None, ""):
                lines.append(f"\n{section_title}:")
                lines.extend(_render(doc_out.get(section_key), indent=""))

        # Any remaining fields that weren't covered above.
        remaining = {k: v for k, v in doc_out.items() if k not in used_keys}
        if remaining:
            lines.append("\nAdditional fields:")
            lines.extend(_render(remaining, indent=""))

        # Only show raw JSON if the user explicitly asks.
        ql = (user_question or "").lower()
        wants_json = any(k in ql for k in ["json", "raw", "full json", "show json"]) or os.getenv("LOAN_SHOW_JSON", "").strip() in {"1", "true", "yes"}
        if wants_json:
            lines.append("\nRaw MongoDB document (JSON):")
            lines.append(json.dumps(serialize_for_json(doc_out), indent=2, ensure_ascii=False, default=str))

        return "\n".join(lines)

    def _deterministic_loan_interest_rate_only(doc: Dict[str, Any]) -> str:
        bank_name = _loan_field(doc, "bank_name", "bank") or "UNKNOWN"
        bank_code = _loan_field(doc, "bank_code") or ""
        pname = _loan_field(doc, "product_name", "loan_name") or "(unnamed product)"
        updated = _loan_field(doc, "updated_at", "updated_date")

        ir = doc.get("interest_rate")
        if ir is None or ir == {} or ir == "":
            return "\n".join(
                [
                    "Loan interest rate (from MongoDB):",
                    f"- Bank: {bank_name}" + (f" ({bank_code})" if bank_code else ""),
                    f"- Product: {pname}",
                    (f"- Updated: {updated}" if updated else ""),
                    "- Interest rate: Data not available in this dataset.",
                ]
            ).strip()

        lines: List[str] = []
        lines.append("Loan interest rate (from MongoDB):")
        lines.append(f"- Bank: {bank_name}" + (f" ({bank_code})" if bank_code else ""))
        lines.append(f"- Product: {pname}")
        if updated:
            lines.append(f"- Updated: {updated}")

        # Prefer a human-friendly render for nested interest structures.
        lines.append("\nInterest rate:")

        def _render(obj: Any, indent: str = "") -> List[str]:
            out: List[str] = []
            if isinstance(obj, dict):
                for k in sorted(obj.keys(), key=lambda x: str(x)):
                    v = obj.get(k)
                    if isinstance(v, (dict, list)):
                        out.append(f"{indent}- {k}:")
                        out.extend(_render(v, indent + "  "))
                    else:
                        s = str(v).strip() if v is not None else ""
                        out.append(f"{indent}- {k}: {s if s else 'Data not available'}")
                return out
            if isinstance(obj, list):
                if not obj:
                    return [f"{indent}- (empty)"]
                for item in obj[:50]:
                    if isinstance(item, (dict, list)):
                        out.append(f"{indent}-")
                        out.extend(_render(item, indent + "  "))
                    else:
                        s = str(item).strip() if item is not None else ""
                        out.append(f"{indent}- {s if s else 'Data not available'}")
                if len(obj) > 50:
                    out.append(f"{indent}- … ({len(obj) - 50} more item(s))")
                return out
            s = str(obj).strip() if obj is not None else ""
            return [f"{indent}- {s if s else 'Data not available'}"]

        lines.extend(_render(ir, indent=""))

        # Include a disclaimer if present, since rates often depend on conditions.
        disclaimer = _loan_field(doc, "disclaimer")
        if disclaimer:
            lines.append("\nNote:")
            lines.append(f"- {disclaimer}")

        return "\n".join(lines)

    def _matches_product_keys(doc: Dict[str, Any], keys: List[str]) -> bool:
        if not keys:
            return False
        # Focus matching on product-identifying fields to avoid noise.
        blob = _loan_product_blob(doc)
        for k in keys:
            k = (k or "").strip().lower()
            if not k:
                continue
            if re.search(rf"\b{re.escape(k)}\b", blob):
                return True
        return False

    def _prefixes_from_codes(codes: List[str]) -> List[str]:
        out: List[str] = []
        for c in codes:
            c = (c or "").strip().upper()
            if not c:
                continue
            # Prefer common Cambodian bank-code prefixes (first 3-5 chars).
            out.append(c[:5])
            out.append(c[:4])
            out.append(c[:3])
        dedup: List[str] = []
        for p in out:
            if p and p not in dedup:
                dedup.append(p)
        return dedup

    loans_raw = get_loans(bank_codes=bank_codes or None, currency=currency, limit=200)

    # If no results and we had bank_codes, retry with a prefix fallback.
    if not loans_raw and bank_codes:
        loans_raw = get_loans(
            bank_codes=None,
            bank_code_prefixes=_prefixes_from_codes(bank_codes),
            currency=currency,
            limit=200,
        )
    if not loans_raw:
        scope = "all banks" if not bank_codes else f"banks={bank_codes}"
        c = currency or "any currency"
        return f"No loan data found for {scope} ({c})."

    # Output sizing: keep answers short by default, but allow user override.
    # - Env var LOAN_TOP_N overrides default.
    # - User phrases like "show more" / "list all" increase output.
    try:
        default_top_n = int(os.getenv("LOAN_TOP_N", "5").strip() or "5")
    except Exception:
        default_top_n = 5
    default_top_n = max(1, min(default_top_n, 50))

    q_low = (user_question or "").lower()
    # If the user explicitly asks for interest rate, don't dump full product details.
    asks_interest_rate_only = ("interest rate" in q_low) or ("interest rates" in q_low)
    wants_all = any(k in q_low for k in ["list all", "show all", "all loans", "all loan", "everything"])
    wants_more = any(k in q_low for k in ["show more", "more", "top 10", "top 20", "top 25"])  # simple
    if wants_all:
        top_n = min(50, len(loans_raw))
    elif "top 10" in q_low:
        top_n = 10
    elif "top 20" in q_low:
        top_n = 20
    elif "top 25" in q_low:
        top_n = 25
    elif wants_more:
        top_n = max(default_top_n, 10)
    else:
        top_n = default_top_n

    # Deterministic: list banks that offer unsecured/no-collateral loans.
    asks_unsecured = any(k in q_low for k in ["unsecured", "no collateral", "without collateral"])
    asks_housing = any(k in q_low for k in ["housing", "home loan", "home", "mortgage"])
    asks_banks = any(
        k in q_low
        for k in [
            "which bank",
            "which banks",
            "banks who",
            "banks that",
            "bank has",
            "bank have",
            "list bank",
            "list banks",
        ]
    )

    def _group_products_by_bank(docs: List[Dict[str, Any]], title: str) -> str:
        by_bank: Dict[str, Dict[str, Any]] = {}
        for d in docs:
            bcode = _loan_field(d, "bank_code") or "UNKNOWN"
            bname = _loan_field(d, "bank_name", "bank") or bcode
            pname = _loan_field(d, "product_name", "loan_name")
            pcat = _loan_field(d, "product_category", "loan_category")
            psub = _loan_field(d, "product_subcategory", "loan_subcategory")

            key = f"{bcode}::{bname}".upper()
            if key not in by_bank:
                by_bank[key] = {"bank_code": bcode, "bank_name": bname, "products": []}
            by_bank[key]["products"].append(
                {
                    "product_name": pname,
                    "product_category": pcat,
                    "product_subcategory": psub,
                }
            )

        banks_sorted = sorted(
            by_bank.values(),
            key=lambda x: (str(x.get("bank_name") or "").lower(), str(x.get("bank_code") or "")),
        )

        lines: List[str] = [title]
        for b in banks_sorted:
            products = b.get("products") or []
            seen_prod: set = set()
            prod_lines: List[str] = []
            for p in products:
                pname = (p.get("product_name") or "").strip() or "(unnamed product)"
                pcat = (p.get("product_category") or "").strip()
                psub = (p.get("product_subcategory") or "").strip()
                label = " - ".join([x for x in [pname, pcat, psub] if x])
                if label.lower() in seen_prod:
                    continue
                seen_prod.add(label.lower())
                prod_lines.append(label)

            lines.append(f"- {b.get('bank_name')} ({b.get('bank_code')}): {len(prod_lines)} product(s)")
            for pl in prod_lines[:5]:
                lines.append(f"  • {pl}")

        return "\n".join(lines)
    if asks_unsecured and asks_banks:
        filtered = [d for d in loans_raw if _is_unsecured_loan(d)]
        # The user often mentions salary loans while asking for unsecured; exclude salary loans unless explicitly requested.
        if "salary" in q_low and "unsecured" in q_low and "salary loan" not in q_low:
            filtered = [d for d in filtered if not _is_salary_loan(d)]

        if not filtered:
            scope = "all banks" if not bank_codes else f"banks={bank_codes}"
            return f"No unsecured/no-collateral loan products found in the dataset for {scope}."

        return _group_products_by_bank(filtered, "Banks with unsecured / no-collateral loan products (from MongoDB):")

    # Deterministic: list banks that offer housing/home/mortgage loans.
    if asks_housing and asks_banks:
        filtered = [d for d in loans_raw if _is_housing_loan(d)]
        if not filtered:
            scope = "all banks" if not bank_codes else f"banks={bank_codes}"
            return f"No housing/home/mortgage loan products found in the dataset for {scope}."
        return _group_products_by_bank(filtered, "Banks with housing/home/mortgage loan products (from MongoDB):")

    # Deterministic: when user asks about a specific product type for a specific bank
    # (e.g., "car loan of ACLEDA" / "does ACLEDA have motorbike loan?")
    product_keys = _extract_product_keywords(user_question)
    asks_specific_bank = bool(bank_codes)
    asks_existence = any(k in q_low for k in ["or not", "do they have", "does", "have", "has"]) and ("loan" in q_low or product_keys)
    asks_details = any(k in q_low for k in ["tell me about", "details", "detail", "what is", "explain", "brief", "summarize", "summary", "overview"]) and ("loan" in q_low or product_keys)

    if asks_specific_bank and product_keys and (asks_existence or asks_details):
        filtered = [d for d in loans_raw if _matches_product_keys(d, product_keys)]
        if asks_existence and not filtered:
            return f"No matching loan product found for banks={bank_codes} with keywords={product_keys}."
        if asks_details and not filtered:
            return f"No matching loan product details found for banks={bank_codes} with keywords={product_keys}."
        if asks_existence and filtered:
            # Keep it short: list product names found.
            names: List[str] = []
            for d in filtered[:10]:
                n = _loan_field(d, "product_name", "loan_name")
                if n and n not in names:
                    names.append(n)
            found_bank = _loan_field(filtered[0], "bank_name", "bank") or (bank_codes[0] if bank_codes else "the bank")
            return "Yes. Found these loan product(s) in the dataset for " + found_bank + ": " + "; ".join(names)

        # Details: build a deterministic summary (avoid LLM hallucination for missing fields)
        if filtered:
            d0, _ = _best_match_by_tokens(filtered, product_keys)
            if not d0:
                d0 = filtered[0]
            if asks_interest_rate_only:
                return _deterministic_loan_interest_rate_only(d0)
            return _deterministic_loan_full_details(d0)

    # Generic (no hard-coding): bank-specific product lookup by tokens.
    # Examples:
    # - "brief me the green energy loan of ACLEDA"
    # - "does wing have digital loan?"
    if asks_specific_bank and (asks_existence or asks_details):
        query_tokens = _infer_product_query_tokens(user_question)
        # If user didn't provide any distinguishing tokens, don't guess.
        if query_tokens:
            d0, best_score = _best_match_by_tokens(loans_raw, query_tokens)
            # Require at least some evidence to avoid wrong matches.
            if d0 and best_score >= 3:
                if asks_existence and not asks_details:
                    pname = _loan_field(d0, "product_name", "loan_name") or "(unnamed product)"
                    bank_name = _loan_field(d0, "bank_name", "bank") or (bank_codes[0] if bank_codes else "the bank")
                    return f"Yes. Found a matching loan product in the dataset for {bank_name}: {pname}"
                if asks_interest_rate_only:
                    return _deterministic_loan_interest_rate_only(d0)
                return _deterministic_loan_full_details(d0)
            if asks_existence and not asks_details:
                return f"No matching loan product found for banks={bank_codes} with query tokens={query_tokens}."

    # Deterministic count questions
    if re.search(r"\bhow\s+many\b", user_question.lower()):
        bank_msg = "" if not bank_codes else f" for {', '.join(bank_codes)}"
        return f"Found {len(loans_raw)} loan product(s){bank_msg}."

    def _extract_interest_rate_summary(doc: Dict[str, Any]) -> Dict[str, Any]:
        ir = doc.get("interest_rate")
        out: Dict[str, Any] = {}
        if not isinstance(ir, dict):
            return out

        for ccy in ("USD", "KHR"):
            v = ir.get(ccy.lower()) if ccy.lower() in ir else ir.get(ccy)
            if v is None:
                continue
            # Example USD shape: [{condition: "Up to ...", rate_percent_per_annum: 7}, ...]
            if isinstance(v, list):
                cleaned: List[Dict[str, Any]] = []
                for row in v:
                    if not isinstance(row, dict):
                        continue
                    rate = row.get("rate_percent_per_annum")
                    cond = row.get("condition")
                    if rate is None and cond is None:
                        continue
                    cleaned.append({
                        "condition": cond,
                        "rate_percent_per_annum": rate,
                    })
                if cleaned:
                    out[ccy] = cleaned
            elif isinstance(v, str):
                # Normalize common placeholders into a friendlier, still-grounded message.
                low = v.strip().lower()
                if "as per" in low and "policy" in low:
                    out[ccy] = "Varies by bank policy (rate not specified in this dataset)."
                else:
                    out[ccy] = v
            else:
                out[ccy] = v
        return out

    # Very simple relevance ranking: count keyword hits in name/category/description.
    q = user_question.lower()
    keywords = [k for k in [
        "mortgage",
        "home",
        "housing",
        "car",
        "auto",
        "vehicle",
        "motorbike",
        "moto",
        "motorcycle",
        "digital",
        "app",
        "instant",
        "personal",
        "business",
        "education",
        "unsecured",
        "collateral",
        "salary",
    ] if k in q]

    def score(doc: Dict[str, Any]) -> int:
        hay = _loan_haystack(doc)
        s = 0
        for k in keywords:
            if k in hay:
                s += 1
        return s

    ranked = sorted(loans_raw, key=score, reverse=True)

    # Attach deterministic rate summaries for the top results to reduce LLM mistakes.
    top = ranked[:top_n]
    top_slim: List[Dict[str, Any]] = []
    for d in top:
        top_slim.append(
            {
                "bank_name": d.get("bank") or d.get("bank_name"),
                "bank_code": d.get("bank_code"),
                "loan_name": d.get("loan_name") or d.get("product_name"),
                "loan_category": d.get("loan_category") or d.get("product_category"),
                "product_subcategory": d.get("product_subcategory") or d.get("loan_subcategory"),
                "currency": d.get("currency"),
                "loan_amount": d.get("loan_amount") or (d.get("loan_terms") or {}).get("loan_amount"),
                "loan_term": d.get("loan_term") or (d.get("loan_terms") or {}).get("loan_term"),
                "fees": d.get("fees"),
                "interest_rate": _extract_interest_rate_summary(d) or "Data not available",
                "source_url": (d.get("additional_information") or {}).get("source_url") or d.get("source_url"),
                "updated_date": (d.get("additional_information") or {}).get("updated_date") or d.get("updated_date"),
            }
        )

    prompt = f"""
You are a banking assistant.

Rules:
- Use ONLY the loan data provided below.
- Do NOT use external knowledge.
- If a requested detail is missing, say it's not provided in the dataset.
- If KHR rate says it varies by policy, explain politely that the KHR rate isn't specified here and needs confirmation with the bank.
- If USD interest_rate tiers are present, you MUST list the tiers (condition + rate_percent_per_annum).

User question:
{user_question}

Top matched loan products (from MongoDB):
{json.dumps(serialize_for_json(top_slim), indent=2)}

Answer:
1) Provide a direct answer.
2) Compare the top options briefly (currency, amount range, max years, fees if present).
3) Mention key requirements (documents, eligibility, collateral) only if present.
Keep it concise.
"""

    try:
        return ollama_chat(
            [
                {"role": "system", "content": "Be concise, factual, and strictly grounded in provided data."},
                {"role": "user", "content": prompt},
            ]
        )
    except Exception as e:
        summary = _format_plain_list(
            serialize_for_json(top_slim),
            title=f"Loan products (top {len(top_slim)} of {len(loans_raw)} matches)",
            keys=["bank_name", "loan_name", "loan_category", "interest_rate", "source_url"],
            limit=min(len(top_slim), top_n),
        )
        return summary + "\n\nNote: Could not generate LLM explanation: " + str(e)


def answer_credit_card_question(user_question: str) -> str:
    bank_codes = extract_bank_codes_from_text(user_question)
    cards_raw = get_credit_cards(bank_codes=bank_codes or None, limit=200)
    if not cards_raw:
        scope = "all banks" if not bank_codes else f"banks={bank_codes}"
        return f"No credit card data found for {scope}."

    if re.search(r"\bhow\s+many\b", user_question.lower()):
        bank_msg = "" if not bank_codes else f" for {', '.join(bank_codes)}"
        return f"Found {len(cards_raw)} credit card product(s){bank_msg}."

    q = user_question.lower()
    keywords = [k for k in ["cashback", "cash back", "reward", "points", "miles", "fee", "annual", "interest", "installment"] if k in q]

    def score(doc: Dict[str, Any]) -> int:
        hay = " ".join([json.dumps(doc, default=str)]).lower()
        s = 0
        for k in keywords:
            if k in hay:
                s += 1
        return s

    ranked = sorted(cards_raw, key=score, reverse=True)

    prompt = f"""
You are a banking assistant.

Rules:
- Use ONLY the credit card data provided below.
- Do NOT use external knowledge.
- Do NOT guess fees or eligibility.

User question:
{user_question}

Top matched credit cards (from MongoDB):
{json.dumps(serialize_for_json(ranked[:5]), indent=2)}

Answer:
1) Provide a direct answer.
2) Compare key differences (benefits, fees, eligibility) only when present.
3) If something isn't present, say 'Data not available'.
Keep it concise.
"""

    try:
        return ollama_chat(
            [
                {"role": "system", "content": "Be concise, factual, and strictly grounded in provided data."},
                {"role": "user", "content": prompt},
            ]
        )
    except Exception as e:
        summary = _format_plain_list(
            serialize_for_json(ranked),
            title="Credit cards (top matches)",
            keys=["bank_name", "card_name", "product_name", "source_url"],
            limit=5,
        )
        return summary + "\n\nNote: Could not generate LLM explanation: " + str(e)


def answer_bank_question(user_question: str) -> str:
    banks = _get_bank_master()
    if not banks:
        return "No bank master data found in MongoDB."
    # Simple list; for small projects this is usually enough.
    title = "Banks (from bank_code master)"
    keys = ["bank_code", "bank_name", "bank_type", "country"]
    return _format_plain_list(serialize_for_json(banks), title=title, keys=keys, limit=25)


# ===============================
# Deterministic Calculations
# ===============================
def compute_fx(amount: float, from_ccy: str, to_ccy: str, fx_rates: List[Dict]) -> List[Dict]:
    """
    Conventional bank FX interpretation for a quote like USD/KHR:
    - buy_rate: bank buys USD from customer (customer sells USD) => USD -> KHR uses buy_rate
    - sell_rate: bank sells USD to customer (customer buys USD)  => KHR -> USD uses sell_rate
    """
    results = []
    direction = f"{from_ccy}-{to_ccy}"

    for fx in fx_rates:
        buy_rate = float(fx.get("buy_rate", 0) or 0)
        sell_rate = float(fx.get("sell_rate", 0) or 0)

        if from_ccy == "USD" and to_ccy == "KHR":
            # Customer sells USD, bank buys USD => use buy_rate (KHR per USD).
            if buy_rate <= 0:
                continue
            out_amt = amount * buy_rate
            used_rate = buy_rate
            used_field = "buy_rate"
        elif from_ccy == "KHR" and to_ccy == "USD":
            # Customer buys USD, bank sells USD => use sell_rate (KHR per USD).
            if sell_rate <= 0:
                continue
            out_amt = amount / sell_rate
            used_rate = sell_rate
            used_field = "sell_rate"
        else:
            continue

        results.append({
            "bank_code": fx.get("bank_code"),
            "bank_name": fx.get("bank_name"),
            "direction": direction,
            "amount_in": amount,
            "ccy_in": from_ccy,
            "amount_out": round(out_amt, 6 if to_ccy == "USD" else 0),
            "ccy_out": to_ccy,
            "rate_used": used_rate,
            "rate_field": used_field,
            "scraped_at": serialize_for_json(fx.get("scraped_at"))
        })

    # Best for customer:
    # - USD->KHR: max amount_out
    # - KHR->USD: max amount_out
    results.sort(key=lambda x: x["amount_out"], reverse=True)
    return results


def compute_fx_two_step_best(amount_usd: float, fx_rates: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Best USD -> KHR -> USD round trip across two banks.

    Step 1 (USD->KHR): maximize KHR received => maximize buy_rate.
    Step 2 (KHR->USD): maximize USD received => minimize sell_rate.
    """
    step1 = compute_fx(amount_usd, "USD", "KHR", fx_rates)
    if not step1:
        return {"error": "No valid USD->KHR rates available."}

    # Precompute KHR->USD conversions for each bank's sell_rate.
    # We won't know the KHR amount until we pick step1 bank, so compute functionally.
    valid_banks: List[Dict[str, Any]] = []
    for r in fx_rates:
        try:
            sell_rate = float(r.get("sell_rate", 0) or 0)
        except Exception:
            sell_rate = 0.0
        if sell_rate > 0:
            valid_banks.append({
                "bank_code": r.get("bank_code"),
                "bank_name": r.get("bank_name"),
                "sell_rate": sell_rate,
            })
    if not valid_banks:
        return {"error": "No valid KHR->USD sell_rate available."}

    # Best pair by brute force; n is small.
    best: Optional[Dict[str, Any]] = None
    for s1 in step1[:50]:
        khr_amt = float(s1.get("amount_out") or 0)
        if khr_amt <= 0:
            continue
        for b2 in valid_banks:
            usd_back = khr_amt / float(b2["sell_rate"])
            candidate = {
                "start_usd": amount_usd,
                "step1": {
                    "bank_code": s1.get("bank_code"),
                    "bank_name": s1.get("bank_name"),
                    "rate_field": "buy_rate",
                    "rate_used": s1.get("rate_used"),
                    "usd_in": amount_usd,
                    "khr_out": khr_amt,
                },
                "step2": {
                    "bank_code": b2.get("bank_code"),
                    "bank_name": b2.get("bank_name"),
                    "rate_field": "sell_rate",
                    "rate_used": b2.get("sell_rate"),
                    "khr_in": khr_amt,
                    "usd_out": round(usd_back, 6),
                },
            }
            if (best is None) or (candidate["step2"]["usd_out"] > best["step2"]["usd_out"]):
                best = candidate

    if not best:
        return {"error": "Could not compute a valid two-step conversion."}

    best["net_usd_gain"] = round(best["step2"]["usd_out"] - amount_usd, 6)
    best["net_usd_gain_pct"] = round((best["net_usd_gain"] / amount_usd) * 100.0, 6) if amount_usd else None
    return best


def compute_interest_simple(principal: float, annual_rate_pct: float, months: int) -> Dict:
    """
    Simple interest:
      interest = principal * rate% * (months/12)
    """
    interest = principal * (annual_rate_pct / 100.0) * (months / 12.0)
    total = principal + interest
    return {"interest": round(interest, 2), "total": round(total, 2)}


# ===============================
# Answer Builders (LLM explains)
# ===============================
def answer_fx_question(user_question: str) -> str:
    amount, ccy = parse_amount_currency(user_question)

    t = user_question.lower()
    # Decide direction with simple rules
    if "to usd" in t or "in usd" in t:
        from_ccy, to_ccy = "KHR", "USD"
    elif "to khr" in t or "in khr" in t:
        from_ccy, to_ccy = "USD", "KHR"
    else:
        # fall back based on parsed currency
        from_ccy, to_ccy = ("USD", "KHR") if ccy == "USD" else ("KHR", "USD")
    pair = "USD-KHR"

    fx_raw = get_latest_rates(currency_pair=pair, rate_type="counter")
    fx_rates = serialize_for_json(fx_raw)

    if not fx_rates:
        return "No exchange rate data found in MongoDB."

    # Two-step / round-trip intent: user wants USD->KHR then KHR->USD (or vice versa).
    wants_roundtrip = _is_fx_roundtrip_text(user_question)
    if wants_roundtrip:
        # Deterministic answer to avoid LLM arithmetic mistakes.
        best = compute_fx_two_step_best(float(amount), fx_raw)
        if best.get("error"):
            return f"Two-step FX requested, but {best['error']}"

        s1 = best["step1"]
        s2 = best["step2"]
        lines: List[str] = []
        lines.append("Two-step FX plan (computed from MongoDB rates):")
        lines.append(f"1) USD→KHR: use {s1['bank_name']} ({s1['bank_code']}) at buy_rate={s1['rate_used']} => {int(s1['khr_out'])} KHR")
        lines.append(f"2) KHR→USD: use {s2['bank_name']} ({s2['bank_code']}) at sell_rate={s2['rate_used']} => {s2['usd_out']} USD")
        lines.append(f"Net: {best['net_usd_gain']} USD ({best['net_usd_gain_pct']}%)")
        lines.append("")
        lines.append("Why you usually lose money on USD→KHR→USD:")
        lines.append("- Banks quote a spread: buy_rate < sell_rate.")
        lines.append("- Round-trip factor ≈ buy_rate / sell_rate (typically < 1), so USD_back < USD_start.")
        lines.append("")
        lines.append("Recommended process:")
        lines.append("- Do step 1 at the bank with the highest buy_rate (more KHR per USD).")
        lines.append("- Do step 2 at the bank with the lowest sell_rate (cheapest USD).")
        lines.append("- Check fees/limits; spreads + fees mean round-trips are not a profit strategy.")
        return "\n".join(lines)

    results = serialize_for_json(compute_fx(amount, from_ccy, to_ccy, fx_raw))

    if not results:
        return "Exchange rate data exists but could not compute conversion (missing buy/sell rates)."

    best = results[0]

    # Deterministic single-step answer by default (prevents LLM math errors).
    lines: List[str] = []
    lines.append("FX best option (computed from MongoDB rates):")
    lines.append(f"- Best bank: {best.get('bank_name')} ({best.get('bank_code')})")
    lines.append(f"- Direction: {best.get('direction')} using {best.get('rate_field')}={best.get('rate_used')}")
    lines.append(f"- Result: {best.get('amount_in')} {best.get('ccy_in')} -> {best.get('amount_out')} {best.get('ccy_out')}")
    lines.append("Formula used:")
    if from_ccy == "USD" and to_ccy == "KHR":
        lines.append("- KHR_out = USD_in × buy_rate")
    elif from_ccy == "KHR" and to_ccy == "USD":
        lines.append("- USD_out = KHR_in ÷ sell_rate")

    # Add brief explanation when the user asks why it's losing / not reasonable.
    t_low = (user_question or "").lower()
    if any(k in t_low for k in ["why", "lose", "loss", "not reasonable", "reasonable", "benefit", "profit"]):
        lines.append("")
        lines.append("Why this can look like a loss:")
        lines.append("- Banks quote two prices (spread): buy_rate vs sell_rate.")
        lines.append("- If you convert USD→KHR and then convert back KHR→USD, you usually get less USD because buy_rate < sell_rate.")

    # Show alternatives (top 3) for transparency.
    lines.append("")
    lines.append("Top alternatives (same direction):")
    for i, r in enumerate(results[:3], start=1):
        lines.append(
            f"{i}) {r.get('bank_name')} ({r.get('bank_code')}): {r.get('amount_out')} {r.get('ccy_out')} using {r.get('rate_field')}={r.get('rate_used')}"
        )

    # If user asked for process, include simple steps.
    if any(k in t_low for k in ["process", "step", "how to", "can you help", "help me"]):
        lines.append("")
        lines.append("Process:")
        if from_ccy == "USD" and to_ccy == "KHR":
            lines.append("- Sell USD to the bank (you receive KHR) using the bank's buy_rate.")
        elif from_ccy == "KHR" and to_ccy == "USD":
            lines.append("- Buy USD from the bank (you pay KHR) using the bank's sell_rate.")
        lines.append("- Confirm fees/limits and the timestamp (as_of) before transacting.")

    return "\n".join(lines)


def get_fixed_deposits(
    bank_codes: Optional[List[str]] = None,
    bank_keyword: Optional[str] = None,
    currency: str = "USD",
    limit: int = 50
) -> List[Dict]:

    ccy = currency.upper()

    # Fixed-deposit data can arrive in multiple schemas:
    # - product_type="Fixed Deposit" with currencies_supported + rates
    # - product_type="Deposit" with product_category="Term Deposit" and interest_rates.growth_modes
    # Filter broadly but still keep it in the term-deposit domain.
    query: Dict[str, Any] = {
        "$and": [
            {
                "$or": [
                    {"product_type": "Fixed Deposit"},
                    {"product_type": "Deposit"},
                    {"product_category": {"$regex": r"\bterm\s+deposit\b", "$options": "i"}},
                    {"product_type": {"$regex": r"\bterm\s+deposit\b", "$options": "i"}},
                ]
            },
            {
                "$or": [
                    {"currencies_supported": {"$in": [ccy]}},
                    {"currencies": {"$in": [ccy]}},
                    {"deposit_features.currencies": {"$in": [ccy]}},
                    # ABA-style nested rates
                    {f"rates.over_the_counter.{ccy}": {"$exists": True}},
                    {f"rates.aba_mobile.{ccy}": {"$exists": True}},
                    # KB PRASAC-style growth modes
                    {f"interest_rates.growth_modes.maturity_growth.{ccy}": {"$exists": True}},
                    {f"interest_rates.growth_modes.monthly_growth.{ccy}": {"$exists": True}},
                    # Wing-style interest_rates array
                    {"interest_rates": {"$elemMatch": {f"at_maturity.{ccy}": {"$exists": True}}}},
                    {"interest_rates": {"$elemMatch": {f"monthly.{ccy}": {"$exists": True}}}},
                ]
            },
        ]
    }

    if bank_codes:
        query["bank_code"] = {"$in": [c.strip().upper() for c in bank_codes if c and str(c).strip()]}

    if bank_keyword:
        query["bank_name"] = {"$regex": f"^{bank_keyword}", "$options": "i"}

    cursor = db[COLL_FIXED].find(
        query,
        {
            "_id": 0,
            "bank_code": 1,
            "bank_name": 1,
            "product_name": 1,
            "product_category": 1,
            "product_slug": 1,
            "product_type": 1,
            "segment": 1,
            "currencies_supported": 1,
            "currencies": 1,
            "channels": 1,
            "interest_payment_options": 1,
            "interest_payment_modes": 1,
            "rates": 1,
            "interest_rates": 1,
            "conditions": 1,
            "early_closure_rules": 1,
            "benefits": 1,
            "requirements": 1,
            "source_url": 1,
            "inserted_at": 1,
            "created_at": 1,
            "updated_at": 1,
        },
    ).limit(limit)
    return list(cursor)


def extract_fd_rate_from_schema(
    product: Dict[str, Any],
    currency: str,
    term_months: int,
    payment: str,
    channel_keys: Optional[List[str]] = None,
) -> Optional[Dict[str, Any]]:
    """Extract a rate record from the MongoDB schema shown in the sample.

    Expected shape:
      rates[channel_key][currency] -> list of {term_months, at_maturity, monthly}
    Returns a dict with: channel_key, currency, term_months, rate_pct
    """
    ccy = currency.upper()
    best: Optional[Dict[str, Any]] = None

    rates = product.get("rates")

    def _pick_better(candidate: Dict[str, Any]) -> None:
        nonlocal best
        if best is None or float(candidate.get("rate_pct") or 0) > float(best.get("rate_pct") or 0):
            best = candidate

    if isinstance(rates, dict) and rates:
        # Schema A (ABA-style): rates[channel_key][CCY] -> list of {term_months, at_maturity, monthly, ...}
        wanted_channels = channel_keys or list(rates.keys())
        for ch in wanted_channels:
            ch_rates = rates.get(ch)
            if not isinstance(ch_rates, dict):
                continue
            term_list = ch_rates.get(ccy)
            if not isinstance(term_list, list):
                continue

            for row in term_list:
                if not isinstance(row, dict):
                    continue
                if int(row.get("term_months") or 0) != int(term_months):
                    continue
                rate_val = row.get(payment)
                if rate_val is None:
                    continue
                try:
                    rate_pct = float(rate_val)
                except Exception:
                    continue

                _pick_better(
                    {
                        "schema": "channel",
                        "channel_key": ch,
                        "currency": ccy,
                        "term_months": int(term_months),
                        "payment": payment,
                        "rate_pct": rate_pct,
                    }
                )

        if best is not None:
            return best

        # Schema B (Canadia-style): rates[group_key] -> list of rows.
        # Each row has term_months and per-currency dicts like USD:{at_maturity, quarterly, ...}
        payment_key_candidates: List[str] = [payment]
        if payment == "at_maturity":
            payment_key_candidates.extend(["maturity", "atMaturity", "at-maturity"])
        if payment == "monthly":
            payment_key_candidates.extend(["per_month", "per-month"])

        for group_key, group_list in rates.items():
            if not isinstance(group_list, list):
                continue
            for row in group_list:
                if not isinstance(row, dict):
                    continue
                tm = row.get("term_months")
                if tm is None:
                    continue
                try:
                    if int(tm) != int(term_months):
                        continue
                except Exception:
                    continue

                cur_obj = row.get(ccy) if ccy in row else row.get(ccy.lower())
                if cur_obj is None:
                    continue

                rate_val = None
                if isinstance(cur_obj, dict):
                    for pk in payment_key_candidates:
                        if pk in cur_obj and cur_obj.get(pk) is not None:
                            rate_val = cur_obj.get(pk)
                            break
                else:
                    rate_val = cur_obj

                if rate_val is None:
                    continue
                try:
                    rate_pct = float(rate_val)
                except Exception:
                    continue

                _pick_better(
                    {
                        "schema": "rate_group",
                        "rate_group": group_key,
                        "currency": ccy,
                        "term_months": int(term_months),
                        "payment": payment,
                        "rate_pct": rate_pct,
                    }
                )

        if best is not None:
            return best

    # Schema C (KB PRASAC-style): interest_rates.growth_modes.{maturity_growth,monthly_growth}
    ir = product.get("interest_rates")
    if isinstance(ir, dict):
        gm = ir.get("growth_modes")
        if isinstance(gm, dict):
            # Map our payment to growth mode.
            mode_key = "maturity_growth" if payment == "at_maturity" else "monthly_growth"
            rows = gm.get(mode_key)
            if isinstance(rows, list):
                for row in rows:
                    if not isinstance(row, dict):
                        continue

                    # Rows can have months as a list or single value.
                    months_val = row.get("months")
                    months_list: List[int] = []
                    if isinstance(months_val, list):
                        for m in months_val:
                            try:
                                months_list.append(int(m))
                            except Exception:
                                pass
                    elif months_val is not None:
                        try:
                            months_list = [int(months_val)]
                        except Exception:
                            months_list = []

                    # Some sources might use term_months instead.
                    if not months_list and row.get("term_months") is not None:
                        try:
                            months_list = [int(row.get("term_months"))]
                        except Exception:
                            months_list = []

                    if int(term_months) not in months_list:
                        continue

                    rate_val = row.get(ccy) if ccy in row else row.get(ccy.lower())
                    if rate_val is None:
                        continue
                    try:
                        rate_pct = float(rate_val)
                    except Exception:
                        continue

                    return {
                        "schema": "growth_modes",
                        "growth_mode": mode_key,
                        "currency": ccy,
                        "term_months": int(term_months),
                        "payment": payment,
                        "rate_pct": rate_pct,
                    }

    # Schema D (Wing-style): interest_rates is a list of rows with term_months and
    # payment blocks like at_maturity: {USD: 5.5, KHR: 5.5}, monthly: {USD: 4, KHR: 4}
    if isinstance(ir, list):
        for row in ir:
            if not isinstance(row, dict):
                continue
            tm = row.get("term_months")
            if tm is None:
                continue
            try:
                if int(tm) != int(term_months):
                    continue
            except Exception:
                continue

            pay_obj = row.get(payment)
            # Some sources may use 'maturity' instead of 'at_maturity'
            if pay_obj is None and payment == "at_maturity":
                pay_obj = row.get("maturity")
            if pay_obj is None:
                continue

            rate_val = None
            if isinstance(pay_obj, dict):
                rate_val = pay_obj.get(ccy) if ccy in pay_obj else pay_obj.get(ccy.lower())
            else:
                rate_val = pay_obj

            if rate_val is None:
                continue
            try:
                rate_pct = float(rate_val)
            except Exception:
                continue

            return {
                "schema": "interest_rates_list",
                "currency": ccy,
                "term_months": int(term_months),
                "payment": payment,
                "rate_pct": rate_pct,
            }

    return None



def answer_fixed_deposit_question(user_question: str) -> str:
    # If the user is asking a definition/explanation (not a quote/comparison),
    # respond with a deterministic, general explanation instead of picking a bank.
    q_low = (user_question or "").lower()
    asks_definition = (
        any(k in q_low for k in ["what is", "what's", "define", "meaning", "explain"])
        and any(k in q_low for k in ["fixed deposit", "term deposit", "fd"])
        and not re.search(r"\b\d+\b", q_low)
        and not any(k in q_low for k in ["best", "compare", "rate", "interest", "how much", "profit", "return"])
    )
    if asks_definition:
        return "\n".join(
            [
                "A fixed deposit (term deposit) is a deposit where you place money with a bank for a fixed term (e.g., 3/6/12 months) at an agreed interest rate.",
                "Key points:",
                "- Term: your money is typically locked until maturity (early withdrawal may reduce interest or incur rules).",
                "- Interest: paid monthly or at maturity depending on the product.",
                "- Currency & channel: rates often differ by USD/KHR and by channel (branch vs mobile/app).",
                "To compare fixed deposits in this app, include: amount, currency (USD/KHR), term, and interest payment (monthly/at maturity), plus bank name if you want a specific bank.",
            ]
        )

    principal, ccy = parse_amount_currency(user_question)
    months = parse_term_months(user_question, default_months=12)
    payment = parse_interest_payment(user_question)
    channel_keys = parse_fd_channel_keys(user_question)

    bank_codes = extract_bank_codes_from_text(user_question) or None
    products_raw = get_fixed_deposits(bank_codes=bank_codes, currency=ccy, limit=500)

    if not products_raw:
        scope = "banks" if not bank_codes else f"banks={bank_codes}"
        return f"No fixed deposit data found for {scope} in {ccy}."

    # Deterministic: user explicitly asks to list all banks' rates for a given term.
    q_low = (user_question or "").lower()
    wants_all_banks = "all banks" in q_low or ("all bank" in q_low) or ("every bank" in q_low)
    wants_list = any(k in q_low for k in ["list", "show", "give me", "display"])
    if wants_all_banks and wants_list:
        rows: List[Dict[str, Any]] = []
        missing: List[str] = []
        missing_with_tenor: List[str] = []

        for p in products_raw:
            bank = p.get("bank_name", "UNKNOWN")
            bcode = p.get("bank_code") or ""
            rate_info = extract_fd_rate_from_schema(
                p,
                currency=ccy,
                term_months=months,
                payment=payment,
                channel_keys=channel_keys,
            )
            if not rate_info:
                label = f"{bank}" + (f" ({bcode})" if bcode else "")
                # If the product explicitly supports the tenor but the numeric table isn't present,
                # call it out separately (common when scraped content didn't include a rate table).
                tenor_list = None
                if isinstance((p.get("deposit_features") or {}).get("tenors_months"), list):
                    tenor_list = (p.get("deposit_features") or {}).get("tenors_months")
                elif isinstance(p.get("tenors_months"), list):
                    tenor_list = p.get("tenors_months")

                if tenor_list and any(str(x) == str(months) for x in tenor_list):
                    missing_with_tenor.append(label)
                else:
                    missing.append(label)
                continue

            src = ""
            if rate_info.get("schema") == "channel" and rate_info.get("channel_key"):
                src = f"channel={rate_info.get('channel_key')}"
            elif rate_info.get("schema") == "rate_group" and rate_info.get("rate_group"):
                src = f"group={rate_info.get('rate_group')}"

            rows.append(
                {
                    "bank": bank,
                    "bank_code": bcode,
                    "rate_pct": float(rate_info.get("rate_pct") or 0),
                    "source": src,
                }
            )

        if not rows:
            return (
                f"Fixed deposit products found, but no matching rate for term={months} months, "
                f"currency={ccy.upper()}, payment={payment} (and channel filter={channel_keys or 'any'})."
            )

        # One row per bank: keep the best rate per bank (if multiple docs exist).
        best_by_bank: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            key = (r.get("bank_code") or r.get("bank") or "UNKNOWN").upper()
            if key not in best_by_bank or float(r.get("rate_pct") or 0) > float(best_by_bank[key].get("rate_pct") or 0):
                best_by_bank[key] = r

        final_rows = sorted(best_by_bank.values(), key=lambda x: (-float(x.get("rate_pct") or 0), str(x.get("bank") or "").lower()))

        lines: List[str] = []
        lines.append("Fixed deposit rates (from MongoDB):")
        lines.append(f"- Term: {months} months")
        lines.append(f"- Currency: {ccy.upper()}")
        lines.append(f"- Interest payment: {payment}")
        if channel_keys:
            lines.append(f"- Channel filter: {', '.join(channel_keys)}")
        lines.append("")
        for r in final_rows:
            label = f"{r.get('bank')}" + (f" ({r.get('bank_code')})" if r.get("bank_code") else "")
            extra = f" ({r.get('source')})" if r.get("source") else ""
            lines.append(f"- {label}: {r.get('rate_pct')}%{extra}")

        if missing:
            lines.append("")
            lines.append("No matching 36-month rate found for:")
            for m in sorted(set(missing), key=lambda x: x.lower()):
                lines.append(f"- {m}")

        if missing_with_tenor:
            lines.append("")
            lines.append("Tenor exists, but rate table missing in dataset (needs re-scrape / schema update):")
            for m in sorted(set(missing_with_tenor), key=lambda x: x.lower()):
                lines.append(f"- {m}")

        return "\n".join(lines)

    ranked: List[Dict[str, Any]] = []
    for p in products_raw:
        bank = p.get("bank_name", "UNKNOWN")
        rate_info = extract_fd_rate_from_schema(
            p,
            currency=ccy,
            term_months=months,
            payment=payment,
            channel_keys=channel_keys,
        )
        if not rate_info:
            continue

        rate = float(rate_info["rate_pct"])
        calc = compute_interest_simple(principal, rate, months)

        min_deposit = ((p.get("conditions") or {}).get("minimum_initial_deposit") or {}).get(ccy.upper())

        ranked.append(
            {
                "bank": bank,
                "product": p.get("product_name"),
                "segment": p.get("segment"),
                "currency": ccy.upper(),
                "term_months": months,
                "interest_payment": payment,
                "channel_key": rate_info.get("channel_key"),
                "annual_rate_pct": rate,
                "principal": principal,
                "interest": calc["interest"],
                "total": calc["total"],
                "minimum_initial_deposit": min_deposit,
                "partial_withdrawal": (p.get("conditions") or {}).get("partial_withdrawal"),
                "early_closure_rules": p.get("early_closure_rules"),
                "benefits": (p.get("benefits") or [])[:5] if isinstance(p.get("benefits"), list) else p.get("benefits"),
                "requirements": (p.get("requirements") or [])[:5] if isinstance(p.get("requirements"), list) else p.get("requirements"),
                "source_url": p.get("source_url"),
            }
        )

    if not ranked:
        return (
            f"Fixed deposit data found, but no matching rate for term={months} months, "
            f"currency={ccy.upper()}, payment={payment} "
            f"(and channel filter={channel_keys or 'any'})."
        )

    ranked.sort(key=lambda x: x["total"], reverse=True)

    prompt = f"""
You are a financial assistant.

Rules:
- Use ONLY the data provided below.
- Do NOT use external knowledge.
- Do NOT guess missing values.
- If information is missing, respond with "Data not available".
- Do NOT recalculate numbers (they are precomputed).
- Do NOT claim "no fees" or "no penalties" unless the dataset explicitly states it.
- Only mention early withdrawal/closure behaviour if early_closure_rules is present; otherwise say "Data not available".

User question:
{user_question}

Matched fixed deposit options (from MongoDB; interest/total computed in Python using simple interest):
Best option:
{json.dumps(serialize_for_json(ranked[0]), indent=2)}

Top options:
{json.dumps(serialize_for_json(ranked[:5]), indent=2)}

Explain:
1) Answer the user's question directly.
2) State which option is best and why (based on computed total).
3) Show the simple interest formula used: interest = principal * (rate/100) * (months/12).
4) If minimum deposit exists, mention whether the principal meets it.
5) Keep it concise.
"""

    try:
        return ollama_chat(
            [
                {"role": "system", "content": "Be concise, factual, and strictly grounded in provided data."},
                {"role": "user", "content": prompt},
            ]
        )
    except Exception as e:
        # If the LLM endpoint is down/misconfigured, still provide a useful answer.
        summary = _format_plain_ranked_table(
            ranked,
            title="Fixed deposit options (computed from MongoDB)",
            limit=5,
        )
        return summary + "\n\nNote: Could not generate LLM explanation: " + str(e)



def answer_interest_question(user_question: str) -> str:
    # Basic parse: amount & currency; assume 12 months if "1 year"
    principal, ccy = parse_amount_currency(user_question)
    months = 12 if "year" in user_question.lower() else 12  # keep simple

    products_raw = get_savings_accounts(currency=ccy, limit=200)
    products = serialize_for_json(products_raw)

    if not products:
        return f"No savings account data found for currency {ccy}."

    def _extract_savings_rate_pct(doc: Dict[str, Any], ccy: str) -> float:
        """Extract a single comparable annual rate for ranking.

        Supports:
        - interest_rate: number
        - interest_rate: {USD: x, KHR: y}
        - interest_rates: {USD: [{max_balance, rate}, ...], KHR: [...]}
        """
        c = (ccy or "").strip().upper()

        v = doc.get("interest_rate")
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            try:
                return float(v)
            except Exception:
                return 0.0
        if isinstance(v, dict):
            vv = v.get(c) if c in v else v.get(c.lower())
            if isinstance(vv, (int, float)):
                return float(vv)
            if isinstance(vv, str):
                try:
                    return float(vv)
                except Exception:
                    return 0.0

        tiers = doc.get("interest_rates")
        if isinstance(tiers, dict):
            tv = tiers.get(c) if c in tiers else tiers.get(c.lower())
            if isinstance(tv, list):
                rates: List[float] = []
                for row in tv:
                    if not isinstance(row, dict):
                        continue
                    r = row.get("rate")
                    if r is None:
                        r = row.get("rate_percent")
                    if r is None:
                        r = row.get("rate_percent_per_annum")
                    try:
                        if r is not None:
                            rates.append(float(r))
                    except Exception:
                        pass
                return max(rates) if rates else 0.0

        return 0.0

    # Build a normalized ranking in Python
    ranked = []
    for p in products_raw:
        bank = p.get("bank_name") or p.get("bank") or p.get("bank_code") or "UNKNOWN"
        rate = _extract_savings_rate_pct(p, ccy)

        calc = compute_interest_simple(principal, rate, months)
        ranked.append({
            "bank": bank,
            "annual_rate_pct": rate,
            "months": months,
            "principal": principal,
            "interest": calc["interest"],
            "total": calc["total"],
            "currency": ccy
        })

    ranked.sort(key=lambda x: x["total"], reverse=True)
    best = ranked[0]

    prompt = f"""
You are a banking advisor.

User question:
{user_question}

Computed interest ranking (Python-calculated, from MongoDB data):
Best:
{json.dumps(best, indent=2)}

Top 5:
{json.dumps(ranked[:5], indent=2)}

Explain:
1) Which bank gives the highest total.
2) Show the simple interest formula.
3) Clearly state assumptions (simple interest, months/12, no fees).
4) Provide a recommendation.
"""

    return ollama_chat([
        {"role": "system", "content": "Do NOT recalculate numbers; explain the computed results."},
        {"role": "user", "content": prompt}
    ])


def answer_savings_question(user_question: str, bank_codes: Optional[List[str]] = None, currency: Optional[str] = None) -> str:
    bank_codes = bank_codes or extract_bank_codes_from_text(user_question)
    currency = currency or parse_currency_hint(user_question)

    products_raw = get_savings_accounts(bank_codes=bank_codes or None, currency=currency, limit=200)
    if not products_raw and bank_codes and not currency:
        # If user didn't mention currency, don't filter and retry.
        products_raw = get_savings_accounts(bank_codes=bank_codes, currency=None, limit=200)

    if not products_raw:
        scope = "all banks" if not bank_codes else f"banks={bank_codes}"
        if currency:
            return f"No savings account data found for {scope} ({currency.upper()})."
        return f"No savings account data found for {scope}."

    def _norm(v: Any) -> str:
        if v is None:
            return ""
        if isinstance(v, (dict, list)):
            try:
                return json.dumps(v, ensure_ascii=False, default=str)
            except Exception:
                return str(v)
        return str(v)

    def _render_value(v: Any, indent: str = "") -> List[str]:
        lines: List[str] = []
        if v is None:
            lines.append(f"{indent}Data not available")
            return lines
        if isinstance(v, dict):
            if not v:
                lines.append(f"{indent}(empty)")
                return lines
            for k in sorted(v.keys(), key=lambda x: str(x)):
                vv = v.get(k)
                if isinstance(vv, (dict, list)):
                    lines.append(f"{indent}{k}:")
                    lines.extend(_render_value(vv, indent + "  "))
                else:
                    s = str(vv).strip() if vv is not None else ""
                    lines.append(f"{indent}{k}: {s if s else 'Data not available'}")
            return lines
        if isinstance(v, list):
            if not v:
                lines.append(f"{indent}(empty)")
                return lines
            for item in v[:50]:
                if isinstance(item, (dict, list)):
                    lines.append(f"{indent}-")
                    lines.extend(_render_value(item, indent + "  "))
                else:
                    s = str(item).strip() if item is not None else ""
                    lines.append(f"{indent}- {s if s else 'Data not available'}")
            if len(v) > 50:
                lines.append(f"{indent}- … ({len(v) - 50} more item(s))")
            return lines

        s = str(v).strip()
        lines.append(f"{indent}{s if s else 'Data not available'}")
        return lines

    def _format_interest_rates(doc: Dict[str, Any]) -> List[str]:
        lines: List[str] = []
        rates = doc.get("interest_rates")
        if isinstance(rates, dict) and rates:
            lines.append("Interest rates:")
            for ccy in sorted(rates.keys(), key=lambda x: str(x)):
                v = rates.get(ccy)
                lines.append(f"- {ccy}:")
                if isinstance(v, list):
                    for row in v:
                        if not isinstance(row, dict):
                            lines.append(f"  - {_norm(row)}")
                            continue
                        max_bal = row.get("max_balance")
                        rate = row.get("rate")
                        label = "Up to" if max_bal is not None else "Above"
                        if max_bal is None:
                            lines.append(f"  - {label}: {rate}%")
                        else:
                            lines.append(f"  - {label} {max_bal}: {rate}%")
                else:
                    lines.extend(["  " + x for x in _render_value(v, indent="").copy()])
            return lines

        legacy = doc.get("interest_rate")
        if isinstance(legacy, dict) and legacy:
            lines.append("Interest rate:")
            for ccy in sorted(legacy.keys(), key=lambda x: str(x)):
                vv = legacy.get(ccy)
                lines.append(f"- {ccy}: {_norm(vv)}")
            return lines
        if isinstance(legacy, (int, float, str)) and str(legacy).strip():
            lines.append(f"Interest rate: {legacy}")
        return lines

    # If the user asked for a specific bank, show those products. Otherwise, show top matches.
    # Keep it deterministic and friendly.
    lines: List[str] = []
    title = "Savings account product(s) (from MongoDB):" if len(products_raw) > 1 else "Savings account product (from MongoDB):"
    lines.append(title)

    for idx, doc in enumerate(products_raw[:10]):
        if idx > 0:
            lines.append("\n---")
        bank_name = doc.get("bank_name") or doc.get("bank") or "UNKNOWN"
        bank_code = doc.get("bank_code") or ""
        pname = doc.get("product_name") or doc.get("account_name") or "(unnamed product)"
        ptype = doc.get("product_type") or ""
        source = doc.get("source_url") or (doc.get("additional_information") or {}).get("source_url")
        updated = doc.get("updated_at") or doc.get("updated_date") or doc.get("scraped_at") or doc.get("inserted_at")

        lines.append(f"- Bank: {bank_name}" + (f" ({bank_code})" if bank_code else ""))
        lines.append(f"- Product: {pname}")
        if ptype:
            lines.append(f"- Type: {ptype}")
        if updated:
            lines.append(f"- Updated: {updated}")
        if source:
            lines.append(f"- Source: {source}")

        ir_lines = _format_interest_rates(doc)
        if ir_lines:
            lines.append("")
            lines.extend(ir_lines)

        for key, label in [
            ("conditions", "Conditions"),
            ("benefits", "Benefits"),
            ("requirements", "Requirements"),
            ("faqs", "FAQs"),
            ("abandonment_fee", "Abandonment fee"),
            ("description", "Description"),
        ]:
            if key in doc and doc.get(key) not in (None, "", [], {}):
                lines.append(f"\n{label}:")
                lines.extend(_render_value(doc.get(key), indent=""))

    if len(products_raw) > 10:
        lines.append(f"\nNote: showing 10 of {len(products_raw)} matched savings products.")

    return "\n".join(lines)


# ===============================
# Main CLI
# ===============================
def main():
    print("=" * 60)
    print("🤖 Ollama + MongoDB Q&A (type exit() to quit)")
    print("=" * 60)

    chat_history = [{"role": "system", "content": "You are a helpful assistant."}]

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == "exit()":
                print("👋 Goodbye")
                break

            if not user_input:
                continue

            # New structure: use LLM to plan tasks (intent/table + bank_code), then query DB, then answer.
            # Validate LLM output against heuristic classifier to prevent misrouting
            try:
                planned = _llm_plan_tasks(user_input)
                
                # Guardrail: validate LLM intent against heuristic classifier.
                # Trust the LLM when it says GENERAL (non-banking question) — the LLM understands
                # context better than keyword matching. Only correct the LLM when it picks a
                # *wrong* banking intent (e.g. LOAN when keywords clearly indicate FX).
                for task in planned:
                    q = task.get("question") or user_input
                    heuristic_intents = classify_intents(q)
                    llm_intent = task.get("intent")

                    # If LLM picked a specific banking intent that differs from a confident
                    # heuristic, override — but NEVER promote GENERAL to a banking intent here.
                    if (
                        llm_intent != "GENERAL"
                        and heuristic_intents
                        and len(heuristic_intents) == 1
                        and llm_intent not in heuristic_intents
                        and heuristic_intents[0] in {"FX", "LOAN", "FIXED_DEPOSIT", "SAVINGS", "CREDIT_CARD"}
                    ):
                        task["intent"] = heuristic_intents[0]
            except Exception as e:
                # Fallback to heuristic Approach B if router fails.
                print(f"⚠️ LLM router failed ({str(e)}), using heuristic classifier")
                planned = []
                for q in split_user_question(user_input):
                    for it in classify_intents(q):
                        planned.append({"intent": it, "question": q, "bank_codes": None, "currency": None})

            # Consolidate round-trip FX so we don't produce multiple contradictory FX answers.
            if _is_fx_roundtrip_text(user_input):
                planned = [{"intent": "FX", "question": user_input, "bank_codes": None, "currency": None}]

            answers: List[str] = []
            for t in planned:
                intent = (t.get("intent") or "GENERAL").strip().upper()
                q = (t.get("question") or user_input).strip()
                # Guardrail: don't allow the router to rewrite the user's question.
                # If the planned question isn't literally present in the user's input, use the user's input.
                if user_input and q:
                    if q.lower() not in user_input.lower():
                        # If we're handling multiple tasks, keep the router's split when it looks plausible.
                        # Otherwise (or when doubtful), fall back to the original user input.
                        if len(planned) == 1:
                            q = user_input
                        else:
                            # If even token overlap is tiny, fall back.
                            qt = set(re.findall(r"\b[a-z0-9]+\b", q.lower()))
                            ut = set(re.findall(r"\b[a-z0-9]+\b", user_input.lower()))
                            overlap = len(qt & ut)
                            if overlap < 2:
                                q = user_input
                bank_codes = t.get("bank_codes")
                currency = t.get("currency")

                # Guardrail: if the LLM router picked a specific banking intent that contradicts
                # the heuristic, correct it — but never override GENERAL, since GENERAL means
                # the LLM decided this is NOT a banking/DB question and should answer directly.
                if intent != "GENERAL":
                    heur = classify_intents(q)
                    if heur and heur[0] != "GENERAL" and intent not in heur:
                        intent = heur[0]

                # Fill missing bank_codes/currency from deterministic parsers.
                if not bank_codes:
                    bank_codes = extract_bank_codes_from_text(q) or None
                if not currency:
                    currency = parse_currency_hint(q)

                if intent == "FX":
                    ans = answer_fx_question(q)
                    label = "FX"
                elif intent == "FIXED_DEPOSIT":
                    ans = answer_fixed_deposit_question(q)
                    label = "FIXED_DEPOSIT"
                elif intent == "SAVINGS":
                    ans = answer_savings_question(q, bank_codes=bank_codes, currency=currency)
                    label = "SAVINGS"
                elif intent == "INTEREST":
                    ans = answer_interest_question(q)
                    label = "INTEREST"
                elif intent == "LOAN":
                    ans = answer_loan_question(q, bank_codes=bank_codes, currency=currency)
                    label = "LOAN"
                elif intent == "CREDIT_CARD":
                    ans = answer_credit_card_question(q)
                    label = "CREDIT_CARD"
                elif intent == "BANK":
                    ans = answer_bank_question(q)
                    label = "BANK"
                else:
                    # General chat uses conversation context only when it's the only task.
                    if len(planned) == 1:
                        chat_history.append({"role": "user", "content": q})
                        ans = ollama_chat(chat_history)
                        chat_history.append({"role": "assistant", "content": ans})
                    else:
                        ans = ollama_chat(
                            [
                                {"role": "system", "content": "Be concise."},
                                {"role": "user", "content": q},
                            ]
                        )
                    label = "GENERAL"

                answers.append(f"[{label}] {ans}")

            print("\nOllama:\n" + "\n\n".join(answers))

        except KeyboardInterrupt:
            print("\n👋 Interrupted. Exiting...")
            sys.exit(0)
        except Exception as e:
            print("❌ Error:", e)


if __name__ == "__main__":
    main()
