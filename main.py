"""Rack Trainer Pro — Weekly Review & Chat API.

Standalone FastAPI service for coaching features.
Calls Claude with prompt caching for cost-efficient analysis.
Tracks per-user token spend with a configurable monthly cap.
"""

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timezone

import anthropic
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="Rack Trainer Pro API", version="2.0.0")
client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

VALID_AUTH_TOKEN = os.getenv("RACK_AUTH_TOKEN")

# Monthly spend cap per user in GBP. Default £5.
MONTHLY_CAP_GBP = float(os.getenv("MONTHLY_CAP_GBP", "5.0"))

# Pricing per million tokens (USD) — convert to GBP at ~0.79
USD_TO_GBP = float(os.getenv("USD_TO_GBP", "0.79"))

PRICING = {
    "claude-opus-4-20250514": {
        "input": 15.0,
        "output": 75.0,
        "cache_read": 1.5,
        "cache_write": 18.75,
    },
    "claude-sonnet-4-6": {
        "input": 3.0,
        "output": 15.0,
        "cache_read": 0.30,
        "cache_write": 3.75,
    },
}


# ---------------------------------------------------------------------------
# SQLite usage tracking
# ---------------------------------------------------------------------------

DB_PATH = os.getenv("USAGE_DB_PATH", "/data/usage.db")


def _init_db():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                month TEXT NOT NULL,
                model TEXT NOT NULL,
                input_tokens INTEGER NOT NULL,
                output_tokens INTEGER NOT NULL,
                cache_read_tokens INTEGER DEFAULT 0,
                cache_write_tokens INTEGER DEFAULT 0,
                cost_gbp REAL NOT NULL,
                endpoint TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_usage_user_month
            ON usage (user_id, month)
        """)


@contextmanager
def _db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _current_month() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m")


def _calculate_cost_gbp(model: str, usage: anthropic.types.Usage) -> float:
    prices = PRICING.get(model, PRICING["claude-sonnet-4-6"])
    input_cost = (usage.input_tokens / 1_000_000) * prices["input"]
    output_cost = (usage.output_tokens / 1_000_000) * prices["output"]

    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
    cache_read_cost = (cache_read / 1_000_000) * prices["cache_read"]
    cache_write_cost = (cache_write / 1_000_000) * prices["cache_write"]

    total_usd = input_cost + output_cost + cache_read_cost + cache_write_cost
    return round(total_usd * USD_TO_GBP, 6)


def get_user_spend(user_id: str, month: str | None = None) -> float:
    month = month or _current_month()
    with _db() as conn:
        row = conn.execute(
            "SELECT COALESCE(SUM(cost_gbp), 0) as total FROM usage WHERE user_id = ? AND month = ?",
            (user_id, month),
        ).fetchone()
        return row["total"]


def record_usage(
    user_id: str,
    model: str,
    usage: anthropic.types.Usage,
    endpoint: str,
):
    cost = _calculate_cost_gbp(model, usage)
    cache_read = getattr(usage, "cache_read_input_tokens", 0) or 0
    cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0

    with _db() as conn:
        conn.execute(
            """INSERT INTO usage
               (user_id, month, model, input_tokens, output_tokens,
                cache_read_tokens, cache_write_tokens, cost_gbp, endpoint, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                user_id,
                _current_month(),
                model,
                usage.input_tokens,
                usage.output_tokens,
                cache_read,
                cache_write,
                cost,
                endpoint,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
    return cost


def check_budget(user_id: str):
    spent = get_user_spend(user_id)
    if spent >= MONTHLY_CAP_GBP:
        raise HTTPException(
            status_code=429,
            detail=f"Monthly spending cap reached (£{spent:.2f}/£{MONTHLY_CAP_GBP:.2f}). Resets next month.",
        )
    return spent


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class WeeklyReviewRequest(BaseModel):
    programId: str
    week: int
    summaryData: dict
    tier2Context: dict
    measurements: dict
    modelHint: str = "opus"


class WeeklyReviewData(BaseModel):
    weekSummary: str
    analysisText: str
    highlights: list[str]
    recommendations: list[str]
    nextWeekPreview: str


class WeeklyReviewResponse(BaseModel):
    status: str = "success"
    data: WeeklyReviewData
    creditsRemaining: float | None = None
    monthlySpend: float | None = None
    monthlyCap: float | None = None


class ChatRequest(BaseModel):
    message: str
    userContext: dict = {}
    tier1Context: dict = {}
    tier2Context: dict = {}
    conversationHistory: list[dict] | None = None
    modelHint: str | None = None


class ChatResponse(BaseModel):
    status: str = "success"
    data: dict
    creditsRemaining: float | None = None
    messagesRemaining: int | None = None
    monthlySpend: float | None = None
    monthlyCap: float | None = None


class AnalyzePlateauRequest(BaseModel):
    liftName: str
    currentWeight: float
    weeksStuck: int
    weeklyVolume: int | None = None
    sleepAvg: float | None = None
    userContext: dict = {}
    exerciseHistory: list[dict] | None = None


# ---------------------------------------------------------------------------
# Auth helper
# ---------------------------------------------------------------------------

def verify_auth(authorization: str | None) -> str:
    """Returns user_id extracted from token (or a default for dev)."""
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing auth token")
    token = authorization.removeprefix("Bearer ")
    if VALID_AUTH_TOKEN and token != VALID_AUTH_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid auth token")
    # In production, decode JWT to get real userId.
    # For now, use a hash of the token as userId.
    return f"user-{hash(token) % 100000:05d}"


# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

WEEKLY_REVIEW_SYSTEM = """You are Rack Trainer Pro, an expert strength & conditioning coach built into a gym tracking app.

You are generating a structured weekly review for a user's training programme. Be specific, data-driven, and actionable.

## Output format
Return valid JSON with exactly these keys:
{
  "weekSummary": "2-3 sentence overview of how the week went",
  "analysisText": "Detailed analysis (3-5 paragraphs). Cover: training volume, progression, adherence, any concerns. Reference specific exercises and numbers where possible.",
  "highlights": ["Up to 4 short bullet points of notable achievements or observations"],
  "recommendations": ["3-5 specific, actionable recommendations for the coming week"],
  "nextWeekPreview": "What to expect next week based on the programme phase, upcoming deloads, etc."
}

## Guidelines
- Use UK English spelling (programme, organisation, favour)
- Be encouraging but honest — don't sugarcoat poor adherence
- Reference the user's actual data (weights, sets, PRs) not generic advice
- If the user has stalled exercises, address them specifically
- Consider the programme phase (accumulation/intensification/peak/deload) in your advice
- Keep recommendations concrete: "Add a 4th set of Romanian Deadlifts" not "Consider increasing volume"
- If it's a deload week, adjust expectations accordingly
"""

CHAT_SYSTEM = """You are Rack Trainer Pro, an expert strength & conditioning coach built into a gym tracking app.

You help users with training advice, form guidance, programme adjustments, and motivation. You have access to their training data, measurements, and programme details.

## Guidelines
- Use UK English spelling (programme, organisation, favour)
- Be concise — users are reading on their phone between sets
- Reference their actual data when relevant (weights, PRs, measurements, trends)
- Be encouraging but honest
- Give specific, actionable advice
- If asked about measurements or body composition, reference the historical data provided
- Keep responses under 200 words unless the question requires more detail
- End when the answer is complete. Do not ask follow-up questions or invite further conversation.
"""

PLATEAU_SYSTEM = """You are Rack Trainer Pro, an expert strength & conditioning coach built into a gym tracking app.

You are analysing a plateau for a specific lift. Provide detailed, actionable analysis.

## Output format
Return valid JSON with exactly these keys:
{
  "analysis": "Detailed paragraph explaining why the plateau likely occurred",
  "recommendations": ["3-5 specific, actionable recommendations to break through"],
  "summary": "One-sentence summary with primary recommendation"
}

## Guidelines
- Use UK English spelling (programme, organisation, favour)
- Be specific to the lift in question
- Consider common plateau causes: insufficient volume, poor recovery, technique breakdown, neural fatigue, inadequate nutrition
- If exercise history is provided, look for patterns (stall duration, rep trends, weight progression rate)
- Reference actual numbers from their data
- Recommendations should be concrete: "Add 2 back-off sets of 8 at 75%" not "Consider adding volume"
"""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/api/coaching/weekly-review", response_model=WeeklyReviewResponse)
async def weekly_review(
    req: WeeklyReviewRequest,
    authorization: str | None = Header(default=None),
):
    user_id = verify_auth(authorization)
    spent = check_budget(user_id)

    model = "claude-opus-4-20250514"

    # Tiered content blocks — tier 1 gets a cache breakpoint in case of
    # retries within the 5-minute TTL (unlikely for weekly reviews, but free).
    tier1_data = json.dumps(
        {"programId": req.programId, "week": req.week, "measurements": req.measurements},
        indent=2,
    )
    tier2_data = json.dumps(
        {"summary": req.summaryData, "recentTraining": req.tier2Context},
        indent=2,
    )

    message = client.messages.create(
        model=model,
        max_tokens=2048,
        system=[
            {
                "type": "text",
                "text": WEEKLY_REVIEW_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"[Programme & Measurements]\n{tier1_data}",
                        "cache_control": {"type": "ephemeral"},
                    },
                    {
                        "type": "text",
                        "text": f"[Training Data]\n{tier2_data}",
                    },
                ],
            }
        ],
    )

    cost = record_usage(user_id, model, message.usage, "weekly-review")
    new_spend = spent + cost

    raw = message.content[0].text

    # Strip markdown fences if present
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=502,
            detail="Failed to parse AI response as JSON",
        )

    return WeeklyReviewResponse(
        data=WeeklyReviewData(**parsed),
        monthlySpend=round(new_spend, 2),
        monthlyCap=MONTHLY_CAP_GBP,
    )


@app.post("/api/coaching/chat", response_model=ChatResponse)
async def chat(
    req: ChatRequest,
    authorization: str | None = Header(default=None),
):
    user_id = verify_auth(authorization)
    spent = check_budget(user_id)

    model = "claude-sonnet-4-6"

    # --- Tiered context with cache breakpoints ---
    # Prefer split tiers; fall back to merged userContext for older clients.
    tier1 = req.tier1Context or {}
    tier2 = req.tier2Context or {}
    if not tier1 and not tier2 and req.userContext:
        tier1 = req.userContext

    context_blocks: list[dict] = []
    if tier1:
        context_blocks.append(
            {
                "type": "text",
                "text": f"[User Profile & Programme]\n{json.dumps(tier1, indent=2)}",
                "cache_control": {"type": "ephemeral"},
            }
        )
    if tier2:
        context_blocks.append(
            {
                "type": "text",
                "text": f"[Recent Training]\n{json.dumps(tier2, indent=2)}",
                "cache_control": {"type": "ephemeral"},
            }
        )

    # Parse conversation history
    history: list[dict] = []
    if req.conversationHistory:
        for msg in req.conversationHistory:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                history.append({"role": role, "content": content})

    # Build messages — context blocks prepended to the first user turn so the
    # prefix (system + context) stays identical across turns and gets cached.
    messages: list[dict] = []
    if history:
        first = history[0]
        if first["role"] == "user" and context_blocks:
            messages.append(
                {
                    "role": "user",
                    "content": context_blocks
                    + [{"type": "text", "text": first["content"]}],
                }
            )
            messages.extend(history[1:])
        else:
            messages.extend(history)
        messages.append({"role": "user", "content": req.message})
    elif context_blocks:
        messages.append(
            {
                "role": "user",
                "content": context_blocks
                + [{"type": "text", "text": req.message}],
            }
        )
    else:
        messages.append({"role": "user", "content": req.message})

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": CHAT_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=messages,
    )

    cost = record_usage(user_id, model, message.usage, "chat")
    new_spend = spent + cost

    return ChatResponse(
        data={"response": message.content[0].text},
        monthlySpend=round(new_spend, 2),
        monthlyCap=MONTHLY_CAP_GBP,
    )


@app.post("/api/coaching/analyze-plateau")
async def analyze_plateau(
    req: AnalyzePlateauRequest,
    authorization: str | None = Header(default=None),
):
    user_id = verify_auth(authorization)
    spent = check_budget(user_id)

    model = "claude-sonnet-4-6"

    user_data: dict = {
        "liftName": req.liftName,
        "currentWeight": req.currentWeight,
        "weeksStuck": req.weeksStuck,
    }
    if req.weeklyVolume is not None:
        user_data["weeklyVolume"] = req.weeklyVolume
    if req.sleepAvg is not None:
        user_data["sleepAvg"] = req.sleepAvg
    if req.exerciseHistory:
        user_data["exerciseHistory"] = req.exerciseHistory
    if req.userContext:
        user_data["userContext"] = req.userContext

    message = client.messages.create(
        model=model,
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": PLATEAU_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": json.dumps(user_data, indent=2)}],
    )

    cost = record_usage(user_id, model, message.usage, "analyze-plateau")
    new_spend = spent + cost

    raw = message.content[0].text
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        if raw.endswith("```"):
            raw = raw[: raw.rfind("```")]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = {"analysis": raw, "recommendations": [], "summary": raw[:200]}

    return {
        "status": "success",
        "data": parsed,
        "monthlySpend": round(new_spend, 2),
        "monthlyCap": MONTHLY_CAP_GBP,
    }


@app.get("/api/coaching/status")
async def status(authorization: str | None = Header(default=None)):
    result = {"status": "ok", "service": "rack-trainer-pro", "version": "2.0.0"}
    try:
        user_id = verify_auth(authorization)
        spent = get_user_spend(user_id)
        result["monthlySpend"] = round(spent, 2)
        result["monthlyCap"] = MONTHLY_CAP_GBP
        result["remainingBudget"] = round(MONTHLY_CAP_GBP - spent, 2)
    except HTTPException:
        pass
    return result


@app.on_event("startup")
def startup():
    _init_db()
