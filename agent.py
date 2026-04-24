"""
TapeDeck — institutional-grade financial research via OpenBB MCP.

Architecture:
  • Fault-tolerant tool wrapper — NEVER raises. Returns error strings on
    exhaustion so the LLM adapts instead of the run dying.
  • Data-provider fallback (fmp → yfinance → tiingo) — transparent, broad
    error signal matching (paywall, auth, credentials, gateway).
  • Tool result cache — shared across LLM fallback chain.
  • LLM fallback across 8 free OpenRouter tool-calling models.
  • Recursion limit 100 to support 10-25 tool call research flows.

Deployment modes:
  • Local CLI:   `python agent.py "analyze MU"` — uses .env + local venv binary
  • Streamlit:   imported by app.py — uses st.secrets + pip-installed binary
  • The secrets ladder (st.secrets → .env → os.environ) works in both.
"""

import asyncio
import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_mcp_adapters.client import MultiServerMCPClient


WORKSPACE = Path(__file__).resolve().parent
load_dotenv(WORKSPACE / ".env")


# =============================================================================
# SECRETS LADDER — works identically local + Streamlit Cloud
# =============================================================================
# Order: st.secrets (when hosted on Streamlit) → .env (local dev) → os.environ

def get_secret(key: str, default=None):
    """Ladder: Streamlit secrets → .env → os.environ → default.

    Safe to call from both CLI and Streamlit runtime. The streamlit import
    only fires if it's already loaded in the process (i.e. we're inside
    `streamlit run app.py`). CLI runs never touch it.
    """
    # Only check st.secrets if streamlit is loaded — avoids import cost
    # and FileNotFoundError when there's no secrets.toml locally.
    if "streamlit" in sys.modules:
        try:
            import streamlit as st
            if hasattr(st, "secrets") and key in st.secrets:
                return st.secrets[key]
        except Exception:
            pass
    return os.getenv(key, default)


def configure_openbb_credentials():
    """Write API keys from the secrets ladder into OpenBB's credential store.

    On local dev this is a no-op because user_settings.json is symlinked at
    ~/.openbb_platform/ with the real keys. On Streamlit Cloud there's no
    JSON file — we programmatically push keys from st.secrets into OpenBB
    at startup so the spawned MCP subprocess inherits them via env vars.

    Set env vars BEFORE the MCP subprocess spawns. OpenBB providers read
    from OPENBB_* env vars as a fallback when user_settings.json is absent.
    """
    key_map = {
        "FMP_API_KEY":      "OPENBB_API_FMP_API_KEY",
        "FRED_API_KEY":     "OPENBB_API_FRED_API_KEY",
        "TIINGO_TOKEN":     "OPENBB_API_TIINGO_TOKEN",
        "BENZINGA_API_KEY": "OPENBB_API_BENZINGA_API_KEY",
    }
    configured = []
    for user_key, openbb_env_key in key_map.items():
        value = get_secret(user_key)
        if value:
            os.environ[openbb_env_key] = str(value)
            # Also set the short form OpenBB reads in some code paths
            os.environ[user_key.lower()] = str(value)
            configured.append(user_key)
    return configured


# =============================================================================
# OPENBB-MCP BINARY RESOLVER — works local + Streamlit Cloud
# =============================================================================

def resolve_openbb_bin() -> str:
    """Find the openbb-mcp binary across deployment environments.

    Resolution order:
      1. Local dev venv inside repo: ./environ/bin/openbb-mcp (or symlink)
      2. PATH resolution (Streamlit Cloud after pip install)
      3. Sibling to current Python interpreter (pip install --user cases)
    """
    # 1. Local dev — venv inside repo, symlinked or real
    local = WORKSPACE / "environ" / "bin" / "openbb-mcp"
    if local.exists():
        return str(local)

    # 2. PATH — Streamlit Cloud installs to the container's Python bin dir
    which = shutil.which("openbb-mcp")
    if which:
        return which

    # 3. Sibling to sys.executable (belt-and-suspenders)
    sibling = Path(sys.executable).parent / "openbb-mcp"
    if sibling.exists():
        return str(sibling)

    raise SystemExit(
        "openbb-mcp binary not found. Install with: pip install openbb-mcp-server"
    )


OPENBB_BIN = resolve_openbb_bin()
RECURSION_LIMIT = 100

HARD_DROPS = {
    "list_prompts", "get_prompt", "list_resources",
    "read_resource", "install_skill",
}

# =============================================================================
# FAULT-TOLERANT TOOL WRAPPER — CORE FIX
# =============================================================================
# Key invariant: this wrapper NEVER raises. Every exception path returns a
# string the LLM can read as a tool result. One failed tool = one adapted
# LLM turn, not a dead run.

_TOOL_CACHE = {}  # (tool_name, json_kwargs) → result — shared across LLM retries

PROVIDER_FALLBACKS = {
    "fmp":      ["yfinance", "tiingo", "intrinio"],
    "intrinio": ["fmp", "yfinance", "tiingo"],
    "benzinga": ["tiingo", "fmp", "yfinance"],
    "tiingo":   ["yfinance", "fmp"],
    "polygon":  ["fmp", "yfinance"],
    "fred":     ["federal_reserve", "oecd"],
}

# Anything matching these phrases → try an alternate provider. Kept broad
# because OpenBB providers return error text in many formats (402, 403, 400,
# wrapped as 502 Bad Gateway, credential errors, etc).
PROVIDER_ERROR_SIGNALS = (
    # paywall/auth/rate
    "premium query parameter", "special endpoint", "restricted endpoint",
    "upgrade your plan", "subscription",
    "missing credential",
    "unauthorized fmp", "unauthorized tiingo",
    "unauthorized benzinga", "unauthorized intrinio",
    "invalid api key", "api key",
    "402", "403", "429",
    "rate limit", "exceeded",
    "bad gateway", "502",
    # schema-validation (means "this provider doesn't fit, try another")
    "422", "unprocessable entity", "literal_error",
    "less_than_equal",
)


def _is_provider_error(err_str: str) -> bool:
    low = err_str.lower()
    return any(sig in low for sig in PROVIDER_ERROR_SIGNALS)


def _cache_key(tool_name, kwargs):
    return (tool_name, json.dumps(kwargs, sort_keys=True, default=str))


def _format_error_for_llm(tool_name, err, attempted_providers):
    """Turn an exhausted-fallback exception into a short, actionable string
    the LLM can read as a tool result."""
    err_short = str(err)[:250].replace("\n", " ")
    providers_str = ", ".join(attempted_providers) if attempted_providers else "default"
    return (
        f"⚠ Tool '{tool_name}' failed across providers [{providers_str}]. "
        f"Error: {err_short}. "
        f"Try a DIFFERENT tool for this data, or proceed without this angle. "
        f"Do not retry this exact tool with the same arguments."
    )


# Optional stream callback — Streamlit UI sets this to render fallbacks live.
# Signature: (level: str, message: str) -> None
_STREAM_CALLBACK = None


def set_stream_callback(cb):
    """UI layer (Streamlit) calls this to receive fallback/cache events."""
    global _STREAM_CALLBACK
    _STREAM_CALLBACK = cb


def _emit(level: str, message: str):
    """Emit to stream callback if set, else print (CLI path)."""
    if _STREAM_CALLBACK:
        try:
            _STREAM_CALLBACK(level, message)
            return
        except Exception:
            pass
    print(message)


def wrap_tool(t):
    if not hasattr(t, "coroutine") or t.coroutine is None:
        return t

    original_coro = t.coroutine
    tool_name = t.name
    # Detect expected return shape (LangChain supports content vs. content+artifact)
    response_format = getattr(t, "response_format", "content")

    def _shape(payload):
        """Return error payload in the shape langchain expects for this tool."""
        if response_format == "content_and_artifact":
            return (payload, None)
        return payload

    async def wrapped(*args, **kwargs):
        key = _cache_key(tool_name, kwargs)
        if key in _TOOL_CACHE:
            return _TOOL_CACHE[key]

        attempted = []
        provider = kwargs.get("provider")
        if provider:
            attempted.append(provider)

        try:
            result = await original_coro(*args, **kwargs)
            _TOOL_CACHE[key] = result
            return result
        except Exception as e:
            primary_err = e
            err_str = str(e)
            if not provider or not _is_provider_error(err_str):
                return _shape(_format_error_for_llm(tool_name, e, attempted))

        alternates = PROVIDER_FALLBACKS.get(provider, [])
        last_err = primary_err
        for alt in alternates:
            attempted.append(alt)
            retry_kwargs = {**kwargs, "provider": alt}
            retry_key = _cache_key(tool_name, retry_kwargs)
            if retry_key in _TOOL_CACHE:
                _emit("cache", f"  ♻ {tool_name} ({alt}): cached")
                return _TOOL_CACHE[retry_key]
            _emit("fallback", f"  ⤷ {tool_name}: {provider} blocked → {alt}")
            try:
                result = await original_coro(*args, **retry_kwargs)
                _TOOL_CACHE[retry_key] = result
                return result
            except Exception as retry_err:
                last_err = retry_err
                continue

        return _shape(_format_error_for_llm(tool_name, last_err, attempted))

    t.coroutine = wrapped
    return t


def apply_wrappers(tools):
    count = 0
    for t in tools:
        if hasattr(t, "coroutine") and t.coroutine is not None:
            wrap_tool(t)
            count += 1
    return tools, count


# =============================================================================
# LLM PROVIDER
# =============================================================================

OPENROUTER_MODELS = {
    "gpt-oss":   "openai/gpt-oss-120b:free",
    "qwen3":     "qwen/qwen3-next-80b-a3b-instruct:free",
    "nemotron":  "nvidia/nemotron-3-super-120b-a12b:free",
    "minimax":   "minimax/minimax-m2.5:free",
    "glm":       "z-ai/glm-4.5-air:free",
    "ling":      "inclusionai/ling-2.6-1t:free",
    "gemma4":    "google/gemma-4-26b-a4b-it:free",
    "qwen-code": "qwen/qwen3-coder:free",
    "deepseek":  "deepseek/deepseek-chat-v3-0324",  # PAID (~$0.02/query)
}

# Ling proven end-to-end, Nemotron proven at parallel tool calls
DEFAULT_FALLBACK_ORDER = [
    "ling", "nemotron", "qwen3", "minimax",
    "glm", "gpt-oss", "gemma4", "qwen-code",
]


def build_model(provider: str, variant: str):
    if provider == "openrouter":
        from langchain_openai import ChatOpenAI
        api_key = get_secret("OPENROUTER_API_KEY")
        if not api_key:
            raise SystemExit("OPENROUTER_API_KEY missing from secrets/.env")
        model_name = OPENROUTER_MODELS.get(variant, variant)
        return ChatOpenAI(
            model=model_name,
            temperature=0,
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        ), model_name
    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        print("⚠  Gemini has int-enum schema issues with OpenBB.")
        api_key = get_secret("GOOGLE_API_KEY") or get_secret("GEMINI_API_KEY")
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            temperature=0,
            google_api_key=api_key,
        ), "gemini-2.5-flash"
    elif provider == "groq":
        from langchain_groq import ChatGroq
        api_key = get_secret("GROQ_API_KEY")
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,
            groq_api_key=api_key,
        ), "llama-3.3-70b-versatile"
    else:
        raise SystemExit(f"Unknown provider: {provider}")


# =============================================================================
# TOOL CATALOG
# =============================================================================

def default_toolset(all_tools, max_tools=127):
    kept = [t for t in all_tools if t.name not in HARD_DROPS]
    if len(kept) > max_tools:
        deprioritize = {
            "fixedincome_rate_sonia", "fixedincome_rate_estr",
            "fixedincome_rate_ecb", "fixedincome_rate_ameribor",
            "economy_shipping_port_info", "economy_shipping_port_volume",
            "economy_shipping_chokepoint_info", "economy_shipping_chokepoint_volume",
            "economy_survey_sloos", "economy_survey_nonfarm_payrolls",
            "economy_survey_economic_conditions_chicago",
            "economy_survey_manufacturing_outlook_texas",
            "economy_survey_manufacturing_outlook_ny",
            "economy_survey_inflation_expectations",
            "economy_survey_university_of_michigan",
            "economy_survey_bls_series", "economy_survey_bls_search",
            "regulators_sec_schema_files", "regulators_sec_sic_search",
            "regulators_sec_rss_litigation", "regulators_sec_institutions_search",
            "fixedincome_government_treasury_auctions",
            "fixedincome_government_svensson_yield_curve",
            "fixedincome_corporate_hqm", "fixedincome_corporate_spot_rates",
            "fixedincome_corporate_commercial_paper",
            "fixedincome_bond_indices", "fixedincome_mortgage_indices",
            "fixedincome_spreads_tcm", "fixedincome_spreads_tcm_effr",
            "fixedincome_spreads_treasury_effr",
            "economy_primary_dealer_positioning", "economy_primary_dealer_fails",
            "economy_central_bank_holdings", "economy_composite_leading_indicator",
            "economy_total_factor_productivity", "economy_direction_of_trade",
            "economy_export_destinations", "economy_share_price_index",
            "economy_house_price_index", "economy_retail_prices",
            "commodity_psd_data", "commodity_psd_report",
            "commodity_weather_bulletins", "commodity_weather_bulletins_download",
            "commodity_petroleum_status_report", "commodity_short_term_energy_outlook",
            "imf_utils_list_dataflows", "imf_utils_get_dataflow_dimensions",
            "imf_utils_list_port_id_choices", "imf_utils_list_tables",
            "imf_utils_list_table_choices", "imf_utils_list_dataflow_choices",
            "imf_utils_presentation_table_choices", "imf_utils_presentation_table",
            "uscongress_bills", "uscongress_bill_text_urls",
            "uscongress_bill_info", "uscongress_bill_text",
        }
        kept = [t for t in kept if t.name not in deprioritize]
        if len(kept) > max_tools:
            kept = kept[:max_tools]
    return kept


def make_broaden_tool(full_tool_catalog):
    catalog_str = "\n".join(
        f"  • {t.name}: {(t.description or '').strip().splitlines()[0][:120]}"
        for t in full_tool_catalog
    )

    @tool
    def broaden_toolset() -> str:
        """Return the full 189-tool OpenBB catalog."""
        return f"Full OpenBB MCP catalog (189 tools):\n\n{catalog_str}"

    return broaden_toolset


# =============================================================================
# SYSTEM PROMPT — INSTITUTIONAL-GRADE RESEARCH
# =============================================================================

SYSTEM_PROMPT = """\
You are TapeDeck, a senior financial research analyst with direct access to
OpenBB's full data platform (189 tools). Institutional-grade output,
not surface-level summaries.

═══════════════════════════════════════════════════════════════════════
RESEARCH DEPTH PROTOCOL
═══════════════════════════════════════════════════════════════════════

NARROW/FACTUAL ("pull X's Q3 revenue", "get dividend history"):
→ 1-3 tool calls, concise answer.

THESIS/ANALYSIS ("should I invest", "analyze trajectory", "deep dive",
"evaluate", "is X a good entry", "full picture"):
→ 10-25 tool calls across MULTIPLE categories, structured output
  (400-800 words). Investigate at least 5 of the 7 angles below.

═══════════════════════════════════════════════════════════════════════
THESIS FRAMEWORK — 7 angles
═══════════════════════════════════════════════════════════════════════

1. FUNDAMENTALS: equity_fundamental_income, _balance, _cash, _ratios,
   _metrics, _revenue_per_segment/_geography. Period='quarter', limit=8-12.

2. VALUATION: equity_fundamental_metrics (current multiples),
   equity_estimates_consensus, _price_target,
   equity_historical_market_cap.

3. FORWARD ESTIMATES: equity_estimates_consensus, _price_target,
   _historical (beat/miss track record), equity_calendar_earnings.

4. OWNERSHIP: equity_ownership_insider_trading, _institutional,
   _major_holders, _form_13f, _government_trades.

5. NEWS: news_company (limit=20), equity_fundamental_transcript.

6. MACRO (for cyclicals — semis, autos, banks, housing):
   economy_fred_series with series_id — e.g., semis: "IPB50001N",
   housing: "HOUST", consumer: "UMCSENT". Also economy_cpi,
   economy_unemployment, fixedincome_rate_effr.

7. PEERS & PRICE: equity_compare_peers, equity_price_historical
   (period='2y'), equity_price_performance,
   derivatives_options_chains.

═══════════════════════════════════════════════════════════════════════
CRITICAL: DATA PROVIDER RULES
═══════════════════════════════════════════════════════════════════════

PREFERRED provider by tool category (free tier compatible):
  • equity_fundamental_*     → provider='fmp' (auto-falls to yfinance)
  • equity_price_*           → provider='yfinance' (most reliable)
  • equity_estimates_*       → provider='fmp' first, try 'yfinance' on fail
  • equity_ownership_*       → provider='fmp' (auto-falls to sec)
  • news_company             → provider='tiingo' or 'benzinga'
  • economy_fred_series      → provider='fred'
  • equity_compare_*         → provider='fmp' (auto-falls to finviz/yfinance)

AVOID as primary provider (no credentials configured):
  • intrinio, polygon

If a tool returns "⚠ Tool 'X' failed across providers [...]", that means
all fallbacks exhausted. Do NOT retry the same tool. Instead:
  - Try a DIFFERENT tool for similar data, OR
  - Proceed without that angle and note the data gap in your synthesis.

The system automatically retries alternate data providers for you.
Just pick the preferred one above and move on.

═══════════════════════════════════════════════════════════════════════
OUTPUT STRUCTURE (thesis questions)
═══════════════════════════════════════════════════════════════════════

**[Ticker] — Research Brief**

**Executive Summary** (3-5 lines — thesis in one paragraph)

**Fundamentals**
- Revenue/margin trajectory with specific numbers
- Balance sheet (cash, debt, leverage)
- Free cash flow trend

**Valuation**
- Current multiples vs historical vs peers
- Forward estimates, implied growth
- Consensus target and upside/downside

**Ownership & Flows**
- Insider activity (last 3-6 months, with $ where available)
- Institutional changes, notable holders

**News & Catalysts**
- 3-5 meaningful recent headlines
- Next earnings, material catalysts

**Macro Context** (if cyclical)
- Relevant cycle indicators

**Bull Case** — 3-5 specific points, each anchored to data you pulled
**Bear Case** — 3-5 specific points, each anchored to data
**Balanced Take** — synthesis weighing both; metrics to watch;
  base/bull/bear 12-month scenarios if valuation data permits.

End with: *"This is research and analysis, not investment advice.
Verify data and do your own diligence before any position change."*

═══════════════════════════════════════════════════════════════════════
HARD RULES
═══════════════════════════════════════════════════════════════════════

- NEVER fabricate. If a tool returns "⚠ Tool failed", state the gap.
- Ground every claim in specific numbers from tool output.
- For thesis questions, DO NOT shortcut — pull the full picture.
- If ≥4 tools fail, acknowledge the data limitations in output but
  still produce the best analysis possible from what you got.
"""


# =============================================================================
# RUNTIME
# =============================================================================

async def stream_agent(agent, prompt: str, config=None, on_event=None):
    """Stream agent turns. If on_event is provided, emit structured events
    to it for UI rendering; otherwise print to stdout (CLI mode).

    Event shapes emitted to on_event:
      {"type": "tool_call", "name": str, "args": dict}
      {"type": "tool_result", "name": str, "content": str, "is_error": bool}
      {"type": "assistant_message", "content": str}
    """
    config = config or {}
    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": prompt}]},
        stream_mode="values",
        config=config,
    ):
        msg = chunk["messages"][-1]
        msg_type = type(msg).__name__
        if msg_type == "AIMessage":
            tool_calls = getattr(msg, "tool_calls", None) or []
            for tc in tool_calls:
                if on_event:
                    on_event({"type": "tool_call", "name": tc["name"],
                              "args": tc.get("args", {})})
                else:
                    args_str = str(tc.get("args", {}))[:150]
                    print(f"🔧 {tc['name']}({args_str})")
            if msg.content and not tool_calls:
                if on_event:
                    on_event({"type": "assistant_message", "content": msg.content})
                else:
                    print(f"\n💬 {msg.content}\n")
        elif msg_type == "ToolMessage":
            content = str(msg.content)
            is_error = content.startswith("⚠")
            if on_event:
                on_event({"type": "tool_result", "name": msg.name,
                          "content": content, "is_error": is_error})
            else:
                content_flat = content.replace("\n", " ")[:180]
                if is_error:
                    print(f"📨 {msg.name} → {content_flat}")
                else:
                    print(f"📨 {msg.name} → {content_flat[:100]}...")


def is_fatal_llm_error(err: Exception) -> bool:
    """LLM-provider auth errors only. Tool errors no longer reach here."""
    err_cls = type(err).__name__
    if err_cls in ("AuthenticationError", "PermissionDeniedError"):
        return True
    msg = str(err).lower()
    if "openrouter api key" in msg or "openai api key" in msg:
        return True
    return False


async def run_with_fallback(prompt, provider, variant, agent_tools, on_event=None):
    """LLM fallback chain. Hits `variant` first, then cascades through
    DEFAULT_FALLBACK_ORDER on non-fatal errors.

    If on_event is provided, structured events go there for UI rendering.
    Otherwise prints to stdout (CLI mode).
    """
    config = {"recursion_limit": RECURSION_LIMIT}

    def _emit_model_event(model_name, status):
        if on_event:
            on_event({"type": "model_status", "model": model_name, "status": status})
        else:
            if status == "trying":
                print(f"\n→ trying LLM: {model_name}")
            elif status == "succeeded":
                print(f"\n✓ succeeded with {model_name} "
                      f"(cached {len(_TOOL_CACHE)} unique tool results)")

    if provider != "openrouter":
        model, model_name = build_model(provider, variant)
        _emit_model_event(model_name, "trying")
        agent = create_agent(model, agent_tools, system_prompt=SYSTEM_PROMPT)
        if not on_event:
            print("─" * 70)
        await stream_agent(agent, prompt, config=config, on_event=on_event)
        _emit_model_event(model_name, "succeeded")
        return

    order = [variant] + [v for v in DEFAULT_FALLBACK_ORDER if v != variant]
    errors = []

    for v in order:
        model_name = OPENROUTER_MODELS.get(v, v)
        _emit_model_event(model_name, "trying")
        try:
            model, _ = build_model(provider, v)
            agent = create_agent(model, agent_tools, system_prompt=SYSTEM_PROMPT)
            if not on_event:
                print("─" * 70)
            await stream_agent(agent, prompt, config=config, on_event=on_event)
            _emit_model_event(model_name, "succeeded")
            return
        except Exception as e:
            err_cls = type(e).__name__
            msg = str(e)[:180]
            if is_fatal_llm_error(e):
                if on_event:
                    on_event({"type": "fatal_error",
                              "message": f"{err_cls}: {msg}"})
                else:
                    print(f"  ✗ FATAL {err_cls}: {msg}")
                raise
            if on_event:
                on_event({"type": "model_status", "model": model_name,
                          "status": "failed", "error": f"{err_cls}: {msg}"})
            else:
                print(f"  ✗ {err_cls}: {msg} — falling through")
            errors.append((v, err_cls))
            continue

    exhausted_msg = ("All OpenRouter free variants exhausted. "
                     "Options: wait 5 min, or use --variant deepseek (paid).")
    if on_event:
        on_event({"type": "fatal_error", "message": exhausted_msg})
    else:
        print(f"\n❌ {exhausted_msg}")
        for v, cls in errors:
            print(f"   {v}: {cls}")


async def build_agent_tools():
    """Spawn OpenBB MCP subprocess, fetch tools, apply wrappers. Returns
    (agent_tools, total_count, active_count, wrapped_count).
    """
    client = MultiServerMCPClient({
        "openbb": {
            "transport": "stdio",
            "command": str(OPENBB_BIN),
            "args": ["--transport", "stdio"],
        }
    })

    all_tools = await client.get_tools()
    default_tools = default_toolset(all_tools)
    default_tools, wrapped_count = apply_wrappers(default_tools)
    broaden = make_broaden_tool(all_tools)
    agent_tools = default_tools + [broaden]
    return agent_tools, len(all_tools), len(agent_tools), wrapped_count


async def main(prompt: str, provider: str, variant: str):
    configured = configure_openbb_credentials()
    if configured:
        print(f"→ configured OpenBB credentials: {', '.join(configured)}")

    agent_tools, total, active, wrapped = await build_agent_tools()

    print(f"\n→ provider: {provider}")
    print(f"→ {total} tools available, {active} active "
          f"({wrapped} wrapped fault-tolerant)")
    print(f"→ recursion limit: {RECURSION_LIMIT} (deep research enabled)")

    await run_with_fallback(prompt, provider, variant, agent_tools)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("prompt", nargs="?", help="Research question")
    parser.add_argument("--provider", default="openrouter",
                        choices=["openrouter", "gemini", "groq"])
    parser.add_argument("--variant", default="ling",
                        help="Preferred LLM.")
    args = parser.parse_args()
    prompt_text = args.prompt or input("Prompt: ")
    asyncio.run(main(prompt_text, args.provider, args.variant))