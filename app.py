"""
TapeDeck — Streamlit frontend.

Public demo of the TapeDeck financial research agent. Same agent.py powers it;
this file is just the UI layer. Users can either type free-form questions
(chat mode) or click preset prompt buttons for common portfolio questions.

Deployment:
  streamlit run app.py                # local
  Streamlit Cloud                     # point at this repo, add secrets
"""
import asyncio
import json
import time
from pathlib import Path

import streamlit as st

# Streamlit runs on tornado which has its own event loop. nest_asyncio
# patches asyncio to allow our cached MCP loop to be re-entered safely
# across Streamlit reruns — otherwise run_until_complete() raises
# RuntimeError and coroutines get orphaned ("never awaited" warning).
import nest_asyncio
nest_asyncio.apply()

# agent.py MUST be imported after Streamlit is in sys.modules so that
# get_secret() can find st.secrets at runtime.
import agent as tapedeck


# =============================================================================
# PAGE CONFIG + CLAUDE-INSPIRED THEME
# =============================================================================

st.set_page_config(
    page_title="TapeDeck — Personal Financial Research Terminal",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


CLAUDE_THEME_CSS = """
<style>
/* ─── Claude.ai / Anthropic light-mode aesthetic ─── */

/* Import fonts: Tiempos-style serif for headings, clean sans for body */
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,500;0,8..60,600;0,8..60,700;1,8..60,400&family=Inter:wght@400;500;600;700&display=swap');

/* Root palette — matches Anthropic brand */
:root {
    --tapedeck-bg:           #F5F1EB;  /* warm sand/cream */
    --tapedeck-bg-card:      #FAF7F2;  /* lighter cream for cards */
    --tapedeck-bg-deep:      #EDE8DF;  /* slightly deeper cream for hover */
    --tapedeck-ink:          #1F1F1E;  /* near-black warm text */
    --tapedeck-ink-soft:     #57534E;  /* muted brown-grey */
    --tapedeck-ink-faded:    #8A847A;  /* faded text */
    --tapedeck-accent:       #CC785C;  /* Claude signature terracotta */
    --tapedeck-accent-deep:  #A15A42;  /* darker terracotta for hover */
    --tapedeck-success:      #5F7A58;  /* olive green */
    --tapedeck-warn:         #C89B4B;  /* amber */
    --tapedeck-error:        #B54848;  /* warm red */
    --tapedeck-border:       #DCD5C9;  /* card border */
    --tapedeck-border-subtle:#E5DFD4;
}

/* ─── Base overrides ─── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: var(--tapedeck-bg) !important;
    color: var(--tapedeck-ink) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}

[data-testid="stHeader"] { background: transparent !important; }

/* Main content container */
.main .block-container {
    padding-top: 2rem !important;
    padding-bottom: 4rem !important;
    max-width: 1100px !important;
}

/* ─── Typography ─── */
h1 {
    font-family: 'Source Serif 4', Georgia, serif !important;
    font-weight: 600 !important;
    color: var(--tapedeck-ink) !important;
    letter-spacing: -0.02em !important;
    font-size: 2.5rem !important;
    line-height: 1.1 !important;
}

h2, h3 {
    font-family: 'Source Serif 4', Georgia, serif !important;
    color: var(--tapedeck-ink) !important;
    font-weight: 600 !important;
    letter-spacing: -0.015em !important;
}

p, li, .stMarkdown { color: var(--tapedeck-ink) !important; }

/* Tagline / subheader */
.tapedeck-tagline {
    font-family: 'Source Serif 4', Georgia, serif;
    font-size: 1.15rem;
    font-style: italic;
    color: var(--tapedeck-ink-soft);
    margin-bottom: 2rem;
    line-height: 1.5;
}

/* ─── Sidebar ─── */
[data-testid="stSidebar"] {
    background: var(--tapedeck-bg-card) !important;
    border-right: 1px solid var(--tapedeck-border-subtle) !important;
}
[data-testid="stSidebar"] .stMarkdown { color: var(--tapedeck-ink) !important; }

/* ─── Buttons ─── */
.stButton > button {
    background: var(--tapedeck-bg-card) !important;
    color: var(--tapedeck-ink) !important;
    border: 1px solid var(--tapedeck-border) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 500 !important;
    padding: 0.5rem 1rem !important;
    transition: all 0.15s ease !important;
    box-shadow: 0 1px 2px rgba(0,0,0,0.03) !important;
}
.stButton > button:hover {
    background: var(--tapedeck-bg-deep) !important;
    border-color: var(--tapedeck-accent) !important;
    color: var(--tapedeck-accent-deep) !important;
    transform: translateY(-1px) !important;
    box-shadow: 0 2px 6px rgba(204,120,92,0.12) !important;
}

/* Primary action button variant */
.stButton > button[kind="primary"],
.stButton > button.primary {
    background: var(--tapedeck-accent) !important;
    color: white !important;
    border-color: var(--tapedeck-accent) !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--tapedeck-accent-deep) !important;
    border-color: var(--tapedeck-accent-deep) !important;
}

/* ─── Input fields ─── */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div > div {
    background: var(--tapedeck-bg-card) !important;
    color: var(--tapedeck-ink) !important;
    border: 1px solid var(--tapedeck-border) !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: var(--tapedeck-accent) !important;
    box-shadow: 0 0 0 3px rgba(204,120,92,0.1) !important;
}

/* ─── Chat messages ─── */
[data-testid="stChatMessage"] {
    background: var(--tapedeck-bg-card) !important;
    border: 1px solid var(--tapedeck-border-subtle) !important;
    border-radius: 12px !important;
    padding: 1.25rem !important;
    margin-bottom: 1rem !important;
}
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li {
    color: var(--tapedeck-ink) !important;
    line-height: 1.65 !important;
}

/* Code blocks inside messages */
[data-testid="stChatMessage"] code {
    background: var(--tapedeck-bg-deep) !important;
    color: var(--tapedeck-accent-deep) !important;
    padding: 0.15rem 0.35rem !important;
    border-radius: 4px !important;
    font-size: 0.9em !important;
}

/* ─── Expanders (used for tool call log) ─── */
[data-testid="stExpander"] {
    background: var(--tapedeck-bg-card) !important;
    border: 1px solid var(--tapedeck-border-subtle) !important;
    border-radius: 8px !important;
    margin-bottom: 0.5rem !important;
}
[data-testid="stExpander"] summary { font-family: 'Inter', sans-serif !important; }

/* ─── Alerts / info boxes ─── */
[data-testid="stAlert"] {
    background: var(--tapedeck-bg-card) !important;
    border-radius: 8px !important;
    border-left: 3px solid var(--tapedeck-accent) !important;
    color: var(--tapedeck-ink) !important;
}

/* ─── Custom containers ─── */
.tapedeck-banner {
    background: linear-gradient(135deg, #FAF7F2 0%, #F0E8DA 100%);
    border: 1px solid var(--tapedeck-border);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 1.5rem;
    font-size: 0.9rem;
    color: var(--tapedeck-ink-soft);
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.tapedeck-banner strong { color: var(--tapedeck-ink); }
.tapedeck-banner a {
    color: var(--tapedeck-accent-deep);
    text-decoration: underline;
    text-underline-offset: 2px;
}

.tapedeck-footer {
    margin-top: 4rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--tapedeck-border-subtle);
    text-align: center;
    font-size: 0.85rem;
    color: var(--tapedeck-ink-faded);
    line-height: 1.7;
}
.tapedeck-footer a {
    color: var(--tapedeck-accent-deep);
    text-decoration: none;
    font-weight: 500;
}
.tapedeck-footer a:hover { text-decoration: underline; }

/* Tool call log rows */
.tapedeck-log-row {
    font-family: 'JetBrains Mono', 'SF Mono', monospace;
    font-size: 0.82rem;
    padding: 0.35rem 0.75rem;
    margin: 0.15rem 0;
    border-radius: 6px;
    background: var(--tapedeck-bg-deep);
    color: var(--tapedeck-ink-soft);
    border-left: 2px solid var(--tapedeck-ink-faded);
}
.tapedeck-log-row.tool-call  { border-left-color: var(--tapedeck-accent); }
.tapedeck-log-row.fallback   { border-left-color: var(--tapedeck-warn); }
.tapedeck-log-row.error      { border-left-color: var(--tapedeck-error); color: var(--tapedeck-error); }
.tapedeck-log-row.success    { border-left-color: var(--tapedeck-success); }
.tapedeck-log-row.model      { border-left-color: var(--tapedeck-ink-soft); font-weight: 500; }

/* Section dividers */
hr {
    border: none !important;
    border-top: 1px solid var(--tapedeck-border-subtle) !important;
    margin: 2rem 0 !important;
}

/* Hide Streamlit chrome we don't want */
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
</style>
"""

st.markdown(CLAUDE_THEME_CSS, unsafe_allow_html=True)


# =============================================================================
# SESSION STATE
# =============================================================================

def init_session():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "tool_log" not in st.session_state:
        st.session_state.tool_log = []  # per-turn list of log rows
    if "last_run_had_error" not in st.session_state:
        st.session_state.last_run_had_error = False
    if "last_model_used" not in st.session_state:
        st.session_state.last_model_used = None
    if "agent_tools_cache" not in st.session_state:
        st.session_state.agent_tools_cache = None
    if "tool_counts" not in st.session_state:
        st.session_state.tool_counts = None


init_session()


# =============================================================================
# AGENT WIRING
# =============================================================================

@st.cache_resource(show_spinner=False)
def bootstrap_agent():
    """Configure credentials and spawn the OpenBB MCP subprocess once per
    Streamlit session. Caches the tool list so we don't respawn on every
    user question.
    """
    configured_keys = tapedeck.configure_openbb_credentials()
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    agent_tools, total, active, wrapped = loop.run_until_complete(
        tapedeck.build_agent_tools()
    )
    return {
        "tools": agent_tools,
        "total": total,
        "active": active,
        "wrapped": wrapped,
        "configured_keys": configured_keys,
        "loop": loop,
    }


def render_log_row(level: str, text: str):
    """Append a log row to the current turn's tool log."""
    cls_map = {
        "tool_call": "tool-call",
        "fallback":  "fallback",
        "cache":     "success",
        "error":     "error",
        "model":     "model",
        "success":   "success",
    }
    cls = cls_map.get(level, "")
    st.session_state.tool_log.append({"level": cls, "text": text})


def run_research(prompt: str, variant: str):
    """Execute one research run. Streams events into the sidebar log.
    Returns (final_answer, model_used, had_error).
    """
    boot = bootstrap_agent()
    agent_tools = boot["tools"]
    loop = boot["loop"]

    final_answer_parts = []
    model_used = {"name": None}
    had_fatal = {"flag": False}

    # Wire the agent.py internal fallback logger into our Streamlit log
    tapedeck.set_stream_callback(
        lambda lvl, msg: render_log_row(lvl, msg.strip())
    )

    def on_event(evt):
        t = evt.get("type")
        if t == "model_status":
            status = evt.get("status")
            m = evt.get("model", "?")
            if status == "trying":
                render_log_row("model", f"→ trying {m}")
            elif status == "succeeded":
                render_log_row("success", f"✓ {m} produced the answer")
                model_used["name"] = m
            elif status == "failed":
                render_log_row("error",
                               f"✗ {m} failed — {evt.get('error','')}")
        elif t == "tool_call":
            args_preview = json.dumps(evt.get("args", {}),
                                      default=str)[:120]
            render_log_row("tool_call",
                           f"🔧 {evt['name']}({args_preview})")
        elif t == "tool_result":
            name = evt.get("name", "?")
            content = evt.get("content", "")
            if evt.get("is_error"):
                render_log_row("error", f"⚠ {name} → {content[:140]}")
            else:
                snippet = content.replace("\n", " ")[:100]
                render_log_row("cache", f"📨 {name} → {snippet}…")
        elif t == "assistant_message":
            final_answer_parts.append(evt.get("content", ""))
        elif t == "fatal_error":
            had_fatal["flag"] = True
            render_log_row("error", f"❌ {evt.get('message','')}")

    try:
        loop.run_until_complete(
            tapedeck.run_with_fallback(
                prompt, "openrouter", variant, agent_tools, on_event=on_event
            )
        )
    except Exception as e:
        had_fatal["flag"] = True
        render_log_row("error", f"❌ {type(e).__name__}: {str(e)[:180]}")

    final_answer = "\n\n".join(final_answer_parts).strip()
    if not final_answer and not had_fatal["flag"]:
        final_answer = ("_The agent finished without producing a final "
                        "message. Check the tool log on the left for what "
                        "happened, and try again or pick a different model._")
    return final_answer, model_used["name"], had_fatal["flag"]


# =============================================================================
# UI LAYOUT
# =============================================================================

# ─── Header ───
st.markdown("# TapeDeck 📈")
st.markdown(
    '<div class="tapedeck-tagline">'
    "Institutional-grade equity research, grounded in real market data. "
    "Ask anything — the agent reasons over 127 live financial tools across "
    "fundamentals, valuation, ownership, news, and macro."
    "</div>",
    unsafe_allow_html=True,
)

# ─── Public demo banner ───
st.markdown(
    '<div class="tapedeck-banner">'
    '<span>🔓</span>'
    '<span><strong>Public demo.</strong> Running on free-tier data APIs — '
    'some endpoints may be paywalled or rate-limited. The agent automatically '
    'falls back across providers. For unlimited use, '
    '<a href="https://github.com/realarmaansidhu/TapeDeck" target="_blank">'
    'clone the repo</a> and run it locally with your own keys.</span>'
    '</div>',
    unsafe_allow_html=True,
)


# ─── Sidebar: model picker + tool log ───
with st.sidebar:
    st.markdown("### ⚙ Settings")

    model_variant = st.selectbox(
        "LLM",
        options=list(tapedeck.OPENROUTER_MODELS.keys()),
        index=list(tapedeck.OPENROUTER_MODELS.keys()).index("ling"),
        help=("Default: Ling 2.6 1T (free, proven end-to-end). If it's rate-"
              "limited or fails, the agent automatically cascades through "
              "Nemotron → Qwen3 → MiniMax → GLM → GPT-OSS → Gemma → Qwen-Code. "
              "Pick a specific model here to start with that one instead."),
    )

    if st.button("🔄 Retry with different model", use_container_width=True,
                 disabled=not st.session_state.messages):
        # Reruns the most recent user prompt with the currently-selected variant
        last_user_msg = next((m for m in reversed(st.session_state.messages)
                              if m["role"] == "user"), None)
        if last_user_msg:
            st.session_state.tool_log = []
            st.session_state.pending_retry = {
                "prompt": last_user_msg["content"],
                "variant": model_variant,
            }
            st.rerun()

    st.markdown("---")
    st.markdown("### 📡 Tool Call Log")

    if st.session_state.tool_log:
        log_html = "".join(
            f'<div class="tapedeck-log-row {row["level"]}">{row["text"]}</div>'
            for row in st.session_state.tool_log
        )
        st.markdown(log_html, unsafe_allow_html=True)
    else:
        st.markdown(
            '<div style="color: var(--tapedeck-ink-faded); '
            'font-size: 0.85rem; font-style: italic;">'
            "Logs stream here when the agent runs — "
            "every tool call, every fallback, every model switch."
            "</div>",
            unsafe_allow_html=True,
        )


# ─── Preset prompt buttons ───
PRESET_PROMPTS = {
    "📊 Analyze a stock": (
        "Give me a full research brief on {TICKER}. "
        "Pull 6-8 quarters of fundamentals, forward estimates, insider trading, "
        "institutional ownership, recent news, and peer comparables. Produce a "
        "structured thesis with bull case, bear case, and a balanced take with "
        "base/bull/bear 12-month scenarios."
    ),
    "⚖ Compare two stocks": (
        "Compare {TICKER_A} vs {TICKER_B} head-to-head. Growth trajectory, "
        "margins, valuation multiples, capital returns, and near-term catalysts. "
        "Tell me which one has the better setup over the next 12 months and why."
    ),
    "🛡 Portfolio risk check": (
        "I hold a concentrated position in {TICKER} — roughly 30%+ of my equity "
        "portfolio. Assess the concentration risk: cyclicality, valuation at "
        "current levels, sector exposure, and near-term catalysts that could "
        "move the stock ±20%. Give me a trim-vs-hold framework I can actually "
        "execute against."
    ),
    "🌐 Macro dashboard": (
        "Pull the key macro indicators that matter for US equity investors "
        "right now: Fed funds rate, 10-year yield, CPI trend, unemployment, "
        "ISM manufacturing, consumer sentiment. Synthesize into a one-page "
        "read on where we are in the cycle and what it implies for risk assets."
    ),
}


with st.container():
    st.markdown("#### Quick starts")
    preset_cols = st.columns(len(PRESET_PROMPTS))
    selected_preset = None
    for i, (label, template) in enumerate(PRESET_PROMPTS.items()):
        with preset_cols[i]:
            if st.button(label, use_container_width=True,
                         key=f"preset_{i}"):
                selected_preset = {"label": label, "template": template}

# If a preset was clicked, show the input form for its variables
if selected_preset:
    st.session_state.preset_active = selected_preset

preset_active = st.session_state.get("preset_active")
if preset_active:
    with st.form("preset_form", clear_on_submit=True):
        st.markdown(f"**{preset_active['label']}**")
        template = preset_active["template"]
        # Parse placeholders like {TICKER}, {TICKER_A}, {TICKER_B}
        import re
        placeholders = re.findall(r"\{(\w+)\}", template)
        inputs = {}
        if placeholders:
            input_cols = st.columns(len(placeholders))
            for i, ph in enumerate(placeholders):
                with input_cols[i]:
                    inputs[ph] = st.text_input(
                        ph.replace("_", " ").title(),
                        placeholder="e.g. NVDA",
                        key=f"preset_input_{ph}",
                    ).strip().upper()
        submitted = st.form_submit_button("Run research",
                                          use_container_width=False,
                                          type="primary")
        if submitted:
            if all(inputs.values()) or not placeholders:
                filled_prompt = template.format(**inputs) if inputs else template
                st.session_state.messages.append(
                    {"role": "user", "content": filled_prompt}
                )
                st.session_state.pending_run = {
                    "prompt": filled_prompt,
                    "variant": model_variant,
                }
                st.session_state.preset_active = None
                st.session_state.tool_log = []
                st.rerun()
            else:
                st.warning("Please fill in all fields first.")


# ─── Bootstrap status indicator ───
try:
    boot = bootstrap_agent()
    if boot.get("configured_keys"):
        st.caption(
            f"✓ OpenBB connected · {boot['total']} tools discovered · "
            f"{boot['active']} active ({boot['wrapped']} fault-tolerant)"
        )
    else:
        st.warning(
            "⚠ No data-provider API keys configured. The agent will still "
            "work against free providers (yfinance), but FMP/FRED/Tiingo/"
            "Benzinga endpoints will return empty."
        )
except Exception as e:
    st.error(f"❌ Failed to initialize OpenBB MCP: {e}")
    st.stop()


# ─── Chat history ───
st.markdown("#### Chat")
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# ─── Chat input ───
user_input = st.chat_input(
    "Ask anything — e.g. 'Is NVDA a good entry near $200?'"
)

# If user typed a new question
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.tool_log = []
    st.session_state.pending_run = {
        "prompt": user_input,
        "variant": model_variant,
    }
    st.rerun()


# ─── Execute pending run ───
# We route all runs through a single "pending_run" slot so we can re-run after
# st.rerun() lands on a clean page state. The retry-button flow uses
# pending_retry which gets normalized to pending_run below.
pending_retry = st.session_state.pop("pending_retry", None)
if pending_retry:
    st.session_state.pending_run = pending_retry

pending = st.session_state.pop("pending_run", None)
if pending:
    # Render the user bubble (already in history) plus the live assistant bubble
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Researching… (see tool log in the sidebar)_")
        t_start = time.time()

        final, model_used, had_error = run_research(
            pending["prompt"], pending["variant"]
        )

        t_elapsed = time.time() - t_start
        if final:
            footer = ""
            if model_used:
                footer = (f"\n\n<sub style='color:var(--tapedeck-ink-faded)'>"
                          f"Answered by <code>{model_used}</code> in "
                          f"{t_elapsed:.1f}s.</sub>")
            placeholder.markdown(final + footer, unsafe_allow_html=True)
        else:
            placeholder.markdown(
                "_The agent couldn't produce an answer. Try the retry button "
                "in the sidebar with a different model._"
            )

        st.session_state.messages.append(
            {"role": "assistant", "content": final}
        )
        st.session_state.last_run_had_error = had_error
        st.session_state.last_model_used = model_used


# ─── Footer ───
st.markdown(
    '<div class="tapedeck-footer">'
    "Built by <a href='https://realarmaansidhu.com' target='_blank'>"
    "Armaan Sidhu</a> · "
    "<a href='https://github.com/realarmaansidhu/TapeDeck' target='_blank'>"
    "GitHub</a> · "
    "Powered by <a href='https://openbb.co' target='_blank'>OpenBB</a>, "
    "<a href='https://openrouter.ai' target='_blank'>OpenRouter</a>, "
    "and <a href='https://langchain.com' target='_blank'>LangChain</a>"
    "<br><br>"
    "<em>This is research and analysis, not investment advice. "
    "Verify data independently before any position change.</em>"
    "</div>",
    unsafe_allow_html=True,
)