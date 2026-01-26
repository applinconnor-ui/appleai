from __future__ import annotations

import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Literal, Optional, TypedDict

import streamlit as st
# Load local .env explicitly from the project directory (next to this file)
# This avoids Streamlit cwd surprises.
try:
    from dotenv import load_dotenv

    env_path = Path(__file__).resolve().with_name(".env")
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=True)
    else:
        # Fallback: try current working directory
        load_dotenv(override=True)
except Exception:
    pass

# If OPENAI_API_KEY is still missing, parse .env manually (works even without python-dotenv)
try:
    env_path = Path(__file__).resolve().with_name(".env")
    if env_path.exists() and not os.getenv("OPENAI_API_KEY"):
        for raw_line in env_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip().strip('"').strip("'")
            if k and v and k not in os.environ:
                os.environ[k] = v
except Exception:
    pass

# OpenAI client (optional)
try:
    from openai import OpenAI
except Exception:
    OpenAI = None  # type: ignore

# NOTE: set_page_config must be the FIRST Streamlit command.
st.set_page_config(page_title="AppleAI", page_icon="🍎")

# -----------------------------
# 1) Framework: Core Types
# -----------------------------
Role = Literal["user", "assistant", "system"]


class ChatMessage(TypedDict):
    role: Role
    content: str


@dataclass(frozen=True)
class BotConfig:
    """Framework config you can reuse across projects."""

    app_title: str = "🧠 Chatbot"
    app_caption: str = "Reusable Streamlit chatbot framework"

    # Session key for conversation storage
    session_messages_key: str = "messages"

    # Optional system prompt to seed the chat (useful for specialized bots)
    system_prompt: Optional[str] = None

    # UI
    user_input_placeholder: str = "Ask me anything…"

    # LLM (optional)
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.4
    max_history_messages: int = 20


# -----------------------------
# 2) Framework: Memory Layer
# -----------------------------
class MessageStore:
    """Thin wrapper around st.session_state to manage chat memory."""

    def __init__(self, key: str, system_prompt: Optional[str] = None):
        self.key = key
        if key not in st.session_state:
            st.session_state[key] = []  # type: ignore

        # Seed system prompt once (optional)
        if system_prompt and not self._has_system_prompt():
            self.append({"role": "system", "content": system_prompt})

    def _has_system_prompt(self) -> bool:
        msgs: List[ChatMessage] = st.session_state[self.key]
        return any(m["role"] == "system" for m in msgs)

    def get(self) -> List[ChatMessage]:
        return st.session_state[self.key]

    def append(self, msg: ChatMessage) -> None:
        st.session_state[self.key].append(msg)

    def clear(self) -> None:
        st.session_state[self.key] = []


# -----------------------------
# 3.5) Framework: LLM Provider (optional)
# -----------------------------
class LLMProvider:
    """Swap implementations (OpenAI, local model, etc.) without changing the app."""

    def generate(self, messages: List[ChatMessage], *, model: str, temperature: float) -> str:
        raise NotImplementedError


class OpenAIChatProvider(LLMProvider):
    def __init__(self, api_key: str):
        if OpenAI is None:
            raise RuntimeError("openai package not installed")
        self.client = OpenAI(api_key=api_key)

    def generate(self, messages: List[ChatMessage], *, model: str, temperature: float) -> str:
        oa_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        resp = self.client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=oa_messages,
        )
        return resp.choices[0].message.content or ""


def build_llm_provider() -> Optional[LLMProvider]:
    # Clear prior error
    st.session_state["_llm_error"] = ""

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        st.session_state["_llm_error"] = "OPENAI_API_KEY not found in environment"
        return None

    if OpenAI is None:
        st.session_state["_llm_error"] = "openai package not importable in this Python environment"
        return None

    try:
        return OpenAIChatProvider(api_key=api_key)
    except Exception as e:
        st.session_state["_llm_error"] = repr(e)
        return None


# -----------------------------
# 4) Framework: Handler Registry
# -----------------------------
HandlerFn = Callable[[str, List[ChatMessage]], str]


def llm_general_handler(user_text: str, history: List[ChatMessage]) -> str:
    """General Q&A powered by an LLM when configured, otherwise falls back."""

    provider: Optional[LLMProvider] = st.session_state.get("_llm_provider")
    config: Optional[BotConfig] = st.session_state.get("_bot_config")

    if provider is None or config is None:
        return (
            "(GENERAL MODE) LLM is not configured. "
            "Make sure your `.env` contains `OPENAI_API_KEY=...` and restart Streamlit.\n\n"
            f"You said: **{user_text}**"
        )

    # Build a bounded prompt: system (optional) + last N turns + current user
    system_msgs = [m for m in history if m["role"] == "system"]
    convo_msgs = [m for m in history if m["role"] != "system"]

    recent = convo_msgs[-config.max_history_messages :] if config.max_history_messages > 0 else convo_msgs

    messages: List[ChatMessage] = []
    if system_msgs:
        messages.append(system_msgs[0])
    messages.extend(recent)
    messages.append({"role": "user", "content": user_text})

    try:
        answer = provider.generate(
            messages,
            model=config.llm_model,
            temperature=config.llm_temperature,
        ).strip()
        return answer or "(GENERAL MODE) I didn’t generate a response."
    except Exception as e:
        return f"(GENERAL MODE) LLM call failed: `{e}`"


# -----------------------------
# 5) Framework: App Orchestrator
# -----------------------------
class ChatbotApp:
    def __init__(
        self,
        config: BotConfig,
        handler: HandlerFn,
    ):
        self.config = config
        self.handler = handler
        self.store = MessageStore(
            key=self.config.session_messages_key,
            system_prompt=self.config.system_prompt,
        )

        # Expose config/provider to handlers without global variables
        st.session_state["_bot_config"] = self.config
        # (Re)build provider at startup so .env changes take effect after a restart
        st.session_state["_llm_provider"] = build_llm_provider()

    def render_header(self) -> None:
        st.title(self.config.app_title)
        st.caption(self.config.app_caption)

    def render_sidebar(self) -> None:
        with st.sidebar:
            st.subheader("Controls")
            if st.button("Reset chat"):
                self.store.clear()
                st.rerun()

            st.divider()
            st.subheader("Debug")
            st.caption(
                "LLM: "
                + ("✅ configured" if st.session_state.get("_llm_provider") else "⚠️ missing OPENAI_API_KEY")
            )
            try:
                env_path = Path(__file__).resolve().with_name(".env")
                st.caption(f".env: {'✅ found' if env_path.exists() else '❌ not found'} • {env_path}")
            except Exception:
                pass

            # Debug: confirm what the process can see
            key_present = bool(os.getenv("OPENAI_API_KEY", "").strip())
            st.caption(f"OPENAI_API_KEY in process env: {'✅ yes' if key_present else '❌ no'}")
            st.caption(f"Python: {sys.executable}")
            st.caption(f"OpenAI import: {'✅ ok' if OpenAI is not None else '❌ failed'}")
            err = (st.session_state.get("_llm_error") or "").strip()
            if err:
                st.caption(f"LLM init error: {err}")

    def render_history(self) -> None:
        for msg in self.store.get():
            # Don't render system messages in the UI
            if msg["role"] == "system":
                continue
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    def run_once(self) -> None:
        self.render_header()
        self.render_sidebar()
        self.render_history()

        user_text = st.chat_input(self.config.user_input_placeholder)
        if not user_text:
            st.divider()
            st.caption(f"RUNNING ✅ {time.time()}")
            return

        # 1) Persist + show user message
        self.store.append({"role": "user", "content": user_text})
        with st.chat_message("user"):
            st.markdown(user_text)

        # 2) Generate response via handler
        assistant_text = self.handler(user_text, self.store.get())

        # 3) Persist + show assistant response
        self.store.append({"role": "assistant", "content": assistant_text})
        with st.chat_message("assistant"):
            st.markdown(assistant_text)

        st.divider()
        st.caption(f"RUNNING ✅ {time.time()}")


# -----------------------------
# 6) Instantiate a bot (this is the part you swap per project)
# -----------------------------
newsgenie_config = BotConfig(
    app_title="🍎 AppleAI",
    app_caption="created by Connor Applin — 01/25/26",
    system_prompt=(
        "You are AppleAI, created by Connor Applin. "
        "You are a general-purpose AI assistant. "
        "Always identify yourself as 'AppleAI, created by Connor Applin' if asked about your identity. "
        "Be concise, helpful, and accurate."
    ),
    user_input_placeholder="Ask a question or request the latest news…",
)

app = ChatbotApp(
    config=newsgenie_config,
    handler=llm_general_handler,
)

app.run_once()
