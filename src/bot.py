from __future__ import annotations

import logging
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any

import telebot
from haystack import Document

from src.config import Settings
from src.rag import RagServices, index_documents, participant_reply, session_summary
from src.state import StateStore


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat()


def _author_meta(message: telebot.types.Message) -> dict[str, Any]:
    u = message.from_user
    if u is None:
        return {}
    return {
        "author_id": u.id,
        "author_username": u.username,
        "author_first_name": u.first_name,
        "author_last_name": u.last_name,
        "author_is_bot": u.is_bot,
    }


def _message_record(message: telebot.types.Message) -> dict[str, Any]:
    dt = datetime.fromtimestamp(message.date, tz=timezone.utc)
    rec: dict[str, Any] = {
        "chat_id": message.chat.id,
        "chat_type": message.chat.type,
        "message_id": message.message_id,
        "date": _iso(dt),
        "text": message.text or "",
        "reply_to_message_id": getattr(getattr(message, "reply_to_message", None), "message_id", None),
    }
    rec.update(_author_meta(message))
    return rec


def _doc_from_message(message: telebot.types.Message) -> Document:
    rec = _message_record(message)
    content = rec["text"]
    # Stable id for overwrite capability
    doc_id = f"tg:{rec['chat_id']}:{rec['message_id']}"
    return Document(id=doc_id, content=content, meta=rec)


def _render_dialogue(records: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for r in records:
        who = r.get("author_username") or (r.get("author_first_name") or "Unknown")
        when = r.get("date") or ""
        text = (r.get("text") or "").strip()
        if not text:
            continue
        lines.append(f"[{when}] {who}: {text}")
    return "\n".join(lines).strip()

def _meta_eq(field: str, value: Any) -> dict[str, Any]:
    # Haystack metadata filtering syntax (Comparison)
    return {"field": f"meta.{field}", "operator": "==", "value": value}


def _is_mentioning_bot(message: telebot.types.Message, bot_username: str) -> bool:
    text = message.text or ""
    if not text:
        return False
    tag = f"@{bot_username}".lower()
    if tag in text.lower():
        return True
    # Entities handling (best-effort)
    ents = getattr(message, "entities", None) or []
    for e in ents:
        if getattr(e, "type", None) == "mention":
            off = getattr(e, "offset", 0)
            ln = getattr(e, "length", 0)
            if text[off : off + ln].lower() == tag:
                return True
    return False


class TeamAssistantBot:
    def __init__(self, settings: Settings, services: RagServices, state: StateStore):
        self.settings = settings
        self.services = services
        self.state = state
        self.bot = telebot.TeleBot(settings.telegram_bot_token, parse_mode=None)

        self._last_messages: dict[int, deque[dict[str, Any]]] = defaultdict(
            lambda: deque(maxlen=settings.last_messages_context)
        )
        self._bot_username: str | None = None

        self._register_handlers()

    def _register_handlers(self) -> None:
        @self.bot.message_handler(commands=["start_listening"])
        def _start_listening(m: telebot.types.Message) -> None:
            logging.info("cmd /start_listening chat_id=%s", m.chat.id)
            session_id = self.state.start_listening(m.chat.id)
            self.bot.reply_to(
                m,
                f"Ок. Начал слушать чат и сохранять контекст. session_id={session_id}",
            )

        @self.bot.message_handler(commands=["stop_listening"])
        def _stop_listening(m: telebot.types.Message) -> None:
            logging.info("cmd /stop_listening chat_id=%s", m.chat.id)
            session_id = self.state.stop_listening(m.chat.id)
            if not session_id:
                self.bot.reply_to(m, "Сессия не найдена. Сначала вызови /start_listening.")
                return

            records = self.state.read_session_messages(m.chat.id, session_id)
            dialogue = _render_dialogue(records)
            if not dialogue:
                self.bot.reply_to(m, "В сессии нет сообщений для саммари.")
                return

            # Summarize (with retrieval, filtered by chat_id + session_id)
            filters = {
                "operator": "AND",
                "conditions": [
                    _meta_eq("chat_id", m.chat.id),
                    _meta_eq("session_id", session_id),
                ],
            }
            logging.info("summarizing session chat_id=%s session_id=%s", m.chat.id, session_id)
            try:
                summary = session_summary(
                    self.services,
                    question="Подведи итоги обсуждения за сессию.",
                    dialogue=dialogue,
                    filters=filters,
                    top_k=self.settings.top_k,
                )
            except Exception:
                logging.exception("failed to summarize session")
                self.bot.reply_to(m, "Ошибка при построении саммари (см. логи).")
                return

            # Store summary as a doc too
            summary_doc = Document(
                id=f"tg:{m.chat.id}:session:{session_id}:summary",
                content=summary,
                meta={
                    "chat_id": m.chat.id,
                    "session_id": session_id,
                    "type": "session_summary",
                    "date": _iso(datetime.now(timezone.utc)),
                },
            )
            index_documents(self.services, [summary_doc])

            self.bot.reply_to(m, summary)

        @self.bot.message_handler(content_types=["text"])
        def _on_text(m: telebot.types.Message) -> None:
            if m.text is None:
                return

            # Remember last messages (for mention context)
            rec = _message_record(m)
            self._last_messages[m.chat.id].append(rec)
            logging.info(
                "text chat_id=%s msg_id=%s from=%s username=%s text=%r",
                m.chat.id,
                m.message_id,
                rec.get("author_id"),
                rec.get("author_username"),
                (m.text or "")[:200],
            )

            chat_state = self.state.get_chat(m.chat.id)
            # Index EVERY message (global chat context).
            # If listening session is active, also attach session_id and persist JSONL.
            doc = _doc_from_message(m)
            if chat_state.listening and chat_state.session_id:
                self.state.append_message(m.chat.id, chat_state.session_id, rec)
                doc.meta["session_id"] = chat_state.session_id
                logging.info(
                    "indexing chat_id=%s session_id=%s doc_id=%s",
                    m.chat.id,
                    chat_state.session_id,
                    doc.id,
                )
            else:
                logging.info("indexing chat_id=%s session_id=None doc_id=%s", m.chat.id, doc.id)

            try:
                index_documents(self.services, [doc])
            except Exception:
                logging.exception("failed to index message (continuing)")

            # Mention behavior (acts as participant)
            if self._bot_username and _is_mentioning_bot(m, self._bot_username):
                question = (m.text or "").strip()
                last_context = list(self._last_messages[m.chat.id])
                dialogue = _render_dialogue(last_context)
                logging.info("mention detected chat_id=%s bot=@%s", m.chat.id, self._bot_username)
                try:
                    # If session is active, prioritize session context; otherwise, use whole chat.
                    if chat_state.listening and chat_state.session_id:
                        filters = {
                            "operator": "AND",
                            "conditions": [
                                _meta_eq("chat_id", m.chat.id),
                                _meta_eq("session_id", chat_state.session_id),
                            ],
                        }
                    else:
                        filters = _meta_eq("chat_id", m.chat.id)

                    answer = participant_reply(
                        self.services,
                        question=question,
                        dialogue=dialogue,
                        filters=filters,
                        top_k=self.settings.top_k,
                    )
                    self.bot.reply_to(m, answer)
                except Exception:
                    logging.exception("failed to answer mention")
                    self.bot.reply_to(m, "Ошибка при обработке запроса (см. логи).")

    def run_polling(self) -> None:
        me = self.bot.get_me()
        self._bot_username = getattr(me, "username", None)
        if not self._bot_username:
            raise RuntimeError("Bot username is missing (set a username for the bot in BotFather).")
        logging.info("bot started username=@%s", self._bot_username)
        # If a webhook is set, long polling via getUpdates will receive nothing.
        try:
            self.bot.remove_webhook()
            logging.info("webhook removed (polling mode)")
        except Exception:
            logging.exception("failed to remove webhook")

        self.bot.infinity_polling(
            skip_pending=True,
            allowed_updates=["message"],
        )

