from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class ChatSessionState:
    listening: bool
    session_id: str | None


class StateStore:
    def __init__(self, root_dir: str | os.PathLike = "data"):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.sessions_dir = self.root / "sessions"
        self.sessions_dir.mkdir(parents=True, exist_ok=True)

        if not self.state_path.exists():
            self._write_state({"chats": {}})

    def _read_state(self) -> dict[str, Any]:
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _write_state(self, data: dict[str, Any]) -> None:
        self.state_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def get_chat(self, chat_id: int) -> ChatSessionState:
        data = self._read_state()
        chats = data.get("chats", {})
        raw = chats.get(str(chat_id), {})
        return ChatSessionState(
            listening=bool(raw.get("listening", False)),
            session_id=raw.get("session_id"),
        )

    def start_listening(self, chat_id: int) -> str:
        session_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        data = self._read_state()
        data.setdefault("chats", {})
        data["chats"][str(chat_id)] = {"listening": True, "session_id": session_id}
        self._write_state(data)

        session_path = self._session_path(chat_id, session_id)
        session_path.parent.mkdir(parents=True, exist_ok=True)
        session_path.touch(exist_ok=True)
        return session_id

    def stop_listening(self, chat_id: int) -> str | None:
        data = self._read_state()
        chat = data.get("chats", {}).get(str(chat_id))
        if not chat:
            return None
        session_id = chat.get("session_id")
        data["chats"][str(chat_id)] = {"listening": False, "session_id": session_id}
        self._write_state(data)
        return session_id

    def append_message(self, chat_id: int, session_id: str, record: dict[str, Any]) -> None:
        path = self._session_path(chat_id, session_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read_session_messages(self, chat_id: int, session_id: str) -> list[dict[str, Any]]:
        path = self._session_path(chat_id, session_id)
        if not path.exists():
            return []
        out: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            out.append(json.loads(line))
        return out

    def _session_path(self, chat_id: int, session_id: str) -> Path:
        return self.sessions_dir / str(chat_id) / f"{session_id}.jsonl"

