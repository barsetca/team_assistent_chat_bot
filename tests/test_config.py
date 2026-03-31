import os

import pytest

from src.config import load_settings


def test_load_settings_requires_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PINECONE_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    # Ensure .env from workspace doesn't re-populate values
    monkeypatch.setenv("DOTENV_DISABLE", "1")

    with pytest.raises(RuntimeError):
        load_settings()


@pytest.mark.parametrize(
    "index_name,ok",
    [
        ("team-chat", True),
        ("teamchat123", True),
        ("team_chat", False),
        ("Team-Chat", False),
        ("", False),
    ],
)
def test_pinecone_index_name_validation(monkeypatch: pytest.MonkeyPatch, index_name: str, ok: bool) -> None:
    monkeypatch.setenv("PINECONE_API_KEY", "x")
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "x")
    monkeypatch.setenv("PINECONE_INDEX_NAME", index_name)

    if ok:
        s = load_settings()
        assert s.pinecone_index_name == index_name
    else:
        with pytest.raises(RuntimeError):
            load_settings()

