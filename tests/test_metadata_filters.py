from src.bot import _meta_eq


def test_meta_eq_builds_haystack_filter() -> None:
    f = _meta_eq("chat_id", -123)
    assert f == {"field": "meta.chat_id", "operator": "==", "value": -123}

