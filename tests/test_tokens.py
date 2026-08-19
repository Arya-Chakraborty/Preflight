from preflight import tokens


def test_count_text():
    assert tokens.count_text("hello world") > 0
    assert tokens.count_text("") == 0


def test_count_messages_includes_overhead():
    msgs = [{"role": "user", "content": "hi"}]
    assert tokens.count_messages(msgs) > tokens.count_text("hi")


def test_unknown_model_falls_back():
    assert tokens.count_text("hello", model="totally-unknown-model-xyz") > 0


def test_message_text_multimodal():
    msg = {"role": "user", "content": [{"type": "text", "text": "part one"}, {"type": "image_url"}]}
    assert tokens.message_text(msg) == "part one"
