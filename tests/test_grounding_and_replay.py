from preflight.assembler.grounding import GroundingStore
from preflight.gateway import Gateway
from preflight.replay import replay_log
from tests.conftest import make_payload


def test_grounding_store_roundtrip(settings, tmp_path):
    store = GroundingStore(settings)
    doc = tmp_path / "facts.md"
    doc.write_text("The Eiffel Tower is 330 metres tall and located in Paris, France.")
    n = store.add_path(doc)
    assert n == 1
    hits = store.query("How tall is the Eiffel Tower in Paris?")
    assert hits and "330 metres" in hits[0].text


def test_grounding_directory_and_floor(settings, tmp_path):
    store = GroundingStore(settings)
    d = tmp_path / "docs"
    d.mkdir()
    (d / "a.txt").write_text("Photosynthesis converts sunlight into chemical energy in plants.")
    (d / "b.bin").write_bytes(b"\x00\x01")  # ignored extension
    assert store.add_path(d) == 1
    assert store.query("completely unrelated cryptocurrency question", floor=0.9) == []


async def test_grounded_request_uses_a4_when_cheapest(settings, provider_calls):
    gateway = Gateway(settings)
    try:
        gateway.grounding.add_text(
            "Preflight is a local gateway that optimizes LLM inference costs."
        )
        resp = await gateway.handle(make_payload("What is Preflight, the local gateway?"))
        assert "choices" in resp
        row = gateway.logger.rows()[0]
        assert row["action"] in ("A4", "A5")  # A4 feasible; scorer picks the cheaper
    finally:
        gateway.close()


async def test_replay_after_traffic(settings, provider_calls):
    gateway = Gateway(settings)
    try:
        for i in range(4):
            await gateway.handle(make_payload(f"Question {i} on subject {i}?"))
        report = replay_log(settings)
        assert report["rows"] == 4
        assert report["realized_usd"] > 0
        assert sum(report["shift"].values()) == 4
    finally:
        gateway.close()
