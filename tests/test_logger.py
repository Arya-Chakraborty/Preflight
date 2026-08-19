from preflight.outcomes.logger import Outcome, OutcomeLogger


def _outcome(**kw):
    base = dict(
        session_id="s1",
        model="gpt-4o-mini",
        provider="openai",
        action="A5",
        tokens_out=10,
        cost_realized=0.001,
        cost_baseline=0.002,
        latency_ms=12.0,
    )
    base.update(kw)
    return Outcome(**base)


def test_log_and_summary(tmp_path):
    logger = OutcomeLogger(tmp_path)
    logger.log(_outcome())
    logger.log(_outcome(action="A1", cost_realized=0.0))
    s = logger.summary()
    assert s["requests"] == 2
    assert abs(s["realized_usd"] - 0.001) < 1e-9
    assert s["baseline_usd"] > s["realized_usd"]
    assert set(s["by_action"]) == {"A5", "A1"}


def test_retry_flag(tmp_path):
    logger = OutcomeLogger(tmp_path)
    rid = logger.log(_outcome())
    logger.flag_retry(rid)
    row = logger.rows()[0]
    assert row["retry_flag"] == 1


def test_session_rows(tmp_path):
    logger = OutcomeLogger(tmp_path)
    logger.log(_outcome(session_id="a"))
    logger.log(_outcome(session_id="b"))
    assert len(logger.last_in_session("a")) == 1
