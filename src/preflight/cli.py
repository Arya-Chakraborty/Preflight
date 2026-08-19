"""Preflight command-line interface."""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(name="preflight", help="Local cost-optimizing gateway for frontier LLM APIs.")

_config_opt = typer.Option(None, "--config", "-c", help="Path to preflight.yaml")


@app.command()
def serve(
    config: Path | None = _config_opt,
    host: str | None = typer.Option(None, help="Override bind host"),
    port: int | None = typer.Option(None, help="Override bind port"),
):
    """Run the local OpenAI-compatible proxy."""
    import uvicorn

    from preflight.config import load_settings
    from preflight.proxy.server import create_app

    settings = load_settings(config)
    if host:
        settings.host = host
    if port:
        settings.port = port
    typer.echo(f"Preflight listening on http://{settings.host}:{settings.port}/v1")
    uvicorn.run(create_app(settings), host=settings.host, port=settings.port, log_level="info")


@app.command()
def stats(config: Path | None = _config_opt):
    """Show spend and action statistics from the outcome log."""
    from preflight.config import load_settings
    from preflight.outcomes.logger import OutcomeLogger

    settings = load_settings(config)
    logger = OutcomeLogger(settings.data_dir)
    summary = logger.summary()
    if summary["requests"] == 0:
        typer.echo("No requests logged yet.")
        raise typer.Exit()

    typer.echo(f"Requests:        {summary['requests']}")
    typer.echo(f"Realized spend:  ${summary['realized_usd']:.4f}")
    typer.echo(f"Baseline (raw):  ${summary['baseline_usd']:.4f}")
    typer.echo(f"Savings:         ${summary['baseline_usd'] - summary['realized_usd']:.4f}")
    typer.echo(f"Mean latency:    {summary['mean_latency_ms']:.0f} ms")
    typer.echo("Actions:")
    for action, row in sorted(summary["by_action"].items()):
        typer.echo(
            f"  {action}: n={row['n']}  spend=${row['usd']:.4f}  mean_out_tokens={row['mean_out']:.0f}"
        )


@app.command()
def refit(config: Path | None = _config_opt):
    """Retrain the output-length and failure estimators from the outcome log."""
    from preflight.config import load_settings
    from preflight.costs.estimators import refit_from_log
    from preflight.outcomes.logger import OutcomeLogger

    settings = load_settings(config)
    logger = OutcomeLogger(settings.data_dir)
    report = refit_from_log(logger, settings)
    typer.echo(
        f"Refit complete: {report['rows']} rows, "
        f"output-len MAE {report['outlen_mae']:.1f} tokens, "
        f"failure base rates {report['pfail']}"
    )


@app.command()
def replay(
    config: Path | None = _config_opt,
    limit: int = typer.Option(500, help="Max logged requests to replay"),
):
    """Re-simulate logged traffic under the current policy (offline, no API calls)."""
    from preflight.config import load_settings
    from preflight.replay import replay_log

    settings = load_settings(config)
    report = replay_log(settings, limit=limit)
    if report["rows"] == 0:
        typer.echo("Nothing to replay.")
        raise typer.Exit()
    typer.echo(f"Replayed {report['rows']} requests.")
    typer.echo(f"Historical realized spend: ${report['realized_usd']:.4f}")
    typer.echo(f"Current-policy estimate:   ${report['policy_usd']:.4f}")
    typer.echo("Action shift (old -> new):")
    for pair, n in sorted(report["shift"].items()):
        typer.echo(f"  {pair}: {n}")


@app.command()
def ground(
    path: Path = typer.Argument(..., help="Text/markdown file or directory to index"),
    config: Path | None = _config_opt,
):
    """Add documents to the local grounding store (used by action A4)."""
    from preflight.assembler.grounding import GroundingStore
    from preflight.config import load_settings

    settings = load_settings(config)
    store = GroundingStore(settings)
    n = store.add_path(path)
    typer.echo(f"Indexed {n} chunks from {path}")


@app.command()
def version():
    """Print the package version."""
    from preflight import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
