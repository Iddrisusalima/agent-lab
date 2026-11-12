"""Provider adapters and the factory that selects one (Req 2.8, Req 2.9).

One adapter ships with the project, ``openai_client.OpenAIClient``. A second
provider is a new module in this package plus one entry in :data:`BUILDERS` --
no edit to the agent loop, which only ever sees the ``ModelClient`` protocol.

:func:`build_model_client` is the only way the application constructs a client.
It is here rather than in :mod:`agent_lab.model_client` for the reason that
module states about itself: the protocol and its value types import no provider
SDK, and the factory necessarily knows the adapters. Keeping it in this package
means ``model_client.py`` stays importable, and testable, with no SDK present.

Each entry in :data:`BUILDERS` imports its adapter inside the builder rather
than at module import time, so importing ``agent_lab.providers`` -- which the
CLI does to reach this factory -- does not pull in the SDK of a provider the
operator did not select.
"""

from __future__ import annotations

from typing import Any, Callable

from agent_lab.config import Config, ConfigError, SUPPORTED_PROVIDERS, api_key_var_for
from agent_lab.model_client import ModelClient


def _build_openai(config: Config, **overrides: Any) -> ModelClient:
    """Construct the OpenAI-compatible adapter (Req 2.8)."""
    from agent_lab.providers.openai_client import OpenAIClient

    return OpenAIClient(config.model_id, api_key=config.api_key, **overrides)


#: Provider identifier to builder, keyed by the ``PROVIDER`` value exactly as
#: :data:`agent_lab.config.SUPPORTED_PROVIDERS` lists it. The two are checked
#: against each other below, so adding a provider to one and forgetting the
#: other fails at import rather than at the first run.
BUILDERS: dict[str, Callable[..., ModelClient]] = {
    "openai": _build_openai,
}

assert set(BUILDERS) == set(SUPPORTED_PROVIDERS), (
    "config.SUPPORTED_PROVIDERS and providers.BUILDERS must list the same "
    f"identifiers; got {sorted(SUPPORTED_PROVIDERS)} and {sorted(BUILDERS)}"
)


def build_model_client(config: Config, **overrides: Any) -> ModelClient:
    """Return the model client for the configured provider (Req 2.8, Req 2.9).

    Keyed by ``config.provider``, compared case-sensitively against
    :data:`BUILDERS`. An identifier with no adapter and an absent or empty
    credential both raise :class:`agent_lab.config.ConfigError`, which the CLI
    turns into its configuration exit code -- so the process stops before step 1
    and before any provider request is issued (Req 2.9, Req 1.8).

    ``load_config`` already refuses both conditions, which makes the checks here
    redundant on the CLI path and worth keeping anyway: this function is also
    the entry point for the eval runner and for any future caller that builds a
    ``Config`` directly, and "the credential was never checked" is not a failure
    mode worth leaving open to save four lines.

    ``overrides`` is forwarded to the adapter and exists for the seams the
    adapter documents -- an injected SDK client and an injected ``sleep`` -- so
    a test can exercise the retry schedule through the real factory without a
    network or an API key.

    One client per run: the adapter holds the run's issued tool-call identifiers
    and its step counter (Req 2.4, Req 2.10, Req 2.11), so a client must not be
    shared between runs. Each eval task builds its own.
    """
    provider = getattr(config, "provider", None)
    builder = BUILDERS.get(provider) if isinstance(provider, str) else None
    if builder is None:
        raise ConfigError(
            f"PROVIDER is set to {provider!r}, for which no provider adapter is "
            f"available. Supported identifiers are "
            f"{', '.join(repr(name) for name in sorted(BUILDERS))}. "
            f"No provider request was made."
        )

    credential = getattr(config, "api_key", "") or ""
    if not credential.strip():
        variable = getattr(config, "api_key_var", None) or api_key_var_for(provider)
        raise ConfigError(
            f"{variable} is not set (or is empty), so the {provider} provider "
            f"cannot be called. Set it in .env or in your shell. No provider "
            f"request was made."
        )

    model_id = getattr(config, "model_id", "") or ""
    if not model_id.strip():
        raise ConfigError(
            "MODEL_ID is not set (or is empty); the provider needs a model "
            "identifier. See .env.example for the default. No provider request "
            "was made."
        )

    return builder(config, **overrides)


def available_providers() -> tuple[str, ...]:
    """The provider identifiers with a shipped adapter, sorted."""
    return tuple(sorted(BUILDERS))


__all__ = [
    "BUILDERS",
    "available_providers",
    "build_model_client",
]
