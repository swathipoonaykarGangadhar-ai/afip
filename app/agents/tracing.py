"""
Langfuse tracing integration for the agent pipeline.

Every serious 2026 agent deployment wires in tracing from day one --
without it you can't answer "why did the agent decide X" after the
fact, which is a non-starter for anything touching compliance.

This wrapper is a true no-op (not a fake/mocked one) when Langfuse
isn't configured, so local dev works without any external service.
Set LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY / LANGFUSE_HOST to
activate real tracing.
"""
import os
import functools

_ENABLED = bool(os.environ.get("LANGFUSE_PUBLIC_KEY") and os.environ.get("LANGFUSE_SECRET_KEY"))

_client = None
if _ENABLED:
    try:
        from langfuse import Langfuse
        _client = Langfuse(
            public_key=os.environ["LANGFUSE_PUBLIC_KEY"],
            secret_key=os.environ["LANGFUSE_SECRET_KEY"],
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except Exception as e:
        import logging
        logging.getLogger(__name__).warning(f"Langfuse init failed, tracing disabled: {e}")
        _ENABLED = False


def traced_agent(name: str):
    """Decorator for agent node functions -- traces inputs/outputs when Langfuse is configured."""
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(state, *args, **kwargs):
            if not _ENABLED or _client is None:
                return fn(state, *args, **kwargs)

            with _client.start_as_current_span(name=name) as span:
                span.update(input={"transaction_id": state.get("transaction", {}).get("transaction_id")})
                result = fn(state, *args, **kwargs)
                span.update(output=result)
                return result
        return wrapper
    return decorator


def flush():
    if _ENABLED and _client is not None:
        _client.flush()


def is_enabled() -> bool:
    return _ENABLED
