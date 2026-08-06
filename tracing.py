"""Minimal, stdlib-only span exporter for avartan, JSONL-shaped for HALO's engine.

Matches the wire format HALO's own openai-agents-sdk demo tracing.py produces
(see HALO's docs/integrations/openai-agents-sdk.md and demo/openai-agents-sdk-demo/
sample-traces/traces.jsonl) — same top-level keys (trace_id, span_id,
parent_span_id, trace_state, name, kind, start_time, end_time, status,
resource, scope, attributes), same inference.* projection, same
STATUS_CODE_*/SPAN_KIND_* vocabulary and nanosecond-precision timestamps.

Diverges from HALO's version in one way: avartan doesn't use the
openai-agents SDK, so there's no TracingProcessor to intercept — spans are
opened and closed by hand around run_turn's own loop instead of derived from
SDK-emitted Span/Trace objects. Vendored and stdlib-only, same as HALO's.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from datetime import datetime, timezone

EXPORT_SCHEMA_VERSION = 1
DEFAULT_OUTPUT_PATH = "traces.jsonl"


def _now_otlp() -> str:
    dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond:06d}000Z"


def to_json(v):
    if v is None:
        return None
    try:
        return json.dumps(v, default=str, separators=(",", ":"))
    except (TypeError, ValueError):
        return json.dumps(str(v))


def _drop_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}


class Tracer:
    """Opens a JSONL file at `path` (append mode) and writes one line per span."""

    def __init__(self, path: str, *, project_id: str = "avartan", service_name: str = "avartan"):
        self.project_id = project_id
        self.service_name = service_name
        self._lock = threading.Lock()
        self._fh = open(path, "a", encoding="utf-8")

    def new_id(self, length: int = 32) -> str:
        return uuid.uuid4().hex[:length]

    def span(self, name: str, kind: str, trace_id: str, parent_span_id: str = "") -> "Span":
        return Span(self, name, kind, trace_id, parent_span_id)

    def write(self, line: dict) -> None:
        encoded = json.dumps(line, separators=(",", ":"), ensure_ascii=False)
        with self._lock:
            self._fh.write(encoded)
            self._fh.write("\n")

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.flush()
                self._fh.close()
            except Exception:
                pass


class Span:
    """One span. Set `span.attributes[...]` inside the `with` block, then it's
    written to the tracer on exit — OK status normally, ERROR status (with the
    exception message) if the block raised."""

    def __init__(self, tracer: Tracer, name: str, kind: str, trace_id: str, parent_span_id: str):
        self.tracer = tracer
        self.name = name
        self.kind = kind
        self.trace_id = trace_id
        self.parent_span_id = parent_span_id
        self.span_id = tracer.new_id(24)
        self.attributes: dict = {}
        self.error: str | None = None
        self._start = None

    def __enter__(self) -> "Span":
        self._start = _now_otlp()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        end = _now_otlp()

        if exc is not None:
            status = {"code": "STATUS_CODE_ERROR", "message": str(exc)}
        elif self.error:
            status = {"code": "STATUS_CODE_ERROR", "message": self.error}
        else:
            status = {"code": "STATUS_CODE_OK", "message": ""}

        attrs = _drop_none(self.attributes)
        attrs["openinference.span.kind"] = self.kind
        attrs.update(
            {
                "inference.export.schema_version": EXPORT_SCHEMA_VERSION,
                "inference.project_id": self.tracer.project_id,
                "inference.observation_kind": self.kind,
            }
        )

        self.tracer.write(
            {
                "trace_id": self.trace_id,
                "span_id": self.span_id,
                "parent_span_id": self.parent_span_id,
                "trace_state": "",
                "name": self.name,
                "kind": "SPAN_KIND_CLIENT" if self.kind == "LLM" else "SPAN_KIND_INTERNAL",
                "start_time": self._start,
                "end_time": end,
                "status": status,
                "resource": {"attributes": {"service.name": self.tracer.service_name}},
                "scope": {"name": "avartan", "version": "1"},
                "attributes": attrs,
            }
        )
        return False


def tracer_from_env(*, project_id: str = "avartan", service_name: str = "avartan") -> Tracer:
    """Construct a Tracer writing to $HALO_TRACES_PATH (default ./traces.jsonl)."""
    path = os.environ.get("HALO_TRACES_PATH", DEFAULT_OUTPUT_PATH)
    return Tracer(path, project_id=project_id, service_name=service_name)


class NullSpan:
    """No-op stand-in for Span so tracing-aware code doesn't need an `if tracer` branch."""

    def __init__(self):
        self.attributes: dict = {}
        self.error: str | None = None
        self.span_id = ""

    def __enter__(self) -> "NullSpan":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False
