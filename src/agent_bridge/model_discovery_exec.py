"""Bounded model discovery for one-shot exec workers.

Antigravity (``agy models``) and Codex (``codex app-server model/list``) can
answer "which models can this worker run" without dispatching a turn. No prompt
is ever sent and Bridge does not change worker configuration; the listing
reflects whatever the resolved, configured worker reports — including custom
endpoints configured through the worker's own config overrides.

Worker output is untrusted: diagnostics that cross into MCP responses or logs
pass through ``redact_diagnostic`` first, and stdout/stderr are bounded so a
chatty or stuck worker cannot stall or exhaust the coordinator. The registry
owns the outer timeout and cache; the bounds here keep one run self-contained.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from agent_bridge.diagnostics import redact_diagnostic
from agent_bridge.processes import reap_subprocess

log = logging.getLogger(__name__)

# The registry owns the configurable deadline and caching. Helpers may also
# receive an explicit timeout when used independently in focused tests.
# asyncio's default 64 KiB stream limit is too small for worker banners; one
# JSON event per line still must fit under this.
DISCOVERY_STDIO_LIMIT = 1024 * 1024
# Kept output caps: enough for any real model catalog, small enough to ignore
# a worker that streams garbage forever.
DISCOVERY_OUTPUT_BYTE_LIMIT = 4 * 1024 * 1024
DISCOVERY_LINE_LIMIT = 20000
DISCOVERY_STDERR_TAIL_LIMIT = 16000
# model/list pages and a hard page cap so a broken cursor loop cannot run away.
MODEL_LIST_PAGE_LIMIT = 100
MODEL_LIST_MAX_PAGES = 100

CLIENT_INFO = {"name": "agent-bridge", "title": "Agent Bridge", "version": "1"}

_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_AGY_SLUG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:+/-]*$")
_AGY_BULLET_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_AGY_DEFAULT_SUFFIX_RE = re.compile(r"\s*\(default\)\s*$", re.IGNORECASE)
_AGY_BORDER_CHARS = set("-\u2013\u2014=*_#+|")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


def _efforts_of(raw: Mapping[str, Any]) -> list[str] | None:
    """Worker-native offered effort values, verbatim, never Bridge-mapped."""
    for key in ("supported_efforts", "supportedEfforts", "supportedReasoningEfforts"):
        if key not in raw:
            continue
        items = raw[key]
        if not isinstance(items, list):
            continue
        efforts: list[str] = []
        for item in items:
            if isinstance(item, str) and item.strip():
                efforts.append(item)
            elif isinstance(item, Mapping):
                for sub_key in ("reasoningEffort", "effort", "value", "id"):
                    value = item.get(sub_key)
                    if isinstance(value, str) and value.strip():
                        efforts.append(value)
                        break
        return efforts
    return None


def normalize_model_entry(raw: Any) -> dict[str, Any] | None:
    """One worker-native model entry normalized to the discovery shape.

    The identifier is returned exactly as the worker reported it. ``name`` and
    ``supported_efforts`` stay ``None`` when the worker did not offer them;
    effort values are the worker's own offered labels, not values Bridge
    accepts as input.
    """
    if not isinstance(raw, Mapping):
        return None
    ident = raw.get("id")
    if not isinstance(ident, str) or not ident.strip():
        return None
    name = raw.get("displayName") or raw.get("name")
    if not isinstance(name, str) or not name.strip():
        name = None
    return {
        "id": ident,
        "name": name,
        "is_default": raw.get("is_default") is True or raw.get("isDefault") is True,
        "supported_efforts": _efforts_of(raw),
    }


def normalize_model_list(entries: Sequence[Any]) -> tuple[list[dict[str, Any]], str | None, list[str]]:
    """Normalize a raw entry list; dedupe by id, warn on malformed rows."""
    warnings: list[str] = []
    models: list[dict[str, Any]] = []
    seen: set[str] = set()
    default_id: str | None = None
    defaults_seen = 0
    skipped = 0
    for raw in entries:
        entry = normalize_model_entry(raw)
        if entry is None:
            skipped += 1
            continue
        if entry["id"] in seen:
            continue
        seen.add(entry["id"])
        models.append(entry)
        if entry["is_default"]:
            defaults_seen += 1
            if default_id is None:
                default_id = entry["id"]
    if defaults_seen > 1:
        warnings.append(f"worker reported {defaults_seen} default models; keeping {default_id}")
        for model in models:
            model["is_default"] = model["id"] == default_id
    if skipped:
        plural = "entry" if skipped == 1 else "entries"
        warnings.append(f"skipped {skipped} malformed model {plural}")
    return models, default_id, warnings


def _models_from_payload(payload: Any) -> tuple[list[dict[str, Any]], str | None, list[str]]:
    warnings: list[str] = []
    raw_default: Any = None
    if isinstance(payload, Mapping):
        raw_entries = payload.get("models")
        if raw_entries is None:
            raw_entries = payload.get("data")
        raw_default = payload.get("default_model", payload.get("defaultModel"))
    elif isinstance(payload, list):
        raw_entries = payload
    else:
        raw_entries = None
    if not isinstance(raw_entries, list):
        return [], None, ["unrecognized JSON payload from agy models"]
    models, default_id, list_warnings = normalize_model_list(raw_entries)
    warnings.extend(list_warnings)
    if isinstance(raw_default, str) and raw_default.strip() and default_id is None:
        if any(model["id"] == raw_default for model in models):
            default_id = raw_default
            for model in models:
                model["is_default"] = model["id"] == default_id
        else:
            warnings.append("agy default model was absent from its advertised catalog")
    return models, default_id, warnings


def parse_agy_models_output(stdout: str) -> tuple[list[dict[str, Any]], str | None, list[str]]:
    """Parse ``agy models`` output.

    The verified AGY 1.4 format is tab-separated slug and display name.
    Also accept structured catalogs or one slug per line, optionally with
    bullets, ANSI styling, and a ``(default)`` marker. Other text is ignored
    with a warning rather than guessed into a model id.
    """
    warnings: list[str] = []
    text = (stdout or "").strip()
    if not text:
        return [], None, ["agy models returned no output"]
    if text[0] in "[{":
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = None
        if payload is not None and isinstance(payload, (Mapping, list)):
            models, default_id, json_warnings = _models_from_payload(payload)
            if models or not any("unrecognized" in w for w in json_warnings):
                return models, default_id, json_warnings
            # A JSON-looking payload we cannot map: fall through to the text
            # parse so a slug listing that merely starts with [ or { survives.
            warnings.extend(json_warnings)
    models = []
    seen: set[str] = set()
    default_id = None
    skipped = 0
    for raw_line in strip_ansi(text).splitlines():
        line = raw_line.strip()
        if not line or set(line) <= _AGY_BORDER_CHARS:
            continue
        line = _AGY_BULLET_RE.sub("", line).strip()
        if not line:
            continue
        name = None
        if "\t" in line:
            columns = line.split("\t", 1)
            line, name = columns[0].strip(), columns[1].strip() or None
        is_default = False
        suffix = _AGY_DEFAULT_SUFFIX_RE.search(line)
        if suffix:
            is_default = True
            line = line[: suffix.start()].strip()
        if name and (suffix := _AGY_DEFAULT_SUFFIX_RE.search(name)):
            is_default = True
            name = name[:suffix.start()].strip() or None
        if not _AGY_SLUG_RE.fullmatch(line):
            skipped += 1
            continue
        if line not in seen:
            seen.add(line)
            models.append(
                {"id": line, "name": name, "is_default": is_default, "supported_efforts": None}
            )
        if is_default and default_id is None:
            default_id = line
    if skipped:
        warnings.append(f"ignored {skipped} line(s) that do not look like model slugs")
    if not models:
        warnings.append("no model slugs recognized in agy models output")
    else:
        warnings.append("parsed agy models text output; effort options are unavailable")
    return models, default_id, warnings


async def _drain_stderr(proc: asyncio.subprocess.Process) -> str:
    if proc.stderr is None:
        return ""
    tail = ""
    try:
        while True:
            line = await proc.stderr.readline()
            if not line:
                return tail
            text = line.decode("utf-8", errors="replace").rstrip()
            if text:
                tail = f"{tail}\n{text}"[-DISCOVERY_STDERR_TAIL_LIMIT:]
    except (ValueError, OSError):
        log.warning(
            "discovery stderr drain aborted (line over %s MiB or stream closed); later stderr evidence is lost",
            DISCOVERY_STDIO_LIMIT // (1024 * 1024),
            exc_info=True,
        )
        return tail


async def _read_bounded_stdout(proc: asyncio.subprocess.Process, warnings: list[str]) -> str:
    """Read stdout to EOF, keeping at most the byte/line caps.

    Lines past the caps are still drained (the worker must not block on a full
    pipe) but not retained.
    """
    assert proc.stdout is not None
    parts: list[bytes] = []
    total = 0
    truncated = False
    while True:
        try:
            line = await proc.stdout.readline()
        except ValueError as exc:
            raise RuntimeError(
                f"worker emitted a stdout line longer than {DISCOVERY_STDIO_LIMIT // (1024 * 1024)} MiB"
            ) from exc
        if not line:
            break
        if total + len(line) <= DISCOVERY_OUTPUT_BYTE_LIMIT and len(parts) < DISCOVERY_LINE_LIMIT:
            parts.append(line)
            total += len(line)
        elif not truncated:
            warnings.append("worker output exceeded discovery bounds; extra output was discarded")
            truncated = True
            # Keep draining without retaining so the pipe never backs up.
    return b"".join(parts).decode("utf-8", errors="replace")


async def run_bounded_command(
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    cwd: str,
    timeout: float | None = None,
    stdin_bytes: bytes | None = None,
) -> tuple[int | None, str, str, list[str]]:
    """Run one short-lived read-only worker command with bounded output.

    Returns ``(returncode, stdout_text, stderr_tail, warnings)``. Raises
    ``RuntimeError`` on timeout; the process is always reaped, including when
    the caller's task is cancelled mid-run.
    """
    warnings: list[str] = []
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE if stdin_bytes is not None else asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=dict(env),
        cwd=cwd,
        limit=DISCOVERY_STDIO_LIMIT,
        **kwargs,
    )
    stderr_task = asyncio.create_task(_drain_stderr(proc))

    async def _exchange() -> str:
        if proc.stdin is not None:
            try:
                if stdin_bytes is not None:
                    proc.stdin.write(stdin_bytes)
                await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
            with contextlib.suppress(BrokenPipeError, ConnectionResetError, OSError):
                proc.stdin.close()
        stdout_text = await _read_bounded_stdout(proc, warnings)
        await proc.wait()
        return stdout_text

    try:
        stdout_text = await asyncio.wait_for(_exchange(), timeout=timeout)
        if stderr_task.done():
            with contextlib.suppress(asyncio.CancelledError):
                stderr_tail = stderr_task.result()
        else:
            stderr_tail = await stderr_task
    except TimeoutError:
        await reap_subprocess(proc)
        raise RuntimeError(f"worker command did not finish within {timeout}s") from None
    finally:
        if not stderr_task.done():
            stderr_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stderr_task
        # A registry deadline can land after a successful response while this
        # block is already reaping; finish the kill escalation regardless.
        cleanup = asyncio.create_task(reap_subprocess(proc, timeout=2.0))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise
    return proc.returncode, stdout_text, stderr_tail, warnings


async def rpc_send(proc: asyncio.subprocess.Process, payload: Mapping[str, Any]) -> None:
    assert proc.stdin is not None
    proc.stdin.write((json.dumps(payload) + "\n").encode("utf-8"))
    await proc.stdin.drain()


def _rpc_error_text(error: Any) -> str:
    if isinstance(error, Mapping):
        message = error.get("message")
        if isinstance(message, str) and message:
            return message
        return json.dumps(error, ensure_ascii=False)[:200]
    return str(error)[:200]


async def rpc_read_response(proc: asyncio.subprocess.Process, wanted_id: int) -> Mapping[str, Any]:
    """Read JSON-RPC lines until the response for ``wanted_id`` arrives.

    Notifications and unparseable lines are skipped; an error envelope raises
    with a redacted message.
    """
    assert proc.stdout is not None
    while True:
        try:
            line = await proc.stdout.readline()
        except ValueError as exc:
            raise RuntimeError(
                f"worker emitted a stdout line longer than {DISCOVERY_STDIO_LIMIT // (1024 * 1024)} MiB"
            ) from exc
        if not line:
            raise RuntimeError("worker closed stdout before answering the request")
        text = line.decode("utf-8", errors="replace").strip()
        if not text:
            continue
        try:
            message = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(message, Mapping) or message.get("id") != wanted_id:
            continue
        if "error" in message:
            raise RuntimeError(f"worker request failed: {redact_diagnostic(_rpc_error_text(message['error']))}")
        result = message.get("result")
        if not isinstance(result, Mapping):
            raise RuntimeError("worker returned a malformed result payload")
        return result


async def codex_app_server_list_models(
    argv: Sequence[str],
    *,
    env: Mapping[str, str],
    cwd: str,
    timeout: float | None = None,
) -> list[Any]:
    """``initialize`` → ``initialized`` → paginated ``model/list`` over stdio.

    ``argv`` must already include any ``-c`` config overrides and end with
    ``app-server``. Hidden models are excluded from the selectable catalog.
    Raises ``RuntimeError`` (process reaped) on timeout, RPC errors, or a
    malformed payload; propagates ``CancelledError`` after killing the worker.
    """
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    proc = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
        env=dict(env),
        cwd=cwd,
        limit=DISCOVERY_STDIO_LIMIT,
        **kwargs,
    )
    entries: list[Any] = []

    async def _exchange() -> None:
        await rpc_send(
            proc,
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"clientInfo": CLIENT_INFO}},
        )
        await rpc_read_response(proc, 1)
        await rpc_send(proc, {"jsonrpc": "2.0", "method": "initialized", "params": {}})
        cursor: str | None = None
        seen_cursors: set[str] = set()
        for page in range(MODEL_LIST_MAX_PAGES):
            params: dict[str, Any] = {"limit": MODEL_LIST_PAGE_LIMIT, "includeHidden": False}
            if cursor:
                params["cursor"] = cursor
            request_id = page + 2
            await rpc_send(proc, {"jsonrpc": "2.0", "id": request_id, "method": "model/list", "params": params})
            result = await rpc_read_response(proc, request_id)
            data = result.get("data")
            if not isinstance(data, list):
                raise RuntimeError("Codex model/list returned missing or malformed data")
            entries.extend(data)
            next_cursor = result.get("nextCursor")
            if next_cursor is not None and not isinstance(next_cursor, str):
                raise RuntimeError("Codex model/list returned a malformed nextCursor")
            cursor = next_cursor if isinstance(next_cursor, str) and next_cursor else None
            if cursor is None:
                break
            if cursor in seen_cursors:
                raise RuntimeError("model/list did not finish paginating: repeated cursor")
            seen_cursors.add(cursor)
        else:
            raise RuntimeError(f"model/list did not finish paginating within {MODEL_LIST_MAX_PAGES} pages")
        with contextlib.suppress(BrokenPipeError, ConnectionResetError, OSError):
            assert proc.stdin is not None
            proc.stdin.close()
        await proc.wait()

    try:
        await asyncio.wait_for(_exchange(), timeout=timeout)
    except TimeoutError:
        await reap_subprocess(proc)
        raise RuntimeError(f"model/list did not finish within {timeout}s") from None
    finally:
        cleanup = asyncio.create_task(reap_subprocess(proc, timeout=2.0))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise
    return entries
