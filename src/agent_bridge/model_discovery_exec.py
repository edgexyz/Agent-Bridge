"""Worker model catalog discovery and parsing."""

from __future__ import annotations

import asyncio
import contextlib
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from agent_bridge.diagnostics import redact_diagnostic
from agent_bridge.processes import reap_subprocess
from agent_bridge.quota_codex import CLIENT_INFO, READ_LIMIT, _read_response, _send

AGY_OUTPUT_LIMIT = 4 * 1024 * 1024
AGY_STDERR_LIMIT = 16000
MODEL_LIST_MAX_PAGES = 100
_ANSI_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def normalize_model_list(entries: Sequence[Any]) -> tuple[list[dict[str, Any]], str | None, list[str]]:
    """Normalize the documented Codex model/list schema, preserving level order."""
    models = []
    seen = set()
    warnings = []
    default = None
    for entry in entries:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("id"), str) or not entry["id"].strip():
            warnings.append("Codex model/list contained a row without a model ID")
            continue
        ident = entry["id"]
        if ident in seen:
            continue
        seen.add(ident)
        raw_efforts = entry.get("supportedReasoningEfforts")
        efforts = None
        if isinstance(raw_efforts, list):
            parsed = [
                item["reasoningEffort"]
                for item in raw_efforts
                if isinstance(item, Mapping)
                and isinstance(item.get("reasoningEffort"), str)
                and item["reasoningEffort"]
            ]
            efforts = parsed if parsed or not raw_efforts else None
            if raw_efforts and not parsed:
                warnings.append(f"Codex did not provide recognizable reasoning levels for {ident}")
        if entry.get("isDefault") is True:
            if default is None:
                default = ident
            else:
                warnings.append("Codex reported multiple default models; keeping the first")
        models.append(
            {
                "id": ident,
                "name": entry.get("displayName") if isinstance(entry.get("displayName"), str) else None,
                "is_default": ident == default,
                "supported_efforts": efforts,
            }
        )
    return models, default, warnings


def parse_agy_models_output(stdout: str) -> list[dict[str, Any]]:
    """Parse AGY tab-separated model ID and display name lines."""
    models = []
    seen = set()
    for line in _ANSI_RE.sub("", stdout).splitlines():
        if not line.strip():
            continue
        fields = line.split("\t")
        if len(fields) != 2:
            raise RuntimeError("Unrecognized agy models output: expected tab-separated ID and name")
        ident, name = (field.strip() for field in fields)
        if not ident or any(char.isspace() for char in ident) or not name:
            raise RuntimeError("Unrecognized agy models output: missing or invalid ID/name")
        if ident in seen:
            continue
        seen.add(ident)
        models.append({"id": ident, "name": name, "is_default": False, "supported_efforts": None})
    if not models:
        raise RuntimeError("agy models returned no model catalog")
    return models


async def _agy_stdout(proc: asyncio.subprocess.Process) -> str:
    assert proc.stdout is not None
    parts = []
    size = 0
    while chunk := await proc.stdout.read(65536):
        size += len(chunk)
        if size > AGY_OUTPUT_LIMIT:
            raise RuntimeError("agy models exceeded the output limit")
        parts.append(chunk)
    return b"".join(parts).decode("utf-8", errors="replace")


async def _agy_stderr(proc: asyncio.subprocess.Process) -> bytes:
    assert proc.stderr is not None
    tail = b""
    while chunk := await proc.stderr.read(65536):
        tail = (tail + chunk)[-AGY_STDERR_LIMIT:]
    return tail


async def read_agy_models(command: Sequence[str], *, env: Mapping[str, str], cwd: str) -> str:
    """Run agy models; caller owns deadline, this function always reaps its child."""
    kwargs: dict[str, Any] = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    proc = await asyncio.create_subprocess_exec(
        *command,
        "models",
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=dict(env),
        cwd=cwd,
        **kwargs,
    )
    stderr = asyncio.create_task(_agy_stderr(proc))
    try:
        stdout = await _agy_stdout(proc)
        await proc.wait()
        diagnostic = (await stderr).decode("utf-8", errors="replace").strip()
        if proc.returncode != 0:
            raise RuntimeError(f"agy models exited with code {proc.returncode}: {redact_diagnostic(diagnostic)}")
        return stdout
    finally:
        cleanup = asyncio.create_task(reap_subprocess(proc, timeout=2.0))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise
        finally:
            stderr.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stderr


async def codex_app_server_list_models(argv: Sequence[str], *, env: Mapping[str, str], cwd: str) -> list[Any]:
    """Read visible pages using the existing Codex stdio request helpers."""
    entries: list[Any] = []
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
        limit=READ_LIMIT,
        **kwargs,
    )
    try:
        await _send(proc, {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"clientInfo": CLIENT_INFO}})
        await _read_response(proc, 1)
        await _send(proc, {"jsonrpc": "2.0", "method": "initialized", "params": {}})
        cursor = None
        seen_cursors = set()
        for page in range(MODEL_LIST_MAX_PAGES):
            params: dict[str, Any] = {"limit": 100, "includeHidden": False}
            if cursor:
                params["cursor"] = cursor
            request_id = page + 2
            await _send(proc, {"jsonrpc": "2.0", "id": request_id, "method": "model/list", "params": params})
            result = await _read_response(proc, request_id)
            if not isinstance(result.get("data"), list):
                raise RuntimeError("Codex model/list returned missing or malformed data")
            entries.extend(result["data"])
            cursor = result.get("nextCursor")
            if cursor is None or cursor == "":
                return entries
            if not isinstance(cursor, str):
                raise RuntimeError("Codex model/list returned a malformed nextCursor")
            if cursor in seen_cursors:
                raise RuntimeError("Codex model/list repeated a pagination cursor")
            seen_cursors.add(cursor)
        raise RuntimeError("Codex model/list exceeded the page limit")
    except Exception as exc:
        raise RuntimeError(f"Codex model discovery failed: {redact_diagnostic(str(exc))}") from None
    finally:
        cleanup = asyncio.create_task(reap_subprocess(proc, timeout=2.0))
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await cleanup
            raise
