from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path

import pytest

from agent_bridge import model_discovery_exec
from agent_bridge.adapters.antigravity import AgyAdapter
from agent_bridge.adapters.codex import CodexAdapter
from agent_bridge.config import AgentConfig, EnvConfig
from agent_bridge.model_discovery_exec import (
    codex_app_server_list_models,
    normalize_model_entry,
    normalize_model_list,
    parse_agy_models_output,
    run_bounded_command,
)

FAKE_AGY = r'''
import json
import os
import sys
import time

mode = os.environ.get("FAKE_AGY_MODE", "text")

if mode == "exit3":
    print("Authorization: Bearer sk-testsecret123", file=sys.stderr)
    print("boom: token=apikeyhunter2", file=sys.stderr)
    sys.exit(3)

if mode == "hang":
    time.sleep(30)
    sys.exit(0)

if mode == "json":
    print(json.dumps({
        "models": [
            {"id": "gemini-3-pro", "name": "Gemini 3 Pro", "isDefault": True,
             "supportedEfforts": ["low", "high"]},
            {"id": "claude-x", "name": "Claude X", "isDefault": False,
             "supportedEfforts": []},
            {"name": "no id here"},
        ],
        "defaultModel": "gemini-3-pro",
    }))
    sys.exit(0)

print("\x1b[32m gemini-3.8-flash-medium \x1b[0m")
print("claude-sonnet-4-5 (default)")
print("2) gemini-3-pro")
print("not a model line with spaces")
print("---")
report = os.environ.get("FAKE_AGY_REPORT")
if report:
    with open(report, "w", encoding="utf-8") as fh:
        json.dump({
            "cwd": os.getcwd(),
            "denied_present": "DISCOVERY_SECRET" in os.environ,
            "agent_env": os.environ.get("AGY_MARKER"),
            "worker_context": os.environ.get("AGENT_BRIDGE_PARENT_CONTEXT"),
        }, fh)
sys.exit(0)
'''

FAKE_CODEX = r'''
import json
import os
import sys
import time

argv = sys.argv[1:]

if argv and argv[-1] != "app-server":
    print("unknown command", file=sys.stderr)
    sys.exit(2)

report = os.environ.get("FAKE_CODEX_REPORT")
requests = []

def emit(payload):
    text = json.dumps(payload)
    print(text, flush=True)

mode = os.environ.get("FAKE_CODEX_MODELS_MODE", "ok")

if mode == "hang":
    time.sleep(30)
    sys.exit(0)

for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    msg = json.loads(line)
    requests.append(msg)
    method = msg.get("method")
    if method == "initialize":
        emit({"id": msg["id"], "result": {"userAgent": "fake-codex"}})
        continue
    if method == "initialized":
        continue
    if method == "model/list":
        params = msg.get("params") or {}
        if mode == "rpc_error":
            emit({"id": msg["id"], "error": {"code": -32000,
                  "message": "Authorization: Bearer sk-leakedtok987"}})
            continue
        if mode == "bad_page":
            result = {"data": "not-a-list"}
        elif mode == "bad_cursor":
            result = {"data": [], "nextCursor": 17}
        elif mode == "malformed":
            result = {"data": [
                {"id": "gpt-good", "displayName": "Good", "isDefault": True},
                "not-a-mapping",
                {"displayName": "missing id"},
            ], "nextCursor": None}
        elif mode == "cursor_loop":
            result = {"data": [{"id": "gpt-a"}], "nextCursor": "again"}
        elif params.get("cursor") == "page2":
            result = {"data": [{"id": "gpt-b", "displayName": "B", "isDefault": False,
                                "supportedReasoningEfforts": [{"reasoningEffort": "high"}]}],
                      "nextCursor": None}
        else:
            result = {"data": [{"id": "gpt-a", "model": "gpt-a", "displayName": "A",
                                "isDefault": True, "hidden": False,
                                "supportedReasoningEfforts": [
                                    {"reasoningEffort": "medium"},
                                    {"reasoningEffort": "xhigh"},
                                ]}],
                      "nextCursor": "page2"}
        emit({"id": msg["id"], "result": result})
        continue
    emit({"id": msg["id"], "error": {"code": -32601, "message": "unknown method"}})

if report:
    with open(report, "w", encoding="utf-8") as fh:
        json.dump({"argv": argv, "cwd": os.getcwd(), "requests": requests}, fh)
sys.exit(0)
'''


def _write_script(tmp_path: Path, name: str, source: str) -> Path:
    script = tmp_path / name
    script.write_text(source, encoding="utf-8")
    return script


def _agy_adapter(home: Path, command: list[str], *, agent_env=None, env_config=None) -> AgyAdapter:
    agent = AgentConfig(name="agy", protocol="antigravity", command=command, env=agent_env or {})
    return AgyAdapter(agent, home, env_config)


def _codex_adapter(home: Path, command: list[str], *, overrides=None) -> CodexAdapter:
    agent = AgentConfig(
        name="codex",
        protocol="codex",
        command=command,
        config_overrides=overrides or [],
    )
    return CodexAdapter(agent, home)


# --- pure helpers -------------------------------------------------------------


def test_normalize_model_entry_verbatim_id_and_raw_efforts():
    entry = normalize_model_entry(
        {
            "id": "gpt-6.1-sol",
            "displayName": "GPT-6.1 Sol",
            "isDefault": True,
            "supportedReasoningEfforts": [
                {"reasoningEffort": "medium", "description": "balances"},
                {"reasoningEffort": "xhigh"},
                "raw-string-effort",
                {"no_effort_key": True},
            ],
        }
    )
    assert entry == {
        "id": "gpt-6.1-sol",
        "name": "GPT-6.1 Sol",
        "is_default": True,
        "supported_efforts": ["medium", "xhigh", "raw-string-effort"],
    }
    # Underscore spellings and missing optionals.
    entry = normalize_model_entry({"id": "m", "name": "  ", "is_default": False})
    assert entry == {"id": "m", "name": None, "is_default": False, "supported_efforts": None}
    # id must be a non-empty string; nothing is invented.
    assert normalize_model_entry({"displayName": "no id"}) is None
    assert normalize_model_entry("not-a-mapping") is None
    assert normalize_model_entry({"id": "   "}) is None


def test_normalize_model_list_dedupes_and_warns():
    models, default_id, warnings = normalize_model_list(
        [
            {"id": "a", "isDefault": True},
            {"id": "a", "isDefault": False},
            {"id": "b", "isDefault": True},
            {"oops": 1},
            "garbage",
        ]
    )
    assert [m["id"] for m in models] == ["a", "b"]
    assert default_id == "a"
    assert any("2 default models" in w for w in warnings)
    assert any("skipped 2 malformed" in w for w in warnings)


def test_parse_agy_models_output_json_payload():
    models, default_id, warnings = parse_agy_models_output(
        json.dumps(
            {
                "models": [
                    {"id": "gemini-3-pro", "name": "Gemini 3 Pro", "isDefault": True,
                     "supportedEfforts": ["low", "high"]},
                    {"id": "claude-x", "supportedEfforts": []},
                ]
            }
        )
    )
    assert [m["id"] for m in models] == ["gemini-3-pro", "claude-x"]
    assert models[0]["supported_efforts"] == ["low", "high"]
    assert models[1]["supported_efforts"] == []
    assert default_id == "gemini-3-pro"
    assert not any("text output" in w for w in warnings)


def test_parse_agy_models_output_text_bullets_default_ansi_junk():
    stdout = (
        "\x1b[32m gemini-3.8-flash-medium \x1b[0m\n"
        "claude-sonnet-4-5 (default)\n"
        "2) gemini-3-pro\n"
        "not a model line with spaces\n"
        "---\n"
    )
    models, default_id, warnings = parse_agy_models_output(stdout)
    assert [m["id"] for m in models] == ["gemini-3.8-flash-medium", "claude-sonnet-4-5", "gemini-3-pro"]
    assert all(m["name"] is None and m["supported_efforts"] is None for m in models)
    assert [m["is_default"] for m in models] == [False, True, False]
    assert default_id == "claude-sonnet-4-5"
    assert any("ignored 1 line(s)" in w for w in warnings)
    assert any("text output" in w for w in warnings)


def test_parse_agy_models_output_empty_and_unrecognized():
    assert parse_agy_models_output("") == ([], None, ["agy models returned no output"])
    models, default_id, warnings = parse_agy_models_output('{"totally": "different"}')
    assert models == [] and default_id is None
    assert any("unrecognized JSON" in w for w in warnings)


# --- agy adapter --------------------------------------------------------------


async def test_agy_list_models_text_success(bridge_home, tmp_path):
    script = _write_script(tmp_path, "fake_agy.py", FAKE_AGY)
    report = tmp_path / "report.json"
    adapter = _agy_adapter(
        bridge_home,
        [sys.executable, str(script)],
        agent_env={"AGY_MARKER": "yes", "FAKE_AGY_REPORT": str(report)},
    )
    cwd = str(tmp_path / "work")
    Path(cwd).mkdir()
    result = await adapter.list_models(cwd)
    assert result["source"] == "agy models"
    assert [m["id"] for m in result["models"]] == [
        "gemini-3.8-flash-medium",
        "claude-sonnet-4-5",
        "gemini-3-pro",
    ]
    assert result["default_model"] == "claude-sonnet-4-5"
    assert result["models"][0]["supported_efforts"] is None
    evidence = json.loads(report.read_text(encoding="utf-8"))
    assert evidence["cwd"] == cwd  # discovery ran in the requested cwd
    assert evidence["agent_env"] == "yes"  # agent env block respected
    assert evidence["worker_context"] == "worker"  # nested-bridge marker set


async def test_agy_list_models_env_deny_strips_secret(bridge_home, tmp_path, monkeypatch):
    monkeypatch.setenv("DISCOVERY_SECRET", "topsecret")
    script = _write_script(tmp_path, "fake_agy.py", FAKE_AGY)
    report = tmp_path / "report.json"
    adapter = _agy_adapter(
        bridge_home,
        [sys.executable, str(script)],
        agent_env={"FAKE_AGY_REPORT": str(report)},
        env_config=EnvConfig(deny=["DISCOVERY_*"]),
    )
    result = await adapter.list_models(str(tmp_path))
    assert [m["id"] for m in result["models"]] == [
        "gemini-3.8-flash-medium",
        "claude-sonnet-4-5",
        "gemini-3-pro",
    ]
    evidence = json.loads(report.read_text(encoding="utf-8"))
    assert evidence["denied_present"] is False


async def test_agy_list_models_json_payload(bridge_home, tmp_path, monkeypatch):
    script = _write_script(tmp_path, "fake_agy.py", FAKE_AGY)
    monkeypatch.setenv("FAKE_AGY_MODE", "json")
    adapter = _agy_adapter(bridge_home, [sys.executable, str(script)])
    result = await adapter.list_models(str(tmp_path))
    assert result["default_model"] == "gemini-3-pro"
    by_id = {m["id"]: m for m in result["models"]}
    assert by_id["gemini-3-pro"]["name"] == "Gemini 3 Pro"
    assert by_id["gemini-3-pro"]["supported_efforts"] == ["low", "high"]
    assert by_id["gemini-3-pro"]["is_default"] is True
    assert by_id["claude-x"]["supported_efforts"] == []
    # The id-less entry was skipped with a warning, silently dropped otherwise.
    assert any("malformed" in w for w in result["warnings"])


async def test_agy_list_models_nonzero_exit_redacts_stderr(bridge_home, tmp_path, monkeypatch):
    script = _write_script(tmp_path, "fake_agy.py", FAKE_AGY)
    monkeypatch.setenv("FAKE_AGY_MODE", "exit3")
    adapter = _agy_adapter(bridge_home, [sys.executable, str(script)])
    with pytest.raises(RuntimeError, match="exited with code 3") as caught:
        await adapter.list_models(str(tmp_path))
    joined = str(caught.value)
    assert "sk-testsecret123" not in joined
    assert "apikeyhunter2" not in joined


async def test_agy_list_models_timeout_reaps_worker(bridge_home, tmp_path, monkeypatch):
    script = _write_script(tmp_path, "fake_agy.py", FAKE_AGY)
    monkeypatch.setenv("FAKE_AGY_MODE", "hang")
    adapter = _agy_adapter(bridge_home, [sys.executable, str(script)])
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(adapter.list_models(str(tmp_path)), timeout=0.3)
    assert time.monotonic() - started < 10


async def test_agy_list_models_cancellation_kills_worker(bridge_home, tmp_path, monkeypatch):
    script = _write_script(tmp_path, "fake_agy.py", FAKE_AGY)
    monkeypatch.setenv("FAKE_AGY_MODE", "hang")
    adapter = _agy_adapter(bridge_home, [sys.executable, str(script)])
    task = asyncio.create_task(adapter.list_models(str(tmp_path)))
    await asyncio.sleep(0.2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(task, timeout=10)


# --- codex adapter ------------------------------------------------------------


def _patch_codex_command(monkeypatch, script: Path):
    monkeypatch.setattr(
        "agent_bridge.adapters.codex.resolve_codex_command",
        lambda command, fallbacks=None, *, env=None: [sys.executable, str(script)],
    )


async def test_codex_list_models_pagination_overrides_and_cwd(
    bridge_home, tmp_path, monkeypatch
):
    script = _write_script(tmp_path, "fake_codex.py", FAKE_CODEX)
    _patch_codex_command(monkeypatch, script)
    report = tmp_path / "codex_report.json"
    monkeypatch.setenv("FAKE_CODEX_REPORT", str(report))
    override = 'model_providers.custom.base_url="http://localhost:9999/v1"'
    adapter = _codex_adapter(bridge_home, ["codex"], overrides=[override])
    cwd = str(tmp_path / "work")
    Path(cwd).mkdir()
    result = await adapter.list_models(cwd)
    assert result["source"] == "codex app-server model/list"
    by_id = {m["id"]: m for m in result["models"]}
    assert set(by_id) == {"gpt-a", "gpt-b"}
    # Verbatim worker-native values: ids, display names, raw effort labels.
    assert by_id["gpt-a"]["name"] == "A"
    assert by_id["gpt-a"]["supported_efforts"] == ["medium", "xhigh"]
    assert by_id["gpt-a"]["is_default"] is True
    assert by_id["gpt-b"]["supported_efforts"] == ["high"]
    assert result["default_model"] == "gpt-a"
    evidence = json.loads(report.read_text(encoding="utf-8"))
    argv = evidence["argv"]
    assert argv[-1] == "app-server"
    # config_overrides are forwarded as -c before the subcommand.
    assert argv[argv.index("-c") + 1] == override
    assert evidence["cwd"] == cwd
    list_requests = [r for r in evidence["requests"] if r.get("method") == "model/list"]
    assert len(list_requests) == 2  # paginated via nextCursor
    assert list_requests[0]["params"]["includeHidden"] is False
    assert list_requests[1]["params"]["cursor"] == "page2"


async def test_codex_list_models_malformed_entries_warn(bridge_home, tmp_path, monkeypatch):
    script = _write_script(tmp_path, "fake_codex.py", FAKE_CODEX)
    _patch_codex_command(monkeypatch, script)
    monkeypatch.setenv("FAKE_CODEX_MODELS_MODE", "malformed")
    adapter = _codex_adapter(bridge_home, ["codex"])
    result = await adapter.list_models(str(tmp_path))
    assert [m["id"] for m in result["models"]] == ["gpt-good"]
    assert result["default_model"] == "gpt-good"
    assert any("skipped 2 malformed" in w for w in result["warnings"])


async def test_codex_list_models_rpc_error_redacted(bridge_home, tmp_path, monkeypatch):
    script = _write_script(tmp_path, "fake_codex.py", FAKE_CODEX)
    _patch_codex_command(monkeypatch, script)
    monkeypatch.setenv("FAKE_CODEX_MODELS_MODE", "rpc_error")
    adapter = _codex_adapter(bridge_home, ["codex"])
    with pytest.raises(RuntimeError, match="request failed") as caught:
        await adapter.list_models(str(tmp_path))
    joined = str(caught.value)
    assert "sk-leakedtok987" not in joined


async def test_codex_list_models_timeout_reaps_worker(bridge_home, tmp_path, monkeypatch):
    script = _write_script(tmp_path, "fake_codex.py", FAKE_CODEX)
    _patch_codex_command(monkeypatch, script)
    monkeypatch.setenv("FAKE_CODEX_MODELS_MODE", "hang")
    adapter = _codex_adapter(bridge_home, ["codex"])
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(adapter.list_models(str(tmp_path)), timeout=0.3)
    assert time.monotonic() - started < 10


async def test_codex_list_models_command_not_found_raises(
    bridge_home, tmp_path, monkeypatch
):
    def _raise(command, fallbacks=None, *, env=None):
        raise FileNotFoundError("codex not found")

    monkeypatch.setattr("agent_bridge.adapters.codex.resolve_codex_command", _raise)
    adapter = _codex_adapter(bridge_home, ["codex"])
    with pytest.raises(FileNotFoundError, match="codex not found"):
        await adapter.list_models(str(tmp_path))


# --- run_bounded_command ------------------------------------------------------


async def test_run_bounded_command_truncates_extra_output(tmp_path, monkeypatch):
    script = _write_script(
        tmp_path,
        "chatty.py",
        "for i in range(10):\n    print(f'line-{i}')\n",
    )
    monkeypatch.setattr(model_discovery_exec, "DISCOVERY_LINE_LIMIT", 5)
    returncode, stdout, _stderr, warnings = await run_bounded_command(
        [sys.executable, str(script)], env={"PATH": os.environ.get("PATH", "")}, cwd=str(tmp_path)
    )
    assert returncode == 0
    assert stdout.splitlines() == [f"line-{i}" for i in range(5)]
    assert warnings == ["worker output exceeded discovery bounds; extra output was discarded"]


async def test_codex_app_server_cursor_loop_is_bounded(tmp_path):
    script = _write_script(tmp_path, "fake_codex.py", FAKE_CODEX)
    with pytest.raises(RuntimeError, match="did not finish paginating"):
        await codex_app_server_list_models(
            [sys.executable, str(script), "app-server"],
            env={**os.environ, "FAKE_CODEX_MODELS_MODE": "cursor_loop"},
            cwd=str(tmp_path),
            timeout=5.0,
        )


def test_agy_tab_separated_catalog_matches_live_format():
    models, default, warnings = parse_agy_models_output(
        "gemini-4-argon-low\tGemini 4 Argon (Low)\n"
        "claude-sonnet-5-5-high\tClaude Sonnet 5.5 (High)\n"
    )
    assert [(m["id"], m["name"]) for m in models] == [
        ("gemini-4-argon-low", "Gemini 4 Argon (Low)"),
        ("claude-sonnet-5-5-high", "Claude Sonnet 5.5 (High)"),
    ]
    assert default is None
    assert all(m["supported_efforts"] is None for m in models)
    assert not any("ignored" in warning for warning in warnings)


def test_agy_top_level_default_keeps_row_flag_consistent():
    models, default, _ = parse_agy_models_output(json.dumps({
        "defaultModel": "model-a", "models": [{"id": "model-a"}, {"id": "model-b"}],
    }))
    assert default == "model-a"
    assert [m["is_default"] for m in models] == [True, False]


@pytest.mark.parametrize("mode", ["bad_page", "bad_cursor"])
async def test_codex_rejects_malformed_page(tmp_path, mode):
    script = _write_script(tmp_path, "fake_codex.py", FAKE_CODEX)
    with pytest.raises(RuntimeError, match="malformed"):
        await codex_app_server_list_models(
            [sys.executable, str(script), "app-server"],
            env={**os.environ, "FAKE_CODEX_MODELS_MODE": mode},
            cwd=str(tmp_path), timeout=5,
        )


async def test_agy_unknown_output_is_failure(bridge_home, tmp_path):
    script = _write_script(tmp_path, "fake_agy.py", 'print("unexpected response without model IDs")')
    adapter = _agy_adapter(bridge_home, [sys.executable, str(script)])
    with pytest.raises(RuntimeError, match="no model slugs recognized"):
        await adapter.list_models(str(tmp_path))
