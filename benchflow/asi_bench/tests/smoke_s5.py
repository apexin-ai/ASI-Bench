"""Opt-in S5 dummy Docker acceptance, not a solver or production exporter.

Requires ASI_S5_PREPARED, ASI_S5_RAW and --docker. Model configuration is read
from ASI_S5_MODEL_ENV (default asi-model.env); the dummy makes no model calls.
Artifacts are only exported after this synchronous dummy has disconnected.
This does not guarantee snapshot safety for arbitrary agent child processes.
"""

import asyncio
import base64
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

import benchflow as bf
from benchflow.agents.registry import register_agent
from benchflow.usage_tracking import UsageTrackingConfig
from benchmarks.asi_bench.solve import (
    collect_outputs,
    inspect_prepared_task,
    persisted_outputs,
    result_layout,
    rollout_contract_options,
)
from benchmarks.asi_bench.tests.test_solve import upstream_workspace

ROOT = Path(tempfile.mkdtemp(prefix="asi-s5-"))
EVIDENCE = ROOT / "smoke-evidence"
EVIDENCE.mkdir()
LAYOUT = result_layout(ROOT, "math.mpsc_safety_filter",
                       "math.mpsc_safety_filter__seed31415", "b1")
TASK = Path(os.environ["ASI_S5_PREPARED"])
RAW = Path(os.environ["ASI_S5_RAW"])
FIXTURES = Path(__file__).parent / "fixtures"


def snapshot(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob("*")
        if p.is_file()
    }


before = snapshot(TASK)
raw_before = snapshot(RAW)
c = inspect_prepared_task(TASK)
meta = yaml.safe_load(
    (RAW / "task_bundle/tasks/math/mpsc_safety_filter/task_meta.yaml").read_text()
)
upstream_workspace(
    RAW / "instance/math.mpsc_safety_filter__seed31415",
    EVIDENCE / "asi-input",
    meta,
    EVIDENCE / "framework",
)
inputs = TASK / "environment/inputs"
assert set(c.inputs) == set(snapshot(EVIDENCE / "asi-input"))
for name in c.inputs:
    if name == "task_info.json":
        assert json.loads((inputs / name).read_text()) == json.loads(
            (EVIDENCE / "asi-input" / name).read_text()
        )
    else:
        assert (inputs / name).read_bytes() == (EVIDENCE / "asi-input" / name).read_bytes()
report = {
    "prepared": str(TASK),
    "input_tree_parity": True,
    "input_hashes": snapshot(inputs),
    "sources": c.sources,
}
(EVIDENCE / "static.json").write_text(json.dumps(report, indent=2))
print("EVIDENCE", ROOT, flush=True)
if "--docker" not in sys.argv:
    assert snapshot(TASK) == before and snapshot(RAW) == raw_before
    print("read-only input parity passed")
    sys.exit()
# Minimal real ACP agent from previous smoke; add actual networking evidence.
dummy = (FIXTURES / "s5_dummy.py").read_text()
payload = base64.b64encode(dummy.encode()).decode()
register_agent(
    "asi-s5-dummy",
    f"echo {payload} | base64 -d > /usr/local/bin/asi-s5-dummy.py",
    "python3 /usr/local/bin/asi-s5-dummy.py",
)


async def main():
    from urllib.parse import urlsplit, urlunsplit

    from dotenv import dotenv_values

    settings = dotenv_values(os.environ.get("ASI_S5_MODEL_ENV", "asi-model.env"))
    url = urlsplit(settings["ASI_MODEL_BASE_URL"])
    base = (
        urlunsplit((url.scheme, url.netloc, "/v1", url.query, url.fragment))
        if url.path.rstrip("/") == "/v"
        else settings["ASI_MODEL_BASE_URL"]
    )
    r = await bf.Rollout.create(
        bf.RolloutConfig(
            **rollout_contract_options(c),
            agent="asi-s5-dummy",
            model="vllm/" + settings["ASI_MODEL_ID"],
            agent_env={
                "BENCHFLOW_PROVIDER_BASE_URL": base,
                "BENCHFLOW_PROVIDER_API_KEY": settings["ASI_MODEL_API_KEY"],
            },
            jobs_dir=LAYOUT.benchflow_dir,
            job_name="asi-cli-network",
            usage_tracking=UsageTrackingConfig(mode="off"),
        )
    )
    try:
        for phase in (
            "setup",
            "start",
            "install_agent",
            "connect",
            "execute",
            "disconnect",
        ):
            report["phase"] = phase
            (EVIDENCE / "progress.json").write_text(json.dumps(report, indent=2))
            print("PHASE", phase, flush=True)
            if phase == "connect":
                cid = await r.env._main_container_id()
                inspected = json.loads(
                    subprocess.check_output(["docker", "inspect", cid])
                )[0]
                host = inspected["HostConfig"]
                network = {
                    "network_mode": host["NetworkMode"],
                    "cap_add": host["CapAdd"],
                    "privileged": host["Privileged"],
                    "networks": list(inspected["NetworkSettings"]["Networks"]),
                }
                code = (FIXTURES / "s5_network_probe.py").read_text()
                probe = await r.env.exec(
                    "python3 -c " + shlex.quote(code), cwd="/workspace", user="agent"
                )
                network["agent_probe"] = json.loads(probe.stdout)
                probe = await r.env.exec(
                    "python3 -c "
                    + shlex.quote(
                        "from pathlib import Path; print(next(s for s in Path('/proc/self/status').read_text().splitlines() if s.startswith('CapEff:')))"
                    ),
                    user="root",
                )
                network["root_has_net_admin"] = bool(
                    int(probe.stdout.split(":")[1].strip(), 16) & (1 << 12)
                )
                assert network["privileged"] is False
                assert network["root_has_net_admin"] is True
                assert network["network_mode"] != "none"
                assert network["agent_probe"]["uid"] != 0
                assert network["agent_probe"]["files"] == sorted(c.inputs)
                assert not any(network["agent_probe"]["hidden_readable"].values())
                (EVIDENCE / "docker-network.json").write_text(json.dumps(network, indent=2))
                print("DOCKER_NETWORK", json.dumps(network), flush=True)
            await getattr(r, phase)()
        await r.env.download_file("/logs/agent/asi-probe.json", EVIDENCE / "probe.json")
        probe = json.loads((EVIDENCE / "probe.json").read_text())
        assert (
            "".join(b["text"] for b in probe["prompt_blocks"] if b["type"] == "text")
            == c.prompt
        )
        interfaces = [
            line.split(":")[0].strip()
            for line in probe["net_devices"].splitlines()
            if ":" in line
        ]
        assert any(name != "lo" for name in interfaces), interfaces
        assert probe["workspace_before"] == sorted(c.inputs)
        assert probe["process_cwd"] == probe["session_cwd"] == "/workspace"
        exported = LAYOUT.outputs_dir
        exported.mkdir(parents=True, exist_ok=False)
        for spec in c.outputs:
            await r.env.download_file("/workspace/" + spec.name, exported / spec.name)
        collected = collect_outputs(exported, c.outputs)
        assert collected["prediction_status"] == "present", collected
        for artifact in collected["artifacts"]:
            assert artifact["sha256"] == probe["outputs"][artifact["name"]]["sha256"]
        (EVIDENCE / "collection.json").write_text(json.dumps(collected, indent=2))
        persisted = persisted_outputs(LAYOUT, c.outputs, collected)
        assert LAYOUT.result_file.parent / persisted["dir"] == exported
        assert {x["path"] for x in persisted["files"]} == {s.name for s in c.outputs}
        for item in persisted["files"]:
            content = (exported / item["path"]).read_bytes()
            assert item["bytes"] == len(content)
            assert item["sha256"] == hashlib.sha256(content).hexdigest()
        (EVIDENCE / "persisted_outputs.json").write_text(json.dumps(persisted, indent=2))
        report.update(
            persisted_outputs=persisted,
            collection=collected,
            attempt_status="completed",
            evaluation_status="pending",
            agent_kind="dummy",
        )
        firewall = await r.env.exec(
            "iptables -S OUTPUT; ip6tables -S OUTPUT", user="root"
        )
        assert firewall.return_code == 0, firewall.stderr
        assert "--uid-owner 1000 -j REJECT" in firewall.stdout
        (EVIDENCE / "firewall.txt").write_text(firewall.stdout)
        caps = await r.env.exec("grep CapEff /proc/self/status", user="agent")
        assert int(caps.stdout.split(":")[1].strip(), 16) == 0
        report["agent_effective_capabilities_zero"] = True
        result = await r.finalize()
        assert result.error is None, result.error
        report.update(
            acp_prompt_exact=True,
            interfaces=interfaces,
            routes=probe["routes"],
            error=result.error,
            verifier_error=result.verifier_error,
            scoring=result.scoring,
        )
    except Exception as exc:
        report.update(
            status="failed",
            error_type=type(exc).__name__,
            error=type(exc).__name__ + " (inspect local logs with secrets redacted)",
        )
        (EVIDENCE / "failure.json").write_text(json.dumps(report, indent=2))
        raise RuntimeError(
            "S5 smoke failed: " + type(exc).__name__ + "; see failure.json"
        ) from None
    finally:
        await r.cleanup()
        (EVIDENCE / "integrity.json").write_text(
            json.dumps(
                {
                    "raw_unchanged": snapshot(RAW) == raw_before,
                    "prepared_unchanged": snapshot(TASK) == before,
                }
            )
        )
    assert snapshot(TASK) == before and snapshot(RAW) == raw_before
    report["raw_and_prepared_unchanged"] = True
    (EVIDENCE / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


asyncio.run(main())
