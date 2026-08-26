#!/usr/bin/env python3
"""Gate 1 — Environment / Run Manifest.

Records: git SHA, git status, Python version, Torch version, CUDA version,
GPU name, FACodec upstream commit/revision, checkpoint revision.

Run on Colab:
  colab run scripts/gate1_manifest.py --gpu T4
"""
import subprocess, sys, os, json, time, types, warnings, hashlib

warnings.simplefilter("ignore")


# ── Mock audiotools BEFORE any FAcodec imports ──
def _make_mock(name):
    m = types.ModuleType(name)
    m.__path__ = []
    m.__package__ = name
    return m

mock_audio = _make_mock("audiotools")
mock_ml = _make_mock("audiotools.ml")
mock_ml.BaseModel = type("BaseModel", (), {"INTERN": [], "EXTERN": []})
mock_audio.ml = mock_ml
mock_audio.AudioSignal = type("AudioSignal", (), {})
mock_audio.STFTParams = type("STFTParams", (), {})
mock_core = _make_mock("audiotools.core")
mock_core.util = _make_mock("audiotools.core.util")
sys.modules["audiotools"] = mock_audio
sys.modules["audiotools.ml"] = mock_ml
sys.modules["audiotools.core"] = mock_core
sys.modules["audiotools.core.util"] = mock_core.util


def run(cmd, desc="", check=True, timeout=120):
    print(f"\n>>> {desc or cmd[:80]}")
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
    out = r.stdout.strip()
    if out:
        print(out[:500])
    if check and r.returncode != 0:
        print(f"FAILED: {r.stderr[:500]}")
        sys.exit(1)
    return r


def main():
    t_start = time.time()
    env = {}

    # ── GPU ──
    r = run("nvidia-smi --query-gpu=name --format=csv,noheader", "GPU name", check=False)
    env["gpu_name"] = r.stdout.strip()

    # ── Python ──
    env["python_version"] = sys.version.split()[0]

    # ── Deps ──
    run(
        "pip install -q numpy soundfile librosa scipy jiwer pyyaml "
        "huggingface-hub phonemizer speechbrain torchaudio faster-whisper "
        "pytest pyworld munch einops",
        "install deps",
    )

    # ── Clone repos ──
    run(
        "test -d /content/FAcodec || git clone https://github.com/Plachtaa/FAcodec.git /content/FAcodec",
        "clone FAcodec",
        check=False,
    )
    run(
        "test -f /content/FAcodec/modules/__init__.py || touch /content/FAcodec/modules/__init__.py",
        "modules init",
        check=False,
    )
    run(
        "test -d /content/accentedge || git clone --depth 1 https://github.com/yagami009/accentedge.git /content/accentedge",
        "clone accentedge",
        check=False,
    )

    # ── Path setup ──
    sys.path = [p for p in sys.path if "/content" not in p]
    sys.path.insert(0, "/content/FAcodec")
    sys.path.insert(0, "/content/accentedge/src")
    os.environ["PYTHONPATH"] = "/content/FAcodec/modules:" + os.environ.get("PYTHONPATH", "")
    os.chdir("/content/FAcodec")

    # ── Torch / CUDA ──
    import torch
    env["torch_version"] = torch.__version__
    env["cuda_version"] = torch.version.cuda if torch.cuda.is_available() else "N/A"
    env["cuda_available"] = torch.cuda.is_available()

    # ── Git info: accentedge ──
    r = run("cd /content/accentedge && git rev-parse HEAD", "accentedge SHA", check=False)
    env["accentedge_git_sha"] = r.stdout.strip()
    r = run("cd /content/accentedge && git rev-parse --abbrev-ref HEAD", "accentedge branch", check=False)
    env["accentedge_git_branch"] = r.stdout.strip()
    r = run("cd /content/accentedge && git status --short", "accentedge git status", check=False)
    env["accentedge_git_status"] = r.stdout.strip()

    # ── Git info: FAcodec upstream ──
    r = run("cd /content/FAcodec && git rev-parse HEAD", "FAcodec SHA", check=False)
    env["facodec_upstream_revision"] = r.stdout.strip()
    r = run("cd /content/FAcodec && git rev-parse --abbrev-ref HEAD", "FAcodec branch", check=False)
    env["facodec_upstream_branch"] = r.stdout.strip()
    r = run("cd /content/FAcodec && git status --short", "FAcodec git status", check=False)
    env["facodec_upstream_git_status"] = r.stdout.strip()

    # ── Checkpoint info ──
    from modules.commons import recursive_munch
    from hf_utils import load_custom_model_from_hf

    ckpt_path, config_path = load_custom_model_from_hf("Plachta/FAcodec")
    env["facodec_ckpt_path"] = ckpt_path
    env["facodec_config_path"] = config_path

    # SHA-256 of checkpoint file (first 16 hex chars, matching adapter convention)
    ckpt_hash = hashlib.sha256(open(ckpt_path, "rb").read()).hexdigest()[:16]
    env["facodec_ckpt_hash"] = ckpt_hash
    env["facodec_ckpt_size_mb"] = round(os.path.getsize(ckpt_path) / 1024 / 1024, 1)

    # Config snapshot
    import yaml
    with open(config_path) as f:
        config = yaml.safe_load(f)
    env["facodec_model_params_keys"] = list(config.get("model_params", {}).keys())

    # ── Timestamps ──
    env["manifest_timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    env["elapsed_sec"] = round(time.time() - t_start, 1)

    # ── Save ──
    gate_dir = "/content/gate1_artifacts"
    os.makedirs(gate_dir, exist_ok=True)
    manifest_path = f"{gate_dir}/environment.json"
    with open(manifest_path, "w") as f:
        json.dump(env, f, indent=2)

    print(f"\n{'='*60}")
    print("ENVIRONMENT MANIFEST")
    print(f"{'='*60}")
    for k, v in env.items():
        print(f"  {k}: {v}")
    print(f"\nSaved to {manifest_path}")
    print("DONE")


if __name__ == "__main__":
    main()
