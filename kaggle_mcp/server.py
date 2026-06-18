"""
Kaggle MCP Server — PRISM GPU 실험 자동화.

PRISM 코드를 base64 tarball로 커널에 직접 임베딩.
별도 데이터셋 업로드 불필요.

Tools:
  kaggle_run     : Kaggle T4 GPU에서 실험 스크립트 실행
  kaggle_status  : 실행 상태 확인
  kaggle_logs    : 실행 로그 조회
  kaggle_output  : 결과 파일 다운로드
"""

import base64
import io
import json
import os
import tarfile
import tempfile
import textwrap
import time
from pathlib import Path

import kaggle
from mcp.server.fastmcp import FastMCP

# ------------------------------------------------------------------ #
# 설정                                                                 #
# ------------------------------------------------------------------ #

PRISM_DIR    = Path(__file__).parent.parent  # /home/user/PRISM
USERNAME     = "sion1210"
KERNEL_SLUG  = "prism-experiment"
KERNEL_ID    = f"{USERNAME}/{KERNEL_SLUG}"

EXCLUDE_DIRS  = {".git", "__pycache__", ".claude", "kaggle_mcp", ".ipynb_checkpoints"}
EXCLUDE_FILES = {".gitignore", "best_prism.pt", "energy_convergence.png"}
EXCLUDE_EXTS  = {".pyc", ".pyo", ".pt", ".png", ".jpg"}

mcp = FastMCP("kaggle-prism")


# ------------------------------------------------------------------ #
# 헬퍼                                                                 #
# ------------------------------------------------------------------ #

def _api() -> kaggle.KaggleApi:
    api = kaggle.KaggleApi()
    api.authenticate()
    return api


def _prism_tarball_b64() -> str:
    """PRISM 소스를 gzip tarball → base64 인코딩."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for src in sorted(PRISM_DIR.rglob("*")):
            if src.is_dir():
                continue
            rel = src.relative_to(PRISM_DIR)
            if rel.parts[0] in EXCLUDE_DIRS:
                continue
            if set(rel.parts[:-1]) & EXCLUDE_DIRS:
                continue
            if src.name in EXCLUDE_FILES:
                continue
            if src.suffix in EXCLUDE_EXTS:
                continue
            tar.add(src, arcname=str(rel))
    return base64.b64encode(buf.getvalue()).decode()


def _build_runner(script_name: str, extra_args: str) -> str:
    """Kaggle 커널 스크립트 생성 (PRISM tarball 내장)."""
    b64 = _prism_tarball_b64()
    args_list = json.dumps(extra_args.split()) if extra_args.strip() else "[]"
    return textwrap.dedent(f"""\
        import base64, io, tarfile, os, sys, subprocess

        # PRISM 코드 추출
        dst = '/kaggle/working/prism'
        os.makedirs(dst, exist_ok=True)
        data = base64.b64decode("{b64}")
        with tarfile.open(fileobj=io.BytesIO(data), mode='r:gz') as t:
            t.extractall(dst)
        os.chdir(dst)
        sys.path.insert(0, dst)

        import torch
        print(f"PyTorch {{torch.__version__}}, CUDA={{torch.cuda.is_available()}}")
        if torch.cuda.is_available():
            print(f"GPU: {{torch.cuda.get_device_name(0)}}")

        args = {args_list} + ['--device', 'cuda']
        cmd  = [sys.executable, '-u', '{script_name}'] + args
        print(f"\\n>>> {{' '.join(cmd)}}\\n", flush=True)

        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        with open('/kaggle/working/output.txt', 'w') as f:
            for line in proc.stdout:
                print(line, end='', flush=True)
                f.write(line); f.flush()
        proc.wait()
        print(f"\\n>>> Exit code: {{proc.returncode}}")
    """)


# ------------------------------------------------------------------ #
# MCP Tools                                                            #
# ------------------------------------------------------------------ #

@mcp.tool()
def kaggle_run(script_name: str, extra_args: str = "") -> str:
    """
    Kaggle T4 GPU에서 PRISM 실험 스크립트를 실행.
    현재 코드를 커널에 직접 임베딩하므로 코드 변경 후 바로 반영됨.

    Args:
        script_name : 실행할 스크립트 (예: 'stage14b_remaining.py')
        extra_args  : 추가 인자 문자열 (예: '--epochs 7 --skip_mamba')

    Returns:
        시작 확인. 이후 kaggle_status() / kaggle_logs() 로 진행 확인.
    """
    api = _api()
    runner = _build_runner(script_name, extra_args)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        (tmp_path / "run.py").write_text(runner)
        meta = {
            "id": KERNEL_ID,
            "title": "PRISM Experiment",
            "code_file": "run.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": True,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": False,
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
        }
        (tmp_path / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
        api.kernels_push(tmp)

    return (f"✓ Submitted to Kaggle GPU\n"
            f"  kernel : {KERNEL_ID}\n"
            f"  script : {script_name}\n"
            f"  args   : {extra_args or '(none)'}\n"
            f"  check  : kaggle_status() / kaggle_logs()")


@mcp.tool()
def kaggle_status() -> str:
    """PRISM 실험 커널의 현재 실행 상태를 반환."""
    api = _api()
    try:
        s = api.kernels_status(KERNEL_ID)
        status = getattr(s, 'status', str(s))
        err = getattr(s, 'failureMessage', None)
        return f"Status: {status}" + (f"\nError: {err}" if err else "")
    except Exception as e:
        return f"Error: {e}"


@mcp.tool()
def kaggle_logs(tail: int = 150) -> str:
    """
    실험 로그 (stdout) 마지막 N줄 반환. 실행 중에도 호출 가능.

    Args:
        tail: 반환할 마지막 줄 수 (기본 150)
    """
    api = _api()
    with tempfile.TemporaryDirectory() as tmp:
        try:
            api.kernels_output(KERNEL_ID, path=tmp, quiet=True)
            out = Path(tmp) / "output.txt"
            if out.exists():
                lines = out.read_text().splitlines()
                return "\n".join(lines[-tail:])
            return "No output yet — kernel may still be initializing"
        except Exception as e:
            return f"Log error: {e}"


@mcp.tool()
def kaggle_output(local_path: str = "/tmp/kaggle_output") -> str:
    """
    완료된 실험의 결과 파일을 로컬에 다운로드.

    Args:
        local_path: 저장 경로 (기본 /tmp/kaggle_output)
    """
    api = _api()
    out_dir = Path(local_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        api.kernels_output(KERNEL_ID, path=str(out_dir), quiet=True)
        files = list(out_dir.iterdir())
        result = f"Downloaded {len(files)} file(s) → {out_dir}\n"
        result += "\n".join(f"  {f.name} ({f.stat().st_size:,} B)" for f in files)
        out = out_dir / "output.txt"
        if out.exists():
            result += "\n\n--- output.txt ---\n" + out.read_text()
        return result
    except Exception as e:
        return f"Download error: {e}"


# ------------------------------------------------------------------ #
# 진입점                                                               #
# ------------------------------------------------------------------ #

if __name__ == "__main__":
    mcp.run(transport="stdio")
