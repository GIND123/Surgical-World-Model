"""Hugging Face Hub helpers. The token is read from the Git-ignored .env (key ``hf`` or ``HF_TOKEN``) and is
never printed or logged. Weights go to one public model repo, one sub-folder per checkpoint."""

from __future__ import annotations

from pathlib import Path

from swm.paths import ROOT

MODEL_REPO = "surgical-world-model"


def _env() -> dict[str, str]:
    out = {}
    for line in (ROOT / ".env").read_text().splitlines():
        if "=" in line and not line.lstrip().startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
    return out


def token() -> str:
    env = _env()
    return env.get("HF_TOKEN") or env["hf"]


def user() -> str:
    return _env().get("HF_USER", "GOVINDFROM")


def repo_id(name: str = MODEL_REPO) -> str:
    return f"{user()}/{name}"


def push_folder(local_dir: Path, path_in_repo: str | None, message: str, name: str = MODEL_REPO,
                private: bool = False) -> str:
    """Create the (public) model repo if needed and upload ``local_dir`` under ``path_in_repo``."""
    from huggingface_hub import HfApi

    api = HfApi(token=token())
    rid = repo_id(name)
    api.create_repo(rid, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(repo_id=rid, folder_path=str(local_dir), path_in_repo=path_in_repo, commit_message=message)
    return f"https://huggingface.co/{rid}"


def push_file(local_file: Path, path_in_repo: str, message: str, name: str = MODEL_REPO) -> str:
    from huggingface_hub import HfApi

    api = HfApi(token=token())
    rid = repo_id(name)
    api.create_repo(rid, repo_type="model", private=False, exist_ok=True)
    api.upload_file(repo_id=rid, path_or_fileobj=str(local_file), path_in_repo=path_in_repo, commit_message=message)
    return f"https://huggingface.co/{rid}/blob/main/{path_in_repo}"
