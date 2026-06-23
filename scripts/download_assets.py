"""Download datasets and SSL checkpoints for the ASR project.

The default network mode is direct: proxy environment variables are ignored
unless --use-env-proxy is passed explicitly.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import requests
from huggingface_hub import snapshot_download


OPENSLR_MIRRORS = (
    "https://www.openslr.org/resources/12",
    "https://openslr.trmal.net/resources/12",
    "https://openslr.elda.org/resources/12",
)

PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


@dataclass(frozen=True)
class DataArchive:
    name: str
    filename: str
    urls: tuple[str, ...]
    extracted_path: Path
    md5: str | None = None
    size_bytes: int | None = None


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    local_name: str
    revision: str
    allow_patterns: tuple[str, ...]


DATA_ARCHIVES = (
    DataArchive(
        name="librispeech_finetuning",
        filename="librispeech_finetuning.tgz",
        urls=("https://dl.fbaipublicfiles.com/librilight/data/librispeech_finetuning.tgz",),
        extracted_path=Path("librispeech_finetuning"),
        size_bytes=597_601_132,
    ),
    DataArchive(
        name="dev-clean",
        filename="dev-clean.tar.gz",
        urls=tuple(f"{mirror}/dev-clean.tar.gz" for mirror in OPENSLR_MIRRORS),
        extracted_path=Path("LibriSpeech") / "dev-clean",
        md5="42e2234ba48799c1f50f24a7926300a1",
    ),
    DataArchive(
        name="test-clean",
        filename="test-clean.tar.gz",
        urls=tuple(f"{mirror}/test-clean.tar.gz" for mirror in OPENSLR_MIRRORS),
        extracted_path=Path("LibriSpeech") / "test-clean",
        md5="32fa31d27d2e1cad72775fee3f4849a9",
    ),
    DataArchive(
        name="test-other",
        filename="test-other.tar.gz",
        urls=tuple(f"{mirror}/test-other.tar.gz" for mirror in OPENSLR_MIRRORS),
        extracted_path=Path("LibriSpeech") / "test-other",
        md5="fb5a50374b501bb3bac4815ee91d3135",
    ),
    DataArchive(
        name="train-clean-100",
        filename="train-clean-100.tar.gz",
        urls=tuple(f"{mirror}/train-clean-100.tar.gz" for mirror in OPENSLR_MIRRORS),
        extracted_path=Path("LibriSpeech") / "train-clean-100",
        md5="2a93770f6d5c6c964bc36631d331a522",
    ),
)

MODEL_SPECS = (
    ModelSpec(
        model_id="microsoft/wavlm-base-plus",
        local_name="wavlm-base-plus",
        revision="4c66d4806a428f2e922ccfa1a962776e232d487b",
        allow_patterns=(
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.bin",
        ),
    ),
    ModelSpec(
        model_id="facebook/hubert-base-ls960",
        local_name="hubert-base-ls960",
        revision="dba3bb02fda4248b6e082697eee756de8fe8aa8a",
        allow_patterns=(
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.bin",
        ),
    ),
    ModelSpec(
        model_id="facebook/wav2vec2-base",
        local_name="wav2vec2-base",
        revision="0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8",
        allow_patterns=(
            "config.json",
            "preprocessor_config.json",
            "pytorch_model.bin",
        ),
    ),
)


def build_proxies(force_direct: bool) -> dict[str, str] | None:
    """Return requests-compatible proxy config."""
    return {} if force_direct else None


@contextlib.contextmanager
def proxy_environment(allow_env_proxy: bool) -> Iterator[None]:
    if allow_env_proxy:
        yield
        return

    saved = {name: os.environ.get(name) for name in PROXY_ENV_VARS}
    for name in PROXY_ENV_VARS:
        os.environ.pop(name, None)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def create_session(force_direct: bool) -> requests.Session:
    session = requests.Session()
    session.trust_env = not force_direct
    return session


def md5_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_archive_file(path: Path, asset: DataArchive) -> bool:
    if not path.exists():
        return False
    if asset.size_bytes is not None and path.stat().st_size != asset.size_bytes:
        return False
    if asset.md5 is not None and md5_file(path) != asset.md5:
        return False
    try:
        with tarfile.open(path, "r:*") as archive:
            archive.next()
    except tarfile.TarError:
        return False
    return True


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()

    for member in archive.getmembers():
        target = (destination / member.name).resolve()
        try:
            target.relative_to(destination_root)
        except ValueError as exc:
            raise ValueError(f"Unsafe archive member: {member.name}") from exc

    archive.extractall(destination)


def download_file(
    urls: Sequence[str],
    output_path: Path,
    *,
    force: bool,
    force_direct: bool,
    timeout: int,
    chunk_size: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = output_path.with_suffix(output_path.suffix + ".part")

    if force:
        output_path.unlink(missing_ok=True)
        part_path.unlink(missing_ok=True)

    last_error: Exception | None = None
    for url in urls:
        try:
            resume_at = part_path.stat().st_size if part_path.exists() else 0
            headers = {"Range": f"bytes={resume_at}-"} if resume_at else {}
            mode = "ab" if resume_at else "wb"
            session = create_session(force_direct=force_direct)
            response = session.get(
                url,
                stream=True,
                headers=headers,
                timeout=timeout,
                proxies=build_proxies(force_direct=force_direct),
            )
            if resume_at and response.status_code != 206:
                response.close()
                resume_at = 0
                mode = "wb"
                response = session.get(
                    url,
                    stream=True,
                    timeout=timeout,
                    proxies=build_proxies(force_direct=force_direct),
                )
            response.raise_for_status()
            with part_path.open(mode + "b" if "b" not in mode else mode) as handle:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        handle.write(chunk)
            part_path.replace(output_path)
            print(f"[data] downloaded {output_path.name} from {url}")
            return
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"[data] failed {url}: {exc}", file=sys.stderr)

    manual = "\n".join(f"  - {url}" for url in urls)
    raise RuntimeError(
        f"Could not download {output_path.name}. Try one of these URLs manually:\n{manual}\n"
        f"Place the file at: {output_path}"
    ) from last_error


def download_archive(
    asset: DataArchive,
    downloads_dir: Path,
    *,
    force: bool,
    force_direct: bool,
    timeout: int,
    chunk_size: int,
) -> Path:
    archive_path = downloads_dir / asset.filename
    if not force and validate_archive_file(archive_path, asset):
        print(f"[data] reuse {archive_path}")
        return archive_path

    download_file(
        asset.urls,
        archive_path,
        force=force,
        force_direct=force_direct,
        timeout=timeout,
        chunk_size=chunk_size,
    )
    if not validate_archive_file(archive_path, asset):
        raise RuntimeError(f"Downloaded file failed validation: {archive_path}")
    return archive_path


def extract_archive(asset: DataArchive, archive_path: Path, raw_dir: Path, *, force: bool) -> None:
    extracted_path = raw_dir / asset.extracted_path
    if extracted_path.exists() and not force:
        print(f"[data] reuse extracted {extracted_path}")
        return

    print(f"[data] extracting {archive_path.name} to {raw_dir}")
    with tarfile.open(archive_path, "r:*") as archive:
        safe_extract(archive, raw_dir)
    if not extracted_path.exists():
        raise RuntimeError(f"Expected extracted path was not created: {extracted_path}")


def download_model(
    model: ModelSpec,
    models_dir: Path,
    *,
    force: bool,
    allow_env_proxy: bool,
    workers: int,
) -> Path:
    local_dir = models_dir / model.local_name
    with proxy_environment(allow_env_proxy):
        snapshot_download(
            repo_id=model.model_id,
            revision=model.revision,
            local_dir=local_dir,
            allow_patterns=list(model.allow_patterns),
            force_download=force,
            max_workers=workers,
        )

    missing = [pattern for pattern in model.allow_patterns if not (local_dir / pattern).exists()]
    if missing:
        raise RuntimeError(f"Missing model files for {model.model_id}: {', '.join(missing)}")
    print(f"[model] ready {model.model_id} -> {local_dir}")
    return local_dir


def select_by_name(items: Iterable, selected: Sequence[str] | None):
    if not selected:
        return tuple(items)
    selected_set = set(selected)
    named_items = tuple((item, item_key(item)) for item in items)
    result = tuple(item for item, key in named_items if key in selected_set)
    found = {key for _, key in named_items if key in selected_set}
    missing = selected_set - found
    if missing:
        raise ValueError(f"Unknown selection: {', '.join(sorted(missing))}")
    return result


def item_key(item) -> str:
    if hasattr(item, "name"):
        return item.name
    if hasattr(item, "local_name"):
        return item.local_name
    raise TypeError(f"Unsupported selectable item: {item!r}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    data_names = [asset.name for asset in DATA_ARCHIVES]
    model_names = [model.local_name for model in MODEL_SPECS]

    parser = argparse.ArgumentParser(
        description="Download ASR datasets and frozen SSL checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output-root", type=Path, default=Path("."), help="Project root.")
    parser.add_argument("--data-only", action="store_true", help="Download datasets only.")
    parser.add_argument("--models-only", action="store_true", help="Download checkpoints only.")
    parser.add_argument("--data", nargs="+", choices=data_names, help="Subset of data archives.")
    parser.add_argument("--models", nargs="+", choices=model_names, help="Subset of checkpoints.")
    parser.add_argument("--skip-extract", action="store_true", help="Download archives without extracting.")
    parser.add_argument("--force", action="store_true", help="Re-download files and refresh checkpoints.")
    parser.add_argument(
        "--use-env-proxy",
        action="store_true",
        help="Respect standard HTTP(S) proxy environment variables.",
    )
    parser.add_argument("--timeout", type=int, default=60, help="HTTP timeout in seconds.")
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024, help="Download chunk size.")
    parser.add_argument("--workers", type=int, default=4, help="Parallel workers for checkpoint files.")
    args = parser.parse_args(argv)

    if args.data_only and args.models_only:
        parser.error("--data-only and --models-only cannot be used together")
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    root = args.output_root.resolve()
    downloads_dir = root / "data" / "downloads"
    raw_dir = root / "data" / "raw"
    models_dir = root / "models"
    force_direct = not args.use_env_proxy

    if not args.models_only:
        for asset in select_by_name(DATA_ARCHIVES, args.data):
            with proxy_environment(args.use_env_proxy):
                archive_path = download_archive(
                    asset,
                    downloads_dir,
                    force=args.force,
                    force_direct=force_direct,
                    timeout=args.timeout,
                    chunk_size=args.chunk_size,
                )
            if not args.skip_extract:
                extract_archive(asset, archive_path, raw_dir, force=args.force)

    if not args.data_only:
        models_dir.mkdir(parents=True, exist_ok=True)
        for model in select_by_name(MODEL_SPECS, args.models):
            download_model(
                model,
                models_dir,
                force=args.force,
                allow_env_proxy=args.use_env_proxy,
                workers=args.workers,
            )

    print("[done] assets are ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
