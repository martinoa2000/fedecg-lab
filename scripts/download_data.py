#!/usr/bin/env python
"""Download and extract the PTB-XL dataset from PhysioNet.

PTB-XL ships as a single ~1.8 GB archive containing both the 100 Hz and 500 Hz
waveforms. This project only needs 100 Hz, so by default the 500 Hz records are
skipped at extraction time, which saves roughly half the disk footprint.

Integrity is checked against `SHA256SUMS.txt`, which PhysioNet publishes inside
the archive itself. Every extracted file is hashed and compared, so a truncated
download or a half-finished extraction is caught here rather than surfacing as
an inexplicable NaN during training.

The script is idempotent and resumable: interrupt it and run it again.

Usage:
    python scripts/download_data.py
    python scripts/download_data.py --include-500hz
    python scripts/download_data.py --keep-archive
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

# Make `fedecg` importable when this script is run directly from a clone that
# has not been `pip install -e`'d yet.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fedecg.paths import PTBXL_DIR, RAW_DATA_DIR, ensure_dir

PTBXL_VERSION = "1.0.3"
ARCHIVE_URL = f"https://physionet.org/content/ptb-xl/get-zip/{PTBXL_VERSION}/"
ARCHIVE_NAME = f"ptb-xl-{PTBXL_VERSION}.zip"

CHUNK_SIZE = 1024 * 1024  # 1 MiB
SENTINEL_NAME = ".extracted"
"""Marker written after a successful, verified extraction."""


def human_bytes(num_bytes: float) -> str:
    """Format a byte count as a short human-readable string."""
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num_bytes) < 1024.0:
            return f"{num_bytes:3.1f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.1f} TB"


def report_count(index: int, total: int, noun: str) -> None:
    """Report progress through a count of items, quietly when not on a TTY."""
    if sys.stdout.isatty():
        if index % 500 == 0 or index == total:
            print(f"\r  {index} / {total} {noun}", end="", flush=True)
    elif index % 5000 == 0 or index == total:
        print(f"  {index} / {total} {noun}", flush=True)


def finish_count() -> None:
    """Terminate a `report_count` progress line."""
    if sys.stdout.isatty():
        print()


class ProgressPrinter:
    """Print download progress without flooding a non-interactive log.

    On a terminal this redraws a single line. When stdout is redirected to a
    file (CI, nohup) carriage returns do not overwrite anything, so progress is
    instead reported on its own line only every `LOG_STEP_BYTES`.
    """

    LOG_STEP_BYTES = 200 * 1024 * 1024

    def __init__(self, total: int) -> None:
        self.total = total
        self.is_tty = sys.stdout.isatty()
        self.next_log = self.LOG_STEP_BYTES

    def update(self, downloaded: int) -> None:
        """Report that `downloaded` bytes have been written so far."""
        if not self.total:
            return
        if self.is_tty:
            pct = 100.0 * downloaded / self.total
            print(
                f"\r  {human_bytes(downloaded)} / {human_bytes(self.total)} ({pct:5.1f}%)",
                end="",
                flush=True,
            )
        elif downloaded >= self.next_log:
            pct = 100.0 * downloaded / self.total
            print(
                f"  {human_bytes(downloaded)} / {human_bytes(self.total)} ({pct:.0f}%)", flush=True
            )
            self.next_log += self.LOG_STEP_BYTES

    def close(self) -> None:
        """Finish the progress line."""
        if self.is_tty:
            print()


def download_archive(destination: Path) -> Path:
    """Download the PTB-XL archive, resuming a partial file if one exists.

    Args:
        destination: Path the archive should end up at.

    Returns:
        The path to the completed archive.

    Raises:
        RuntimeError: If the server reports a size but the finished file does
            not match it.
    """
    part_file = destination.with_suffix(destination.suffix + ".part")

    with urllib.request.urlopen(ARCHIVE_URL, timeout=60) as response:
        total_size = int(response.headers.get("Content-Length", 0))

    if destination.is_file() and (total_size == 0 or destination.stat().st_size == total_size):
        print(
            f"Archive already downloaded: {destination} ({human_bytes(destination.stat().st_size)})"
        )
        return destination

    resume_from = part_file.stat().st_size if part_file.is_file() else 0
    if resume_from and total_size and resume_from >= total_size:
        # A stale .part that is somehow complete; restart to be safe.
        part_file.unlink()
        resume_from = 0

    request = urllib.request.Request(ARCHIVE_URL)
    if resume_from:
        request.add_header("Range", f"bytes={resume_from}-")
        print(f"Resuming download at {human_bytes(resume_from)} / {human_bytes(total_size)}")
    else:
        print(f"Downloading PTB-XL {PTBXL_VERSION} ({human_bytes(total_size)}) from PhysioNet")

    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            # 206 means the server honoured the Range header; 200 means it sent
            # the whole file and anything already downloaded must be discarded.
            mode = "ab" if response.status == 206 else "wb"
            if mode == "wb":
                resume_from = 0

            downloaded = resume_from
            progress = ProgressPrinter(total_size)
            with part_file.open(mode) as handle:
                while chunk := response.read(CHUNK_SIZE):
                    handle.write(chunk)
                    downloaded += len(chunk)
                    progress.update(downloaded)
            progress.close()
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"PhysioNet returned HTTP {error.code} for {ARCHIVE_URL}") from error

    if total_size and part_file.stat().st_size != total_size:
        raise RuntimeError(
            f"Incomplete download: got {part_file.stat().st_size} bytes, expected {total_size}. "
            "Re-run this script to resume."
        )

    part_file.replace(destination)
    return destination


def _wanted(member_name: str, *, include_500hz: bool) -> bool:
    """Return True if an archive member should be extracted."""
    if member_name.endswith("/"):
        return False
    if not include_500hz and "records500/" in member_name:
        return False
    return True


def extract_archive(archive: Path, target: Path, *, include_500hz: bool) -> None:
    """Extract the archive into `target`, stripping its top-level directory.

    The archive nests everything under a long versioned folder name. Stripping
    it keeps paths like `data/ptbxl/records100/00000/00001_lr.dat` stable even
    if the upstream folder name changes.
    """
    ensure_dir(target)
    with zipfile.ZipFile(archive) as zf:
        members = [m for m in zf.namelist() if _wanted(m, include_500hz=include_500hz)]
        total = len(members)
        print(f"Extracting {total} files to {target}")

        for index, member in enumerate(members, start=1):
            relative = Path(*Path(member).parts[1:])
            if not relative.parts:
                continue
            out_path = target / relative

            # Guard against path traversal via crafted archive entries.
            if not out_path.resolve().is_relative_to(target.resolve()):
                raise RuntimeError(f"Refusing to extract outside the target directory: {member}")

            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as source, out_path.open("wb") as sink:
                shutil.copyfileobj(source, sink, CHUNK_SIZE)

            report_count(index, total, "files")
        finish_count()


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def verify_extraction(target: Path, *, include_500hz: bool) -> int:
    """Verify extracted files against the checksums PhysioNet ships.

    Args:
        target: The extracted dataset root.
        include_500hz: Whether 500 Hz records were extracted. When False, their
            entries in the checksum file are expected to be absent and skipped.

    Returns:
        The number of files verified.

    Raises:
        RuntimeError: If a checksum mismatches or an expected file is missing.
    """
    sums_file = target / "SHA256SUMS.txt"
    if not sums_file.is_file():
        raise RuntimeError(f"Missing {sums_file}; cannot verify the download.")

    expected: dict[str, str] = {}
    for line in sums_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        digest, _, name = line.partition(" ")
        name = name.strip().lstrip("*")
        if name and (include_500hz or "records500/" not in name):
            expected[name] = digest

    print(f"Verifying {len(expected)} files against SHA256SUMS.txt")
    checked = 0
    mismatches: list[str] = []
    for name, digest in expected.items():
        path = target / name
        if not path.is_file():
            mismatches.append(f"missing: {name}")
        elif sha256_of(path) != digest:
            mismatches.append(f"checksum mismatch: {name}")

        checked += 1
        report_count(checked, len(expected), "files")
    finish_count()

    if mismatches:
        preview = "\n  ".join(mismatches[:10])
        more = f"\n  ... and {len(mismatches) - 10} more" if len(mismatches) > 10 else ""
        raise RuntimeError(f"Dataset verification failed:\n  {preview}{more}")

    return checked


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--include-500hz",
        action="store_true",
        help="Also extract the 500 Hz records (roughly doubles disk usage).",
    )
    parser.add_argument(
        "--keep-archive",
        action="store_true",
        help="Keep the downloaded .zip after extraction instead of deleting it.",
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip SHA-256 verification. Faster, but not recommended.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-extract even if a previous run already completed.",
    )
    return parser.parse_args()


def main() -> int:
    """Download, extract and verify PTB-XL. Returns a process exit code."""
    args = parse_args()
    sentinel = PTBXL_DIR / SENTINEL_NAME

    if sentinel.is_file() and not args.force:
        print(f"PTB-XL already present at {PTBXL_DIR} (use --force to re-extract).")
        return 0

    ensure_dir(RAW_DATA_DIR)
    archive = RAW_DATA_DIR / ARCHIVE_NAME

    try:
        download_archive(archive)
        extract_archive(archive, PTBXL_DIR, include_500hz=args.include_500hz)
        if args.skip_verify:
            print("Skipping verification (--skip-verify).")
        else:
            count = verify_extraction(PTBXL_DIR, include_500hz=args.include_500hz)
            print(f"Verified {count} files.")
    except (RuntimeError, OSError) as error:
        print(f"\nERROR: {error}", file=sys.stderr)
        return 1

    sentinel.write_text(
        f"ptb-xl {PTBXL_VERSION}\ninclude_500hz={args.include_500hz}\n", encoding="utf-8"
    )

    if not args.keep_archive:
        archive.unlink(missing_ok=True)
        print(f"Removed {archive} (pass --keep-archive to keep it).")

    print(f"\nDone. Dataset ready at {PTBXL_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
