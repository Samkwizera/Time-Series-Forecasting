#!/usr/bin/env python
"""Stage 0: list, verify and (where permitted) download the 62 raw daily files.

Harvard Dataverse requires a guestbook response before it serves this dataset, so
anonymous scripted downloads are rejected. The workflow is therefore:

1. Open https://doi.org/10.7910/DVN/EGZHFV in a browser, fill the guestbook and
   download the files (or "Download all" as a zip) into ``data/raw``.
   On Colab, download into Google Drive and point ``paths.raw_dir`` at it.
2. Run ``python scripts/00_download.py --verify`` to check that every expected
   file is present. File sizes are compared when the Dataverse metadata is
   available; on Kaggle the bundled filename list is used so this does not need
   outbound access to dataverse.harvard.edu. MD5 checks run when ``--md5`` is given.

If you hold a Dataverse API token (Account > API Token), pass ``--token`` and the
script downloads any missing files directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milan_forecast.config import PROJECT_ROOT, load_config  # noqa: E402

DOI = "doi:10.7910/DVN/EGZHFV"
API = "https://dataverse.harvard.edu/api"
BUNDLED = PROJECT_ROOT / "config" / "dataverse_manifest.json"


def _parse_api_payload(data: dict) -> list[dict]:
    files = []
    for f in data["data"]["latestVersion"]["files"]:
        df = f["dataFile"]
        files.append({"filename": df["filename"], "id": df["id"], "filesize": df["filesize"],
                      "md5": df.get("md5") or df.get("checksum", {}).get("value")})
    files.sort(key=lambda d: d["filename"])
    return files


def fetch_manifest(cache: Path) -> list[dict]:
    """Return [{filename, ...}] for every file in the dataset.

    Prefers a local copy so ``--verify`` works on Kaggle, where Dataverse often
    returns HTTP 403. The live API is only tried when no local list exists.
    """
    for path in (cache, BUNDLED):
        if path.exists():
            return json.loads(path.read_text())
    url = f"{API}/datasets/:persistentId/?persistentId={DOI}"
    req = urllib.request.Request(url, headers={"User-Agent": "milan-forecast/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            files = _parse_api_payload(json.load(resp))
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"Dataverse API unavailable ({exc}); using the 62 expected filenames only")
        return json.loads(BUNDLED.read_text()) if BUNDLED.exists() else []
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(files, indent=1))
    return files


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def download(entry: dict, dest: Path, token: str) -> None:
    url = f"{API}/access/datafile/{entry['id']}?format=original"
    req = urllib.request.Request(url, headers={"X-Dataverse-key": token})
    tmp = dest.with_suffix(".part")
    with urllib.request.urlopen(req, timeout=120) as resp, open(tmp, "wb") as out:
        for block in iter(lambda: resp.read(1 << 22), b""):
            out.write(block)
    tmp.rename(dest)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--verify", action="store_true", help="check presence and sizes of files in raw_dir")
    parser.add_argument("--md5", action="store_true", help="also verify MD5 checksums (slow)")
    parser.add_argument("--token", default=None, help="Dataverse API token for direct downloads")
    parser.add_argument("--max-days", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    raw_dir = cfg.paths.raw_dir
    try:
        raw_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        # Kaggle's /kaggle/input is read-only; the files have to already be mounted there
        if not raw_dir.exists():
            print(f"cannot create {raw_dir} ({exc}). On Kaggle, attach the dataset first "
                  "(Add Input) and re-run the cell that sets RAW_DIR.")
            sys.exit(1)
    cache = cfg.paths.tables_dir / "dataverse_manifest.json"
    try:
        cfg.paths.tables_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        cache = BUNDLED
    manifest = fetch_manifest(cache)
    if not manifest:
        print("no file list available (bundled manifest missing and Dataverse API blocked)")
        sys.exit(1)
    if args.max_days:
        manifest = manifest[: args.max_days]
    sizes = [m.get("filesize") or 0 for m in manifest]
    if any(sizes):
        total_gb = sum(sizes) / 1e9
        print(f"{len(manifest)} files, {total_gb:.2f} GB ({total_gb * 1e9 / 2**30:.2f} GiB) in the Dataverse record")
    else:
        print(f"{len(manifest)} expected daily files (filename check only; Dataverse sizes not cached)")

    missing, bad = [], []
    existing = {p.name: p for p in raw_dir.rglob("sms-call-internet-mi-*.txt")} if raw_dir.exists() else {}
    for entry in manifest:
        path = existing.get(entry["filename"], raw_dir / entry["filename"])
        if not path.exists():
            if args.token and entry.get("id"):
                print(f"downloading {entry['filename']} ({(entry.get('filesize') or 0) / 2**20:.0f} MB)")
                download(entry, path, args.token)
            else:
                missing.append(entry["filename"])
                continue
        if args.verify or args.md5:
            expected = entry.get("filesize")
            if expected and path.stat().st_size != expected:
                bad.append((entry["filename"], "size"))
            elif args.md5 and entry.get("md5") and md5sum(path) != entry["md5"]:
                bad.append((entry["filename"], "md5"))
    print(f"found {len(existing)} sms-call-internet-mi-*.txt files under {raw_dir}")
    if missing:
        print(f"missing {len(missing)} files, e.g. {missing[:3]} -> attach a full 62-day dataset or download via Dataverse")
        sys.exit(1)
    for name, why in bad:
        print(f"CORRUPT ({why}): {name}")
    if bad:
        sys.exit(1)
    print("all files present" + (" and verified" if args.verify or args.md5 else ""))


if __name__ == "__main__":
    main()
