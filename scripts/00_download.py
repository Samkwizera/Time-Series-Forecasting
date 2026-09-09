#!/usr/bin/env python
"""Stage 0: list, verify and (where permitted) download the 62 raw daily files.

Harvard Dataverse requires a guestbook response before it serves this dataset, so
anonymous scripted downloads are rejected. The workflow is therefore:

1. Open https://doi.org/10.7910/DVN/EGZHFV in a browser, fill the guestbook and
   download the files (or "Download all" as a zip) into ``data/raw``.
   On Colab, download into Google Drive and point ``paths.raw_dir`` at it.
2. Run ``python scripts/00_download.py --verify`` to check that every expected
   file is present and that file sizes match the Dataverse metadata (MD5 checks
   are run when ``--md5`` is given; this reads all 20.8 GB once).

If you hold a Dataverse API token (Account > API Token), pass ``--token`` and the
script downloads any missing files directly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from milan_forecast.config import load_config  # noqa: E402

DOI = "doi:10.7910/DVN/EGZHFV"
API = "https://dataverse.harvard.edu/api"


def fetch_manifest(cache: Path) -> list[dict]:
    """Return [{filename, id, filesize, md5}] for every file in the dataset."""
    if cache.exists():
        return json.loads(cache.read_text())
    url = f"{API}/datasets/:persistentId/?persistentId={DOI}"
    with urllib.request.urlopen(url, timeout=60) as resp:
        data = json.load(resp)
    files = []
    for f in data["data"]["latestVersion"]["files"]:
        df = f["dataFile"]
        files.append({"filename": df["filename"], "id": df["id"], "filesize": df["filesize"],
                      "md5": df.get("md5") or df.get("checksum", {}).get("value")})
    files.sort(key=lambda d: d["filename"])
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
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest = fetch_manifest(cfg.paths.tables_dir / "dataverse_manifest.json")
    if args.max_days:
        manifest = manifest[: args.max_days]
    total_gb = sum(m["filesize"] for m in manifest) / 1e9
    print(f"{len(manifest)} files, {total_gb:.2f} GB ({total_gb * 1e9 / 2**30:.2f} GiB) in the Dataverse record")

    missing, bad = [], []
    for entry in manifest:
        path = raw_dir / entry["filename"]
        if not path.exists():
            if args.token:
                print(f"downloading {entry['filename']} ({entry['filesize'] / 2**20:.0f} MB)")
                download(entry, path, args.token)
            else:
                missing.append(entry["filename"])
                continue
        if args.verify or args.md5:
            if path.stat().st_size != entry["filesize"]:
                bad.append((entry["filename"], "size"))
            elif args.md5 and entry["md5"] and md5sum(path) != entry["md5"]:
                bad.append((entry["filename"], "md5"))
    if missing:
        print(f"missing {len(missing)} files, e.g. {missing[:3]} -> download via the Dataverse UI")
    for name, why in bad:
        print(f"CORRUPT ({why}): {name}")
    if not missing and not bad:
        print("all files present" + (" and verified" if args.verify or args.md5 else ""))


if __name__ == "__main__":
    main()
