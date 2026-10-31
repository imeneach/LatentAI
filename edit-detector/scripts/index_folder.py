"""Bulk-index a folder of licensed originals:  python -m scripts.index_folder ./photos "unsplash" """
import hashlib
import shutil
import sys
from pathlib import Path

from app import config
from app.imageio import decode
from app.main import _embedder, index

EXT = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"}


def main(folder: str, source: str = ""):
    n = 0
    for p in sorted(Path(folder).rglob("*")):
        if p.suffix.lower() not in EXT:
            continue
        data = p.read_bytes()
        key = hashlib.sha256(data).hexdigest()[:32]
        if index.has(key):
            continue
        try:
            img = decode(data)
        except ValueError:
            continue
        dest = config.ORIGINALS_DIR / f"{key}.jpg"
        shutil.copyfile(p, dest)
        index.add(_embedder.embed(img), {"id": key, "path": str(dest), "source": source or str(p)})
        n += 1
    print(f"indexed {n} new images; total {len(index)}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
