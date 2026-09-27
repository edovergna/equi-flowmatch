"""Download the paper's datasets/checkpoints, validating OSF SHA-256 hashes."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def download(item):
    path = ROOT / item['path']
    if not path.exists() or path.stat().st_size != item['size']:
        partial = path.with_name(path.name + '.part')
        print(f"Downloading {item['path']} ({item['size'] / 1e6:.1f} MB)", flush=True)
        subprocess.run(['curl', '-fL', '--retry', '4', '-sS', '-C', '-',
                        item['url'], '-o', str(partial)], check=True)
        if partial.stat().st_size != item['size']:
            raise RuntimeError(f'Incorrect size: {partial}')
        partial.replace(path)
    if item['sha256']:
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        if digest != item['sha256']:
            raise RuntimeError(f'Checksum mismatch: {path}')
    print(f"Verified {item['path']}", flush=True)


if __name__ == '__main__':
    items = json.loads((ROOT / 'scripts/assets.json').read_text())
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(download, items))
