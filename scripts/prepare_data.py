"""Prepare downloaded alanine and split LJ55 data for the notebooks."""
import hashlib
from pathlib import Path
import tarfile
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
data = ROOT / 'data'
archive = data / 'AImplicitUnconstrained.tgz'
if not (data / 'AImplicitUnconstrained/traj0.h5').exists():
    with archive.open('rb') as stream:
        assert hashlib.file_digest(stream, 'md5').hexdigest() == 'f18b9a9c06f3590f1632ca99161c6553'
    with tarfile.open(archive) as stream:
        stream.extractall(data, filter='data')
    print('Extracted alanine trajectory', flush=True)

output = data / 'all_data_LJ55-1000.npy'
if not output.exists():
    parts = [np.load(data / f'all_data_LJ55-1000-part{i}.npy', mmap_mode='r') for i in (1, 2)]
    assert parts[0].shape[1:] == parts[1].shape[1:]
    assert parts[0].dtype == parts[1].dtype
    shape = (sum(len(part) for part in parts), *parts[0].shape[1:])
    partial = output.with_suffix('.npy.part')
    combined = np.lib.format.open_memmap(partial, mode='w+', dtype=parts[0].dtype, shape=shape)
    offset = 0
    for part in parts:
        for start in range(0, len(part), 10000):
            combined[offset + start:offset + min(start + 10000, len(part))] = part[start:start + 10000]
        offset += len(part)
    combined.flush()
    del combined
    partial.replace(output)
    print(f'Combined LJ55 parts: {shape}', flush=True)
