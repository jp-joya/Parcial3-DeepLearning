"""Carga señales temporales; nunca usa X_train.txt (561 características)."""
from __future__ import annotations
import hashlib
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path
import numpy as np

URL = 'https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip'
CHANNELS = [f'{sensor}_{axis}' for sensor in ('body_acc', 'body_gyro', 'total_acc') for axis in 'xyz']
LABELS = ['WALKING', 'WALKING_UPSTAIRS', 'WALKING_DOWNSTAIRS', 'SITTING', 'STANDING', 'LAYING']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def unpack(blob, destination):
    """Extrae solo el dataset esperado, sin aceptar rutas arbitrarias del ZIP."""
    with zipfile.ZipFile(io.BytesIO(blob)) as z:
        for name in z.namelist():
            p = Path(name)
            if name.endswith('.zip'):
                unpack(z.read(name), destination)
            elif 'UCI HAR Dataset' in p.parts and not name.endswith('/') and '__MACOSX' not in p.parts:
                parts = p.parts[p.parts.index('UCI HAR Dataset') + 1:]
                if not parts or any(x in ('.', '..') for x in parts):
                    raise ValueError(f'Ruta ZIP no válida: {name}')
                target = destination.joinpath(*parts)
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(z.read(name))


def download(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    archive = root / 'uci_har.zip'
    manifest = root / 'source.json'
    if not archive.exists():
        temp = archive.with_suffix('.part')
        for attempt in range(3):
            try:
                with urllib.request.urlopen(URL, timeout=120) as response, temp.open('wb') as f:
                    while chunk := response.read(1024 * 1024):
                        f.write(chunk)
                if not zipfile.is_zipfile(temp):
                    raise ValueError('La descarga no es un ZIP válido')
                temp.replace(archive)
                break
            except Exception:
                temp.unlink(missing_ok=True)
                if attempt == 2:
                    raise
                time.sleep(2)
    digest = sha256(archive)
    if manifest.exists() and json.loads(manifest.read_text())['sha256'] != digest:
        raise ValueError('El checksum del dataset cambió; revise la descarga antes de continuar')
    if not (root / 'raw' / 'train' / 'subject_train.txt').exists():
        unpack(archive.read_bytes(), root / 'raw')
    manifest.write_text(json.dumps({'url': URL, 'sha256': digest, 'channels': CHANNELS}, indent=2))
    return digest


def load_official(root, split):
    if split not in ('train', 'test'):
        raise ValueError(split)
    root = Path(root)
    cache = root / f'{split}_signals.npz'
    if cache.exists():
        with np.load(cache) as d:
            return d['x'], d['y'], d['subjects']
    base = root / 'raw' / split
    x = np.stack([np.loadtxt(base / 'Inertial Signals' / f'{c}_{split}.txt', dtype=np.float32)
                  for c in CHANNELS], axis=-1)
    y = np.loadtxt(base / f'y_{split}.txt', dtype=np.int64) - 1
    subjects = np.loadtxt(base / f'subject_{split}.txt', dtype=np.int64)
    assert x.shape == (len(y), 128, 9)
    assert np.isfinite(x).all() and set(np.unique(y)) == set(range(6))
    np.savez_compressed(cache, x=x, y=y, subjects=subjects)
    return x, y, subjects


def split_by_subject(x, y, subjects, dev_subjects):
    dev_subjects = set(dev_subjects)
    if not dev_subjects or not dev_subjects < set(np.unique(subjects)):
        raise ValueError('Los sujetos dev deben ser un subconjunto propio del train oficial')
    dev = np.isin(subjects, list(dev_subjects))
    train = ~dev
    assert not set(subjects[train]) & set(subjects[dev])
    for mask in (train, dev):
        if set(np.unique(y[mask])) != set(range(6)):
            raise ValueError('Cada partición debe contener las seis clases')
    return {name: (x[m], y[m], subjects[m], np.flatnonzero(m))
            for name, m in [('train', train), ('dev', dev)]}


def normalize_fit(x):
    mean = x.mean(axis=(0, 1), keepdims=True)
    std = x.std(axis=(0, 1), keepdims=True).clip(min=1e-6)
    return mean, std


def transform(x, window, mean, std):
    # Sufijo de la ventana original: no se inventan muestras ni se reconstruyen secuencias.
    if window not in (64, 128):
        raise ValueError('Ventanas permitidas: 64 o 128')
    return np.ascontiguousarray((x[:, -window:] - mean) / std, dtype=np.float32)


def summary_features(x):
    return np.concatenate([x.mean(1), x.std(1), x.min(1), x.max(1)], axis=1)
