from __future__ import annotations
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset
from .models import HARModel
from .data import normalize_fit, transform


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False))
    tmp.replace(path)


def save_torch(path, obj):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    torch.save(obj, temp)
    temp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def loader(x, y, batch_size, shuffle=False, generator=None):
    return DataLoader(TensorDataset(torch.from_numpy(x), torch.from_numpy(y)),
                      batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0)


def metrics(y, pred):
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=list(range(6)), average='macro', zero_division=0))}


@torch.no_grad()
def evaluate(model, batches, device):
    model.eval()
    ys, preds, probs = [], [], []
    total_loss = 0.0
    for x, y in batches:
        logits = model(x.to(device))
        total_loss += torch.nn.functional.cross_entropy(logits, y.to(device), reduction='sum').item()
        ys.append(y.numpy())
        probs.append(logits.softmax(1).cpu().numpy())
        preds.append(logits.argmax(1).cpu().numpy())
    y = np.concatenate(ys)
    pred = np.concatenate(preds)
    return {**metrics(y, pred), 'loss': total_loss / len(y)}, pred, np.concatenate(probs)


def fit(run_dir, architecture, hp, seed, cfg, parts, device, deadline):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / 'result.json').exists():
        return json.loads((run_dir / 'result.json').read_text())
    if time.monotonic() >= deadline:
        return None
    seed_all(seed)
    xtr, ytr, _, _ = parts['train']
    xdev, ydev, _, _ = parts['dev']
    # Se ajusta una normalización por longitud, exclusivamente sobre train.
    mean, std = normalize_fit(xtr[:, -hp['window']:])
    xtr = transform(xtr, hp['window'], mean, std)
    xdev = transform(xdev, hp['window'], mean, std)
    generator = torch.Generator().manual_seed(seed)
    train_loader = loader(xtr, ytr, cfg['batch_size'], True, generator)
    train_eval = loader(xtr, ytr, cfg['batch_size'])
    dev_loader = loader(xdev, ydev, cfg['batch_size'])
    model = HARModel(architecture, **hp).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp['lr'], weight_decay=hp['weight_decay'])
    history, start_epoch, best_score, bad_epochs, elapsed = [], 0, -1.0, 0, 0.0
    if (run_dir / 'last.pt').exists():
        state = torch.load(run_dir / 'last.pt', map_location='cpu', weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        # torch mueve el estado de Adam al dispositivo del parámetro durante load_state_dict.
        history = state['history']
        start_epoch, best_score, bad_epochs, elapsed = state['epoch'], state['best_score'], state['bad_epochs'], state['elapsed']
        random.setstate(state['random_state'])
        np.random.set_state(state['numpy_state'])
        torch.set_rng_state(state['torch_state'])
        generator.set_state(state['generator_state'])
        if device.type == 'cuda' and state.get('cuda_state') is not None:
            torch.cuda.set_rng_state_all(state['cuda_state'])
    atomic_json(run_dir / 'spec.json', {'architecture': architecture, 'hp': hp, 'seed': seed})
    for epoch in range(start_epoch, cfg['max_epochs']):
        if bad_epochs >= cfg['patience']:
            break
        if time.monotonic() >= deadline:
            return None
        tick = time.monotonic()
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(x.to(device)), y.to(device))
            if not torch.isfinite(loss):
                raise ValueError('Pérdida no finita')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        train_m, _, _ = evaluate(model, train_eval, device)
        dev_m, _, _ = evaluate(model, dev_loader, device)
        elapsed += time.monotonic() - tick
        history.append({'epoch': epoch + 1, **{f'train_{k}': v for k, v in train_m.items()},
                        **{f'dev_{k}': v for k, v in dev_m.items()}})
        if dev_m['macro_f1'] > best_score + cfg['min_delta']:
            best_score, bad_epochs = dev_m['macro_f1'], 0
            save_torch(run_dir / 'best.pt', {'model': model.state_dict(), 'mean': mean, 'std': std,
                                           'architecture': architecture, 'hp': hp, 'seed': seed,
                                           'epoch': epoch + 1})
        else:
            bad_epochs += 1
        save_torch(run_dir / 'last.pt', {
            'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'history': history,
            'epoch': epoch + 1, 'best_score': best_score, 'bad_epochs': bad_epochs, 'elapsed': elapsed,
            'random_state': random.getstate(), 'numpy_state': np.random.get_state(),
            'torch_state': torch.get_rng_state(), 'generator_state': generator.get_state(),
            'cuda_state': torch.cuda.get_rng_state_all() if device.type == 'cuda' else None})
        atomic_json(run_dir / 'history.json', history)
        print(f'{run_dir.name} epoch={epoch+1:02d} train_F1={train_m["macro_f1"]:.4f} dev_F1={dev_m["macro_f1"]:.4f}', flush=True)
    state = torch.load(run_dir / 'best.pt', map_location='cpu', weights_only=False)
    model.load_state_dict(state['model'])
    train_m, _, _ = evaluate(model, train_eval, device)
    dev_m, pred, probs = evaluate(model, dev_loader, device)
    np.savez_compressed(run_dir / 'dev_predictions.npz', y=ydev, pred=pred, probabilities=probs,
                        subjects=parts['dev'][2], indices=parts['dev'][3])
    result = {'architecture': architecture, 'hp': hp, 'seed': seed, 'train': train_m, 'dev': dev_m,
              'best_epoch': state['epoch'], 'epochs': len(history), 'seconds': elapsed,
              'parameters': sum(p.numel() for p in model.parameters())}
    atomic_json(run_dir / 'result.json', result)
    # Last sirve para reanudar un entrenamiento incompleto; best es el artefacto final.
    (run_dir / 'last.pt').unlink(missing_ok=True)
    return result
