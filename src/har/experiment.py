"""CLI: preparar, entrenar, reanudar, resumir y generar artefactos del informe."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
import itertools
import json
import os
import platform
import pickle
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import sklearn
import torch
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .data import download, load_official, split_by_subject, summary_features, transform, sha256
from .models import HARModel
from .train import atomic_json, fit, loader, evaluate, metrics


def read_json(path):
    return json.loads(Path(path).read_text())


def configurations(cfg):
    keys = list(cfg['space'])
    grid = [dict(zip(keys, values)) for values in itertools.product(*(cfg['space'][k] for k in keys))]
    rng = np.random.default_rng(cfg['seed'])
    rng.shuffle(grid)
    anchor = {'window': 128, 'hidden': 64, 'dropout': 0.25, 'lr': 0.001, 'weight_decay': 0.0001}
    if anchor in grid:
        grid.remove(anchor)
        grid.insert(0, anchor)
    if not 1 <= cfg['trials'] <= len(grid):
        raise ValueError('Cantidad de trials fuera del espacio de búsqueda')
    return [{**hp, **cfg['fixed']} for hp in grid[:cfg['trials']]]


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def prepare_parts(data_dir, cfg):
    x, y, subjects = load_official(data_dir, 'train')
    parts = split_by_subject(x, y, subjects, cfg['dev_subjects'])
    limit = cfg.get('debug_limit_per_split')
    if limit:
        for split, (sx, sy, ss, indices) in parts.items():
            rng = np.random.default_rng(cfg['seed'])
            selected = np.concatenate([rng.permutation(np.flatnonzero(sy == c))[:max(1, limit // 6)] for c in range(6)])
            parts[split] = (sx[selected], sy[selected], ss[selected], indices[selected])
    return parts


def validate_config(cfg):
    if cfg['architectures'] != ['rnn', 'lstm', 'cnn_lstm']:
        raise ValueError('Se requieren las tres arquitecturas en orden rnn, lstm, cnn_lstm')
    if not 0 < cfg['search_fraction'] < 1:
        raise ValueError('search_fraction debe estar entre 0 y 1')
    if cfg['max_epochs'] < 1 or cfg['patience'] < 1 or cfg['batch_size'] < 1:
        raise ValueError('Épocas, paciencia y batch deben ser positivos')
    if not cfg['final_seeds'] or len(set(cfg['final_seeds'])) != len(cfg['final_seeds']):
        raise ValueError('Se requieren semillas finales distintas')
    configurations(cfg)


def identity(cfg, digest):
    code = hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob('*.py')):
        code.update(p.name.encode())
        code.update(p.read_bytes())
    return {'config': cfg, 'dataset_sha256': digest, 'source_sha256': code.hexdigest(),
            'torch': torch.__version__, 'numpy': np.__version__, 'sklearn': sklearn.__version__,
            'python': platform.python_version()}


@contextlib.contextmanager
def exclusive_run(out):
    # Linux: el kernel libera el lock incluso tras una interrupción abrupta.
    import fcntl
    out.mkdir(parents=True, exist_ok=True)
    with (out / '.lock').open('w') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Ya hay otro proceso escribiendo en este experimento') from exc
        yield


def run(args):
    cfg = read_json(args.config)
    validate_config(cfg)
    if args.hours <= 0:
        raise ValueError('--hours debe ser positivo')
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.set_num_threads(cfg['threads'])
    device = get_device()
    out, data_dir = Path(args.out), Path(args.data)
    digest = download(data_dir)
    with exclusive_run(out):
        current_identity = identity(cfg, digest)
        if (out / 'manifest.json').exists():
            old = read_json(out / 'manifest.json')
            if old['identity'] != current_identity or old['device'] != str(device):
                raise ValueError('Cambió código/configuración/datos/entorno/dispositivo. Use otro --out para no mezclar experimentos.')
        else:
            atomic_json(out / 'manifest.json', {'identity': current_identity, 'device': str(device),
                'created_utc': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
                'debug': bool(cfg.get('debug_limit_per_split')), 'metric': 'macro_f1',
                'note': 'Determinismo solicitado; algunas operaciones avisan si no tienen alternativa determinista.'})
        deadline = time.monotonic() + args.hours * 3600
        search_deadline = time.monotonic() + args.hours * 3600 * cfg['search_fraction']
        parts = prepare_parts(data_dir, cfg)
        official_test_subjects = np.loadtxt(data_dir / 'raw/test/subject_test.txt', dtype=int)
        train_subjects, dev_subjects = set(parts['train'][2]), set(parts['dev'][2])
        assert not (train_subjects | dev_subjects) & set(official_test_subjects)
        atomic_json(out / 'split.json', {name: {'n': len(p[1]), 'subjects': sorted(map(int, set(p[2]))),
                    'class_counts': np.bincount(p[1], minlength=6).tolist()} for name, p in parts.items()})
        np.savez_compressed(out / 'split_indices.npz', train=parts['train'][3], dev=parts['dev'][3])
        fit_baselines(out, cfg, parts)
        configs = configurations(cfg)
        atomic_json(out / 'search_plan.json', configs)
        print(f'Dispositivo: {device} | train={len(parts["train"][1])} dev={len(parts["dev"][1])} | '
              f'{len(configs)} configuraciones × 3 arquitecturas', flush=True)
        selection_path = out / 'selection.json'
        if not selection_path.exists():
            stop = False
            for i, hp in enumerate(configs):
                # Rotar el orden evita favorecer sistemáticamente a una arquitectura al agotarse el tiempo.
                architectures = cfg['architectures'][i % 3:] + cfg['architectures'][:i % 3]
                for architecture in architectures:
                    trial_dir = out / 'search' / f'{i:03d}_{architecture}'
                    try:
                        result = fit(trial_dir, architecture, hp, cfg['seed'], cfg, parts, device, search_deadline)
                    except Exception:
                        atomic_json(trial_dir / 'failure.json', {'traceback': traceback.format_exc()})
                        raise  # No esconder fallos en una ejecución nocturna.
                    if result is None:
                        stop = True
                        break
                if stop:
                    break
            # Seleccionar únicamente configuraciones completadas por las tres arquitecturas.
            balanced = [i for i in range(len(configs)) if all(
                (out / 'search' / f'{i:03d}_{a}' / 'result.json').exists() for a in cfg['architectures'])]
            if not balanced:
                print('Presupuesto agotado antes de una comparación completa. Reanude con el mismo comando.', flush=True)
                return False
            selected = {}
            for architecture in cfg['architectures']:
                candidates = [(i, read_json(out / 'search' / f'{i:03d}_{architecture}' / 'result.json')) for i in balanced]
                index, result = max(candidates, key=lambda item: (item[1]['dev']['macro_f1'], -item[0]))
                selected[architecture] = {'index': index, 'hp': result['hp'], 'search_dev_macro_f1': result['dev']['macro_f1']}
            atomic_json(selection_path, {'balanced_trials': balanced, 'planned_trials': len(configs),
                'budget_truncated': len(balanced) < len(configs), 'selected': selected,
                'rule': 'Mayor F1 macro dev; empate: menor índice. Test no consultado.'})
        selection = read_json(selection_path)
        for seed in cfg['final_seeds']:
            for architecture in cfg['architectures']:
                result = fit(out / 'final' / f'{architecture}_seed{seed}', architecture,
                             selection['selected'][architecture]['hp'], seed, cfg, parts, device, deadline)
                if result is None:
                    print('Presupuesto agotado. Checkpoint guardado; reanude con el mismo comando.', flush=True)
                    return False
        # Las configuraciones y todas las semillas quedan fijadas ANTES de leer señales/etiquetas test.
        evaluate_final(out, data_dir, cfg, parts, device)
        from .analysis import analyze
        analyze(out)
        print(f'Completado. Resultados e informe: {out / "analysis"}', flush=True)
        return True


def fit_baselines(out, cfg, parts):
    baseline_dir = out / 'baselines'
    baseline_dir.mkdir(exist_ok=True)
    for name, estimator in [('majority', DummyClassifier(strategy='most_frequent')),
        ('logistic', make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, random_state=cfg['seed'])))]:
        if (baseline_dir / f'{name}.json').exists():
            continue
        estimator.fit(summary_features(parts['train'][0]), parts['train'][1])
        results = {}
        for split, (x, y, subjects, indices) in parts.items():
            pred = estimator.predict(summary_features(x))
            results[split] = metrics(y, pred)
            np.savez_compressed(baseline_dir / f'{name}_{split}.npz', y=y, pred=pred, subjects=subjects, indices=indices)
        tmp = baseline_dir / f'{name}.pkl.tmp'
        with tmp.open('wb') as f:
            pickle.dump(estimator, f)
        tmp.replace(baseline_dir / f'{name}.pkl')
        atomic_json(baseline_dir / f'{name}.json', results)
        print(f'Baseline {name}: dev F1={results["dev"]["macro_f1"]:.4f}', flush=True)


def evaluate_final(out, data_dir, cfg, parts, device):
    if (out / 'evaluation.json').exists():
        return
    xtest, ytest, stest = load_official(data_dir, 'test')
    assert len(set(stest)) == 9
    if cfg.get('debug_limit_per_split'):
        rng = np.random.default_rng(cfg['seed'])
        ids = np.concatenate([rng.permutation(np.flatnonzero(ytest == c))[:cfg['debug_limit_per_split'] // 6] for c in range(6)])
        xtest, ytest, stest = xtest[ids], ytest[ids], stest[ids]
    else:
        ids = np.arange(len(ytest))
    final = []
    for architecture in cfg['architectures']:
        for seed in cfg['final_seeds']:
            folder = out / 'final' / f'{architecture}_seed{seed}'
            result = read_json(folder / 'result.json')
            if (folder / 'test_metrics.json').exists():
                test_m = read_json(folder / 'test_metrics.json')
            else:
                ckpt = torch.load(folder / 'best.pt', map_location='cpu', weights_only=False)
                model = HARModel(architecture, **ckpt['hp']).to(device)
                model.load_state_dict(ckpt['model'])
                x = transform(xtest, ckpt['hp']['window'], ckpt['mean'], ckpt['std'])
                test_m, pred, probs = evaluate(model, loader(x, ytest, cfg['batch_size']), device)
                np.savez_compressed(folder / 'test_predictions.npz', y=ytest, pred=pred, probabilities=probs,
                                    subjects=stest, indices=ids)
                atomic_json(folder / 'test_metrics.json', test_m)
            final.append({**result, 'test': test_m})
    baseline_dir = out / 'baselines'
    baseline_results = {}
    for name in ('majority', 'logistic'):
        results = read_json(baseline_dir / f'{name}.json')
        if 'test' not in results:
            with (baseline_dir / f'{name}.pkl').open('rb') as f:
                estimator = pickle.load(f)  # Solo artefactos creados por este proyecto.
            pred = estimator.predict(summary_features(xtest))
            results['test'] = metrics(ytest, pred)
            np.savez_compressed(baseline_dir / f'{name}_test.npz', y=ytest, pred=pred, subjects=stest, indices=ids)
            atomic_json(baseline_dir / f'{name}.json', results)
        baseline_results[name] = results
    atomic_json(out / 'evaluation.json', {'models': final, 'baselines': baseline_results,
                'test_n': len(ytest), 'test_subjects': sorted(map(int, set(stest))),
                'note': 'Semillas individuales; no se selecciona la mejor semilla usando test ni se reporta un ensemble.'})


def status(out):
    out = Path(out)
    results = []
    for p in sorted(out.glob('search/*/result.json')):
        r = read_json(p)
        results.append({'trial': p.parent.name, 'dev_macro_f1': r['dev']['macro_f1'],
                        'best_epoch': r['best_epoch'], 'seconds': round(r['seconds'], 1)})
    print(json.dumps({'search_completed': len(results), 'final_completed': len(list(out.glob('final/*/result.json'))),
                      'test_evaluated': (out / 'evaluation.json').exists(), 'trials': results}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prepare = sub.add_parser('prepare')
    prepare.add_argument('--data', default='data')
    train = sub.add_parser('run')
    train.add_argument('--data', default='data')
    train.add_argument('--config', default='configs/overnight.json')
    train.add_argument('--out', default='runs/overnight')
    train.add_argument('--hours', type=float, default=8)
    for name in ('analyze', 'status'):
        p = sub.add_parser(name)
        p.add_argument('--out', default='runs/overnight')
    args = parser.parse_args()
    if args.command == 'prepare':
        print('Dataset SHA256:', download(args.data))
    elif args.command == 'run':
        if not run(args):
            sys.exit(2)
    elif args.command == 'analyze':
        from .analysis import analyze
        analyze(Path(args.out))
    else:
        status(args.out)


if __name__ == '__main__':
    main()
