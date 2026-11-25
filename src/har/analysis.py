"""Tablas, curvas, errores por sujeto/clase e informe IEEE con resultados reales."""
from __future__ import annotations
import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from .data import LABELS
from .train import atomic_json, metrics

DISPLAY = {'rnn': 'RNN', 'lstm': 'LSTM', 'cnn_lstm': 'CNN-LSTM',
           'majority': 'Mayoría', 'logistic': 'Reg. logística'}


def read(path):
    return json.loads(Path(path).read_text())


def latex_escape(text):
    return str(text).replace('_', r'\_').replace('%', r'\%').replace('&', r'\&')


def savefig(fig, folder, name):
    fig.savefig(folder / f'{name}.pdf', bbox_inches='tight')
    fig.savefig(folder / f'{name}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def analyze(out):
    out = Path(out).resolve()
    if not (out / 'evaluation.json').exists():
        raise RuntimeError('Aún faltan entrenamientos finales. Consulte status y reanude run; test sigue reservado.')
    manifest = read(out / 'manifest.json')
    cfg = manifest['identity']['config']
    selection = read(out / 'selection.json')
    evaluation = read(out / 'evaluation.json')
    dest = out / 'analysis'
    dest.mkdir(exist_ok=True)
    architectures = cfg['architectures']
    runs = evaluation['models']
    rows = []
    for model in runs:
        rows.append({'model': model['architecture'], 'seed': model['seed'],
            'parameters': model['parameters'], 'seconds': model['seconds'], 'best_epoch': model['best_epoch'],
            **{f'{split}_{m}': model[split][m] for split in ('train', 'dev', 'test') for m in ('accuracy', 'macro_f1')}})
    frame = pd.DataFrame(rows)
    frame.to_csv(dest / 'per_seed_metrics.csv', index=False)
    columns = ['train_macro_f1', 'dev_macro_f1', 'test_macro_f1', 'test_accuracy', 'parameters', 'seconds']
    summary = frame.groupby('model', sort=False)[columns].agg(['mean', 'std'])
    summary.to_csv(dest / 'summary.csv')
    search_rows = []
    for p in sorted(out.glob('search/*/result.json')):
        r = read(p)
        search_rows.append({'trial': p.parent.name, 'architecture': r['architecture'], **r['hp'],
                            'dev_macro_f1': r['dev']['macro_f1'], 'train_macro_f1': r['train']['macro_f1'],
                            'seconds': r['seconds'], 'balanced': int(p.parent.name.split('_')[0]) in selection['balanced_trials']})
    pd.DataFrame(search_rows).to_csv(dest / 'search_results.csv', index=False)
    atomic_json(dest / 'baselines.json', evaluation['baselines'])
    # Curvas por semilla: train y dev medidos ambos en modo evaluación (dropout desactivado).
    fig, axes = plt.subplots(2, 3, figsize=(12, 5), sharex='col', sharey='row')
    for col, architecture in enumerate(architectures):
        for seed_index, seed in enumerate(cfg['final_seeds']):
            history = pd.DataFrame(read(out / 'final' / f'{architecture}_seed{seed}' / 'history.json'))
            for row, metric in enumerate(('loss', 'macro_f1')):
                for split, color, style in [('train', '#2166ac', '-'), ('dev', '#b2182b', '--')]:
                    axes[row, col].plot(history['epoch'], history[f'{split}_{metric}'], color=color,
                                        linestyle=style, alpha=0.6, linewidth=1,
                                        label=split if seed_index == 0 else None)
                axes[row, col].grid(alpha=0.2)
                axes[row, col].set_ylabel('Entropía cruzada' if metric == 'loss' else 'F1 macro')
        axes[0, col].set_title(DISPLAY[architecture])
        axes[0, col].legend(fontsize=8)
        axes[1, col].set_xlabel('Época')
        axes[1, col].set_ylim(0, 1)
    fig.tight_layout()
    savefig(fig, dest, 'learning_curves')
    # Una semilla predefinida para ejemplos y matrices; nunca elegir por test.
    representative_seed = cfg['final_seeds'][0]
    by_subject, by_class, confusions = [], [], []
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    prediction_sets = {}
    for col, architecture in enumerate(architectures):
        folder = out / 'final' / f'{architecture}_seed{representative_seed}'
        prediction_sets[architecture] = {}
        for seed in cfg['final_seeds']:
            sfolder = out / 'final' / f'{architecture}_seed{seed}'
            for split in ('dev', 'test'):
                with np.load(sfolder / f'{split}_predictions.npz') as d:
                    y, pred, subjects = d['y'], d['pred'], d['subjects']
                    for subject in np.unique(subjects):
                        mask = subjects == subject
                        # F1 por sujeto puede penalizar clases ausentes; se exporta soporte para interpretarlo.
                        by_subject.append({'model': architecture, 'seed': seed, 'split': split,
                            'subject': int(subject), 'n': int(mask.sum()),
                            'present_classes': int(len(np.unique(y[mask]))), **metrics(y[mask], pred[mask])})
                    report = classification_report(y, pred, labels=list(range(6)), target_names=LABELS,
                                                   output_dict=True, zero_division=0)
                    for label in LABELS:
                        by_class.append({'model': architecture, 'seed': seed, 'split': split,
                                         'class': label, **report[label]})
                    if seed == representative_seed:
                        confidence = d['probabilities'].max(1)
                        detail = pd.DataFrame({'index': d['indices'], 'subject': subjects,
                            'true': [LABELS[i] for i in y], 'predicted': [LABELS[i] for i in pred],
                            'confidence': confidence, 'error': y != pred})
                        detail.to_csv(dest / f'predictions_{architecture}_{split}.csv', index=False)
                        detail[detail.error].sort_values('confidence', ascending=False).to_csv(
                            dest / f'errors_{architecture}_{split}.csv', index=False)
                        prediction_sets[architecture][split] = (y.copy(), pred.copy())
                        matrix = confusion_matrix(y, pred, labels=list(range(6)))
                        for i in range(6):
                            for j in range(6):
                                if i != j:
                                    confusions.append({'model': architecture, 'split': split,
                                        'true': LABELS[i], 'predicted': LABELS[j], 'count': int(matrix[i, j])})
                        if split == 'test':
                            ConfusionMatrixDisplay.from_predictions(y, pred, labels=list(range(6)),
                                display_labels=['W', 'WU', 'WD', 'Sit', 'Std', 'Lay'], normalize='true',
                                colorbar=False, ax=axes[col], values_format='.2f', cmap='Blues')
                            axes[col].set_title(DISPLAY[architecture])
    fig.tight_layout()
    savefig(fig, dest, 'confusion_test')
    pd.DataFrame(by_subject).to_csv(dest / 'by_subject.csv', index=False)
    pd.DataFrame(by_class).to_csv(dest / 'by_class.csv', index=False)
    pd.DataFrame(confusions).sort_values('count', ascending=False).to_csv(dest / 'confusion_pairs.csv', index=False)
    # Solapamiento de errores: útil para explicar fortalezas complementarias sin entrenar un ensemble.
    overlap = []
    for split in ('dev', 'test'):
        for i, a in enumerate(architectures):
            ya, pa = prediction_sets[a][split]
            for b in architectures[i+1:]:
                yb, pb = prediction_sets[b][split]
                assert np.array_equal(ya, yb)
                ea, eb = pa != ya, pb != yb
                overlap.append({'split': split, 'a': a, 'b': b, 'both_wrong': int((ea & eb).sum()),
                                'only_a_wrong': int((ea & ~eb).sum()), 'only_b_wrong': int((~ea & eb).sum())})
    pd.DataFrame(overlap).to_csv(dest / 'error_overlap.csv', index=False)
    winner = max(architectures, key=lambda a: float(frame.loc[frame.model == a, 'dev_macro_f1'].mean()))
    atomic_json(dest / 'diagnostics.json', {'recommended_by_dev_mean': winner,
        'comparison_scope': 'recommended_by_dev_mean compara las tres arquitecturas neuronales, no los baselines',
        'representative_seed': representative_seed,
        'gaps_train_minus_dev': {a: float((frame.loc[frame.model == a, 'train_macro_f1'] -
                                         frame.loc[frame.model == a, 'dev_macro_f1']).mean()) for a in architectures},
        'interpretation': 'Brechas descriptivas, no prueba de sesgo evitable: falta referencia de error óptimo. '
                          'La desviación entre semillas no es un intervalo de confianza poblacional.'})
    write_report(out, dest, frame, cfg, selection, evaluation, winner, manifest)
    print(frame[['model', 'seed', 'train_macro_f1', 'dev_macro_f1', 'test_macro_f1']].to_string(index=False))
    print(f'Artefactos: {dest}')


def write_report(out, dest, frame, cfg, selection, evaluation, winner, manifest):
    report_dir = dest / 'report'
    report_dir.mkdir(exist_ok=True)
    source = Path(__file__).resolve().parents[2] / 'report'
    shutil.copy2(source / 'template.tex', report_dir / 'main.tex')
    if (source / 'IEEEtran.cls').exists():
        shutil.copy2(source / 'IEEEtran.cls', report_dir / 'IEEEtran.cls')
    for name in ('learning_curves.pdf', 'confusion_test.pdf'):
        shutil.copy2(dest / name, report_dir / name)
    architectures = cfg['architectures']
    lines = [r'\begin{tabular}{lrrrr}', r'\hline',
             r'Modelo & F1 train & F1 dev & F1 test & Acc. test \\', r'\hline']
    for a in architectures:
        subset = frame[frame.model == a]
        entries = []
        for metric in ('train_macro_f1', 'dev_macro_f1', 'test_macro_f1', 'test_accuracy'):
            vals = subset[metric] * 100
            if len(vals) > 1:
                entries.append(f'{vals.mean():.1f}' + r'$\pm$' + f'{vals.std(ddof=1):.1f}')
            else:
                entries.append(f'{vals.mean():.1f}')
        lines.append(DISPLAY[a] + ' & ' + ' & '.join(entries) + r' \\')
    for name, result in evaluation['baselines'].items():
        vals = [result[s][m] * 100 for s, m in [('train','macro_f1'),('dev','macro_f1'),('test','macro_f1'),('test','accuracy')]]
        lines.append(DISPLAY[name] + ' & ' + ' & '.join(f'{v:.1f}' for v in vals) + r' \\')
    lines += [r'\hline', r'\end{tabular}']
    (report_dir / 'metrics_table.tex').write_text('\n'.join(lines))
    lines = [r'\begin{tabular}{lrrrrrr}', r'\hline',
             r'Modelo & $T$ & $h$ & $p$ & $\eta$ & $\lambda$ & Parám. \\', r'\hline']
    for a in architectures:
        hp = selection['selected'][a]['hp']
        params = int(frame.loc[frame.model == a, 'parameters'].iloc[0])
        lines.append(f'{DISPLAY[a]} & {hp["window"]} & {hp["hidden"]} & {hp["dropout"]} & {hp["lr"]:g} & {hp["weight_decay"]:g} & {params}' + r' \\')
    lines += [r'\hline', r'\end{tabular}']
    (report_dir / 'hyperparameters_table.tex').write_text('\n'.join(lines))
    split = read(out / 'split.json')
    winner_rows = frame[frame.model == winner]
    best_dev = winner_rows.dev_macro_f1.mean() * 100
    best_test = winner_rows.test_macro_f1.mean() * 100
    search_n = len(selection['balanced_trials'])
    representative_seed = cfg['final_seeds'][0]
    pairs = pd.read_csv(dest / 'confusion_pairs.csv')
    dev_pairs = pairs[(pairs.model == winner) & (pairs.split == 'dev')].sort_values('count', ascending=False)
    top = dev_pairs.iloc[0]
    error_summaries = []
    for a in architectures:
        ap = pairs[(pairs.model == a) & (pairs.split == 'dev')].sort_values('count', ascending=False).iloc[0]
        if ap['count']:
            error_summaries.append(f"{DISPLAY[a]}: {latex_escape(ap['true'])} hacia {latex_escape(ap['predicted'])} ({int(ap['count'])})")
        else:
            error_summaries.append(f'{DISPLAY[a]}: sin errores')
    logistic_dev = evaluation['baselines']['logistic']['dev']['macro_f1'] * 100
    logistic_test = evaluation['baselines']['logistic']['test']['macro_f1'] * 100
    baseline_comparison = ('La referencia logística supera en validación al mejor sistema neuronal; '
        'estos resultados no justifican preferir una arquitectura recurrente por precisión predictiva.'
        if logistic_dev >= best_dev else
        'El sistema neuronal seleccionado supera a la referencia logística en validación; '
        'la magnitud y estabilidad de la diferencia deben contrastarse con su costo adicional.')
    gap = (winner_rows.train_macro_f1 - winner_rows.dev_macro_f1).mean() * 100
    hours = frame.seconds.sum() / 3600
    macros = {
        'Winner': DISPLAY[winner], 'WinnerDev': f'{best_dev:.2f}', 'WinnerTest': f'{best_test:.2f}',
        'TrainN': split['train']['n'], 'DevN': split['dev']['n'], 'TestN': evaluation['test_n'],
        'TrainSubjects': len(split['train']['subjects']), 'DevSubjects': len(split['dev']['subjects']),
        'DevSubjectList': ', '.join(map(str, cfg['dev_subjects'])),
        'Trials': search_n, 'PlannedTrials': cfg['trials'], 'Seeds': len(cfg['final_seeds']),
        'Epochs': cfg['max_epochs'], 'Patience': cfg['patience'], 'BatchSize': cfg['batch_size'],
        'Device': str(manifest['device']), 'Gap': f'{gap:.2f}',
        'LogisticDev': f'{logistic_dev:.2f}', 'LogisticTest': f'{logistic_test:.2f}',
        'BaselineComparison': baseline_comparison, 'ErrorSummary': '; '.join(error_summaries),
        'ErrorTrue': latex_escape(top['true']), 'ErrorPred': latex_escape(top['predicted']), 'ErrorCount': int(top['count']),
        'RepresentativeSeed': representative_seed, 'FinalHours': f'{hours:.2f}',
        'RunLabel': 'PRUEBA DE HUMO: NO USAR COMO RESULTADO FINAL' if manifest['debug'] else 'Experimento completo',
        'SearchNote': ('El presupuesto limitó la búsqueda; se excluyeron configuraciones incompletas de la selección.'
                       if selection['budget_truncated'] else 'Se completó el presupuesto de configuraciones previsto.')}
    (report_dir / 'values.tex').write_text('\n'.join(
        rf'\newcommand{{\{key}}}{{{value}}}' for key, value in macros.items()))
    # El paquete completo se puede subir a Overleaf como ZIP.
    shutil.make_archive(str(dest / 'informe_ieee'), 'zip', report_dir)
