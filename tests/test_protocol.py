"""Invariantes científicos y recuperación tras interrupciones."""
import json
import time
import numpy as np
import pytest
import torch
from har.data import split_by_subject, normalize_fit, transform
from har.experiment import configurations, validate_config
from har.models import HARModel
from har import train


def artificial_parts():
    rng = np.random.default_rng(123)
    x = rng.normal(size=(48, 128, 9)).astype(np.float32)
    y = np.tile(np.arange(6, dtype=np.int64), 8)
    subjects = np.repeat(np.arange(1, 9), 6)
    return x, y, subjects


def test_subject_partition_keeps_overlapping_windows_together():
    x, y, s = artificial_parts()
    parts = split_by_subject(x, y, s, [2, 5])
    train_ids, dev_ids = parts['train'][3], parts['dev'][3]
    assert set(s[train_ids]).isdisjoint(s[dev_ids])
    assert sorted(np.r_[train_ids, dev_ids]) == list(range(len(y)))
    assert set(s[dev_ids]) == {2, 5}
    with pytest.raises(ValueError):
        split_by_subject(x, y, s, [9])


def test_scaling_is_fitted_on_train_and_crop_preserves_temporal_order():
    x, y, s = artificial_parts()
    parts = split_by_subject(x, y, s, [2, 5])
    train_x, dev_x = parts['train'][0], parts['dev'][0]
    mean, std = normalize_fit(train_x[:, -64:])
    normalized = transform(train_x, 64, mean, std)
    np.testing.assert_allclose(normalized.mean((0, 1)), 0, atol=1e-6)
    np.testing.assert_allclose(normalized.std((0, 1)), 1, atol=2e-6)
    shifted = transform(dev_x + 1000, 64, mean, std)
    assert shifted.mean() > 100
    recovered = normalized * std + mean
    np.testing.assert_allclose(recovered, train_x[:, -64:], atol=5e-7)


@pytest.mark.parametrize('architecture', ['rnn', 'lstm', 'cnn_lstm'])
def test_models_train_with_multichannel_sequences(architecture):
    torch.set_num_threads(1)
    model = HARModel(architecture, hidden=8, dropout=0.25, conv_channels=8)
    x = torch.randn(12, 64, 9)
    y = torch.arange(12) % 6
    logits = model(x)
    assert logits.shape == (12, 6)
    loss = torch.nn.functional.cross_entropy(logits, y)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_search_plan_is_shared_unique_and_reproducible():
    cfg = json.load(open('configs/overnight.json'))
    validate_config(cfg)
    plan = configurations(cfg)
    assert plan == configurations(cfg)
    assert len(plan) == len({json.dumps(hp, sort_keys=True) for hp in plan}) == 16
    assert {p['window'] for p in plan} == {64, 128}


def test_resume_restores_optimizer_rng_and_best_checkpoint(tmp_path, monkeypatch):
    torch.set_num_threads(1)
    x, y, s = artificial_parts()
    parts = split_by_subject(x, y, s, [2, 5])
    cfg = {'max_epochs': 3, 'batch_size': 12, 'patience': 5, 'min_delta': 0.0001}
    hp = {'hidden': 8, 'window': 64, 'dropout': 0.25, 'lr': 0.001, 'weight_decay': 0.0001}
    uninterrupted = train.fit(tmp_path / 'full', 'rnn', hp, 42, cfg, parts, torch.device('cpu'), float('inf'))
    real_save = train.save_torch
    class SimulatedInterruption(Exception):
        pass
    def interrupt_after_checkpoint(path, state):
        real_save(path, state)
        if path.name == 'last.pt':
            raise SimulatedInterruption
    monkeypatch.setattr(train, 'save_torch', interrupt_after_checkpoint)
    with pytest.raises(SimulatedInterruption):
        train.fit(tmp_path / 'resumed', 'rnn', hp, 42, cfg, parts, torch.device('cpu'), float('inf'))
    monkeypatch.setattr(train, 'save_torch', real_save)
    resumed = train.fit(tmp_path / 'resumed', 'rnn', hp, 42, cfg, parts, torch.device('cpu'), float('inf'))
    assert uninterrupted['dev'] == resumed['dev']
    assert uninterrupted['best_epoch'] == resumed['best_epoch']
    full = torch.load(tmp_path / 'full/best.pt', weights_only=False)
    recovered = torch.load(tmp_path / 'resumed/best.pt', weights_only=False)
    assert all(torch.equal(full['model'][k], recovered['model'][k]) for k in full['model'])


def test_expired_budget_does_not_create_a_completed_trial(tmp_path):
    x, y, s = artificial_parts()
    parts = split_by_subject(x, y, s, [2, 5])
    assert train.fit(tmp_path / 'expired', 'rnn', {}, 42, {}, parts,
                     torch.device('cpu'), time.monotonic() - 1) is None
    assert not (tmp_path / 'expired/result.json').exists()
