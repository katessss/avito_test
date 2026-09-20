"""Оценка Recall@50 моделей на наборах признаков.

    python -m scripts.evaluate --models models/ranker_bag0.txt,models/ranker_bag1.txt,models/ranker_bag2.txt \
                               --features val_fold0,honest
Печатает Recall@50 каждой модели, ансамбля, стартового скора (без ранкера) и потолок
(доля релевантных, попавших в кандидаты).
"""
import argparse

import numpy as np

from candgen import config as C
from candgen.ranker import Ensemble, load_features, recall_at_k

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', required=True)
    ap.add_argument('--features', required=True, help='имена файлов в FEATURES_DIR без .parquet, через запятую')
    ap.add_argument('--trees', type=int, default=C.N_TREES_PREDICT)
    a = ap.parse_args()
    ens = Ensemble(a.models.split(','), a.trees)
    for name in a.features.split(','):
        X, M, _ = load_features([C.FEATURES_DIR / f'{name}.parquet'], ens.features)
        print(f'== {name}: {M.q.nunique()} запросов, {len(M)} кандидатов')
        stage = -X[:, ens.features.index('stage_rank')]
        print(f'   без ранкера (стартовый скор): {recall_at_k(M, stage)[0]:.4f}')
        for path, m in zip(a.models.split(','), ens.models):
            print(f'   {path}: {recall_at_k(M, m.predict(X, num_iteration=a.trees))[0]:.4f}')
        rec, ceil = recall_at_k(M, ens.predict(X))
        print(f'   ансамбль: {rec:.4f}   потолок кандидатов: {ceil:.4f}')
