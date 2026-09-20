"""Обучение и применение LightGBM-ранкера, метрика Recall@K."""
import gc

import lightgbm as lgb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from . import config as C
from .candidates import EMB_FEATURES, FEATURES

META = ('q', 'y', 'npos', 'iidx')


def feature_names(path):
    names = set(pq.ParquetFile(path).schema_arrow.names)
    return FEATURES + [c for c in EMB_FEATURES if c in names]


def load_features(paths, features=None):
    """Читает таблицы признаков в плотную float32-матрицу (экономно по памяти)
    и сортирует по номеру запроса — LightGBM требует, чтобы группы шли подряд."""
    paths = [str(p) for p in paths]
    features = features or feature_names(paths[0])
    Xs, metas = [], []
    for p in paths:
        cols = set(pq.ParquetFile(p).schema_arrow.names)
        t = pq.read_table(p, columns=features + [c for c in META if c in cols])
        Xs.append(np.column_stack([t.column(c).to_numpy().astype(np.float32) for c in features]))
        metas.append(pd.DataFrame({c: t.column(c).to_numpy() for c in META if c in cols}))
        del t
        gc.collect()
    X = np.concatenate(Xs) if len(Xs) > 1 else Xs[0]
    del Xs
    M = pd.concat(metas, ignore_index=True)
    order = np.argsort(M.q.values, kind='stable')
    return X[order], M.iloc[order].reset_index(drop=True), features


def recall_at_k(meta, scores, k=C.TOP_N):
    """(Recall@k по ранкеру, потолок — доля релевантных, попавших в кандидаты)."""
    m = meta.assign(p=scores)
    rank = m.groupby('q').p.rank(ascending=False, method='first')
    npos = m.groupby('q').npos.first()
    rec = (m[rank <= k].groupby('q').y.sum() / npos).reindex(m.q.unique()).fillna(0)
    ceiling = m.groupby('q').y.sum() / npos
    return float(rec.mean()), float(ceiling.mean())


def train(paths, n_trees=C.N_TREES_TRAIN, params=None):
    X, M, features = load_features(paths)
    # запросы, у которых нет ни одного релевантного кандидата, ничему не учат ранкер
    has_pos = M.groupby('q').y.transform('max').values > 0
    X, M = X[has_pos], M[has_pos].reset_index(drop=True)
    print(f'обучающая выборка: {X.shape[0]} пар, {M.q.nunique()} запросов, {X.shape[1]} признаков', flush=True)
    ds = lgb.Dataset(X, M.y.values, group=M.groupby('q', sort=False).size().values,
                     feature_name=features, free_raw_data=True)
    model = lgb.train(params or C.LGB_PARAMS, ds, n_trees)
    return model


class Ensemble:
    """Среднее z-нормированных скоров нескольких моделей (бэггинг по разным группам трейна)."""

    def __init__(self, model_paths, n_trees=C.N_TREES_PREDICT):
        self.models = [lgb.Booster(model_file=str(p)) for p in model_paths]
        self.features = self.models[0].feature_name()
        self.n_trees = n_trees

    def predict(self, X):
        total = np.zeros(len(X))
        for m in self.models:
            s = m.predict(X, num_iteration=min(self.n_trees, m.current_iteration()))
            total += (s - s.mean()) / (s.std() + 1e-9)
        return total / len(self.models)
