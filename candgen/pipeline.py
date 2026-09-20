"""Сценарии верхнего уровня: признаки для фолдов, «честная» валидация, предсказание бенчмарка."""
import gc

import numpy as np
import pandas as pd

from . import config as C
from .candidates import generate
from .data import assign_folds, group_queries, load
from .item_index import ItemIndex
from .ranker import Ensemble
from .train_stats import TrainStats


class Context:
    """Данные + индекс корпуса + фолды. Создаётся ~1.5 минуты."""

    def __init__(self):
        self.items, self.train, self.bench = load()
        print('строю индекс корпуса...', flush=True)
        self.index = ItemIndex(self.items)
        self.groups, self.train = assign_folds(self.train)


def _save(df, path):
    df['q'] = df.q.astype(np.int32)
    df['iidx'] = df.iidx.astype(np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def build_fold_features(ctx, name, folds, n_groups, bag=0, K=C.K_TRAIN, neg_keep=None):
    """Признаки для групп выбранных фолдов; статистики трейна — out-of-fold.

    n_groups — сколько групп брать из каждого фолда; bag — номер непересекающегося
    среза (для бэггинга: bag=0 — первые n_groups групп, bag=1 — следующие и т.д.).
    Результат: FEATURES_DIR/{name}_fold{f}.parquet.
    """
    paths = []
    for f in folds:
        stats = TrainStats(ctx.train[ctx.train.fold != f], ctx.index)
        g = ctx.groups[ctx.groups.fold == f].sample(frac=1, random_state=f)
        g = g.iloc[bag * n_groups:(bag + 1) * n_groups].reset_index(drop=True)
        print(f'фолд {f}: {len(g)} групп', flush=True)
        F = generate(g, ctx.index, stats, K=K, neg_keep=neg_keep, seed=f)
        F['q'] = F.q + f * 100000  # уникальный номер запроса между фолдами
        F['npos'] = F.q.map(dict(zip(np.arange(len(g)) + f * 100000, g.pos.map(len))))
        path = C.FEATURES_DIR / f'{name}_fold{f}.parquet'
        _save(F, path)
        paths.append(path)
        del F, stats
        gc.collect()
    return paths


def build_honest_features(ctx, K=C.K_TRAIN, name='honest'):
    """Валидация в условиях бенчмарка: поиск только среди объявлений бенчмарка,
    запросы — из валидационного фолда, у которых все выбранные объявления есть в бенчмарке."""
    inb = ctx.index.in_bench
    g = ctx.groups[ctx.groups.fold == C.VAL_FOLD]
    g = g[g.pos.map(lambda p: all(inb[p]))].reset_index(drop=True)
    print(f'honest: {len(g)} групп', flush=True)
    stats = TrainStats(ctx.train[ctx.train.fold != C.VAL_FOLD], ctx.index)
    F = generate(g, ctx.index, stats, K=K, restrict=inb)
    F['npos'] = F.q.map(g.pos.map(len))
    path = C.FEATURES_DIR / f'{name}.parquet'
    _save(F, path)
    return path


def predict_benchmark(ctx, model_paths, K=C.K_PREDICT, batch=100):
    """Статистики — по всему трейну; поиск — только в объявлениях бенчмарка."""
    stats = TrainStats(ctx.train, ctx.index)
    ens = Ensemble(model_paths)
    queries = ctx.bench.reset_index(drop=True)
    item_ids = ctx.items.item_id.values
    answers = {}
    for s in range(0, len(queries), batch):
        qb = queries.iloc[s:s + batch].reset_index(drop=True)
        F = generate(qb, ctx.index, stats, K=K, restrict=ctx.index.in_bench, with_labels=False, verbose=False)
        F = F[['q', 'iidx']].assign(p=ens.predict(F[ens.features].values.astype(np.float32)))
        for qi, g in F.groupby('q'):
            answers[qb.query_id[qi]] = item_ids[g.nlargest(C.TOP_N, 'p').iidx.values]
        print(f'  {min(s + batch, len(queries))}/{len(queries)}', flush=True)
        del F
        gc.collect()
    return pd.DataFrame({'query_id': list(answers), 'answer': [' '.join(v) for v in answers.values()]})


def check_answer(answer, bench_queries, bench_items):
    """Проверка формата answer.csv по требованиям задачи."""
    ids = set(bench_items.item_id)
    lists = answer.answer.str.split()
    assert list(answer.columns) == ['query_id', 'answer']
    assert answer.query_id.is_unique and set(answer.query_id) == set(bench_queries.query_id), 'не те query_id'
    assert (answer.query_id.str.len() == 16).all()
    assert lists.map(len).max() <= C.TOP_N, 'больше 50 объявлений'
    assert lists.map(lambda x: len(set(x)) == len(x)).all(), 'повторы внутри строки'
    assert lists.map(lambda x: all(i in ids for i in x)).all(), 'item_id не из benchmark_items'
