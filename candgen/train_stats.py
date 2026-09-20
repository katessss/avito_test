"""Статистики, которые извлекаются из обучающих пар «запрос → выбранное объявление».

Строятся на тех строках трейна, которые переданы в конструктор. При генерации
обучающих признаков для фолда f сюда передаётся трейн без фолда f (out-of-fold),
иначе признаки «подсмотрели» бы ответ.

Что считается:
  * trans     — переходы «локация поиска → локация выбранного объявления».
                Поиск по региону/стране (напр. 107620, 621540) не совпадает ни с одной
                локацией объявлений; распределение переходов говорит, где искать.
  * соседи    — похожие запросы из трейна (TF-IDF по словам и символьным n-граммам).
                От них берём: расширение запроса словами из выбранных объявлений,
                распределение подкатегорий, сами выбранные объявления.
  * mc_ctr    — насколько часто объявления подкатегории выбирают относительно их числа.
"""
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from . import config as C
from .data import group_queries


class TrainStats:
    def __init__(self, rows, index):
        # ---- переходы локаций
        counts = rows.groupby(['search_location_id', 'item_location_id']).size()
        self.trans = defaultdict(dict)
        for (s, i), c in counts.items():
            self.trans[s][i] = c
        self.trans_total = {s: sum(d.values()) for s, d in self.trans.items()}

        # ---- центр поиска: для локаций без объявлений — центр самой частой целевой локации
        self.search_center = {}
        for s, d in self.trans.items():
            if s in index.loc_center:
                self.search_center[s] = index.loc_center[s]
            else:
                known = [(c, l) for l, c in d.items() if l in index.loc_center]
                if known:
                    w = np.array([c for c, _ in known], float)
                    self.search_center[s] = index.loc_center[known[int(np.argmax(w))][1]]

        # ---- индекс запросов трейна для поиска соседей
        groups = group_queries(rows)
        self.groups = groups
        self.vec_word = TfidfVectorizer(token_pattern=r'\S+', ngram_range=(1, 2), sublinear_tf=True, dtype=np.float32)
        self.vec_char = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 4), sublinear_tf=True, dtype=np.float32)
        self.Q_word = self.vec_word.fit_transform(groups.query_n.values)
        self.Q_char = self.vec_char.fit_transform(groups.search_query.str.lower().values)

        # A: группа × объявление (строки нормированы на 1) — какие объявления выбирали в группе
        rr, cc = [], []
        for gi, pos in enumerate(groups.pos):
            rr += [gi] * len(pos)
            cc += list(pos)
        A = normalize(sp.csr_matrix((np.ones(len(rr), np.float32), (rr, cc)), shape=(len(groups), index.n)), 'l1')
        self.group_items = A.tocsr()
        # E: группа × термин — средний нормированный вектор «заголовок+параметры» выбранных объявлений
        self.group_terms = (A @ normalize(index.title_params_csr, 'l2')).tocsr()
        self.group_microcats = [index.microcat[p] for p in groups.pos]

        # ---- «кликабельность» подкатегорий (со сглаживанием)
        clicks = pd.Series(index.microcat[rows.iidx.values]).value_counts()
        sizes = pd.Series(index.microcat).value_counts()
        self.mc_ctr = (clicks.reindex(sizes.index).fillna(0) + 1) / (sizes + 50)

        self.query_count = rows.search_query.value_counts()

    def neighbors(self, queries, k=C.N_NEIGHBORS, batch=500):
        """Для каждого запроса — (индексы k ближайших групп трейна, их сходство), по убыванию."""
        qw = self.vec_word.transform(queries.query_n.values)
        qc = self.vec_char.transform(queries.search_query.str.lower().values)
        out = []
        for s in range(0, len(queries), batch):
            S = (0.5 * (qw[s:s + batch] @ self.Q_word.T) + 0.5 * (qc[s:s + batch] @ self.Q_char.T)).toarray()
            for row in S:
                idx = np.argpartition(-row, k)[:k]
                idx = idx[np.argsort(-row[idx])]
                out.append((idx, row[idx]))
        return out
