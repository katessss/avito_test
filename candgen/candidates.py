"""Генерация кандидатов и признаков для ранкера.

Для каждого запроса:
  1. Пул — объявления из «подходящих» локаций (своя + куда ведут переходы из трейна + в радиусе 30 км).
  2. Скоры по всему корпусу: BM25 по полям, покрытие слов запроса, символьная близость,
     расширение запроса по соседям из трейна, (опционально) косинус эмбеддингов.
  3. Кандидаты — объединение топов пула по нескольким скорам (см. _select_candidates).
  4. Для кандидатов — ~50 признаков (FEATURES), по которым ранкер выбирает итоговые 50.

Имена признаков совпадают с именами в моделях models/*.txt — не переименовывайте их.
"""
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp

from . import config as C
from .item_index import haversine

FEATURES = [
    # фразовые совпадения
    'phrase', 'exact', 'bigr',
    # география относительно пула кандидатов
    'dist_rk', 'dist_rel',
    # ранги и доли от максимума внутри пула: заголовок, расширение, символы, описание, число отзывов
    'rk_t', 'rel_t', 'rk_e', 'rel_e', 'rk_c', 'rel_c', 'rk_d', 'rel_d', 'rk_n',
    # подкатегория
    'mc_ctr',
    # текстовые скоры
    's_t', 's_p', 's_d', 's_a', 's_tp', 'cov_tp', 'cov_d', 'cov_all', 'cov_sp', 's_ch', 's_exp',
    # сигналы от похожих запросов трейна
    'clk', 'mc_knn', 'mc_prf', 'stage_rank',
    # локация
    'same_loc', 'trans_p', 'trans_tot', 'dist', 'loc_cnt', 'pool',
    # сам запрос
    'ntok', 'nt_known', 'nbr1', 'qcount',
    # само объявление
    'title_len', 'desc_len', 'rating', 'nrev', 'price', 'phone_h', 'msg_f', 'has_sp',
]
EMB_FEATURES = ['s_emb', 'rk_emb', 'emb_gap']


def _pool(r, index, stats, restrict):
    """Индексы объявлений в географическом пуле запроса и центр поиска."""
    sl = r.search_location_id
    trans = stats.trans.get(sl, {})
    center = stats.search_center.get(sl, index.loc_center.get(sl))
    locs = set(trans) | {sl}
    if center is not None:
        d = haversine(center[0], center[1], index.loc_lat, index.loc_lon)
        locs |= set(index.loc_ids[d < C.GEO_RADIUS_KM])
    parts = [index.loc_items[l] for l in locs if l in index.loc_items]
    pool = np.concatenate(parts) if parts else np.arange(index.n)
    if restrict is not None:
        pool = pool[restrict[pool]]
    if len(pool) < C.MIN_POOL:  # локация неизвестна или почти пуста — ищем везде
        pool = np.arange(index.n) if restrict is None else np.where(restrict)[0]
    return pool, center, trans


def _expansion_score(nidx, nsim, index, stats):
    """Расширение запроса: термины из объявлений, выбранных по похожим запросам (вес — сходство²)."""
    w = np.maximum(nsim, 0) ** 2
    ev = sp.csr_matrix(w[None, :]) @ stats.group_terms[nidx]
    if not ev.nnz:
        return np.zeros(index.n, np.float32)
    ev = ev.toarray().ravel()
    if ev.sum() <= 0:
        return np.zeros(index.n, np.float32)
    top = np.argsort(-ev)[:C.EXPANSION_TERMS]
    top = top[ev[top] > 0]
    return index.title_params.weighted_score(top, ev[top] / ev[top].sum())


class _EmbScores:
    """Косинусные скоры «запрос — все объявления корпуса», посчитанные пачками.

    Наивный вариант `index.emb[pool] @ qe` для каждого запроса собирает из строк пула
    новую матрицу: при пуле в 90 тыс. объявлений и 1024 измерениях это 368 МБ копирования
    на один запрос — дороже самого умножения. Здесь на CHUNK запросов делается одно
    матричное умножение, то есть корпус читается один раз на всю пачку.
    Результат тот же, меняется только порядок вычислений.
    """

    CHUNK = 128  # 515 тыс. объявлений × 128 запросов × 4 байта ≈ 264 МБ на блок

    def __init__(self, index, queries):
        self.emb = index.emb
        self.vecs = [index.query_emb.get(q) for q in queries.search_query]
        self.lo, self.hi, self.block = 0, 0, None

    def __getitem__(self, qi):
        """Столбец скоров по всему корпусу для запроса qi (None, если нет вектора)."""
        if self.vecs[qi] is None:
            return None
        if not self.lo <= qi < self.hi:  # запросы перебираются по порядку, блок считается один раз
            self.lo = qi - qi % self.CHUNK
            self.hi = min(self.lo + self.CHUNK, len(self.vecs))
            Q = np.zeros((self.hi - self.lo, self.emb.shape[1]), np.float32)
            for j, v in enumerate(self.vecs[self.lo:self.hi]):
                if v is not None:
                    Q[j] = v
            self.block = self.emb @ Q.T
        return self.block[:, qi - self.lo]


def _select_candidates(pool, K, s, n_tokens, index, use_emb):
    """Объединение топов пула по нескольким скорам — так в кандидаты попадают
    объявления, найденные разными способами (разные «каналы» поиска)."""
    base = s['t'][pool] + 0.5 * s['p'][pool] + 0.3 * s['d'][pool]
    stage = base + C.EXPANSION_WEIGHT * s['exp'][pool]
    # «популярные»: сначала все слова запроса есть в заголовке/параметрах, затем по числу отзывов
    full = (s['cov_tp'][pool] >= max(n_tokens, 1)).astype(np.float32)
    popular = full * 1e6 + np.nan_to_num(index.n_reviews[pool]) + base * 1e-3
    channels = [(stage, K), (base, K // 3), (s['exp'][pool], K // 3), (s['ch'][pool], K // 5), (popular, K // 4)]
    if use_emb:
        channels.append((s['emb'][pool], K // 2))
    sel = set()
    for score, k in channels:
        k = min(k, len(pool) - 1)
        sel.update(np.argpartition(-score, k)[:k].tolist())
    return pool[np.array(sorted(sel))]


def generate(queries, index, stats, K, restrict=None, with_labels=True, neg_keep=None, seed=0, verbose=True):
    """queries — DataFrame групп (search_query, search_location_id, query_n, sparams_n[, pos]).

    restrict  — булев массив по корпусу: где разрешено искать (для бенчмарка — in_bench).
    neg_keep  — если задано, негативы глубже 60-й позиции стартового скора прореживаются
                с этой вероятностью. По экспериментам прореживание ухудшает ранкер,
                поэтому по умолчанию выключено.
    Возвращает DataFrame: q (номер запроса), iidx (номер объявления), признаки[, y].
    """
    rs = np.random.RandomState(seed)
    nbrs = stats.neighbors(queries)
    use_emb = index.emb is not None
    emb_scores = _EmbScores(index, queries) if use_emb else None
    frames = []
    t0 = time.time()
    for qi, r in enumerate(queries.itertuples()):
        toks = list(dict.fromkeys(r.query_n.split()))
        stoks = list(dict.fromkeys(r.sparams_n.split()))
        sl = r.search_location_id
        pool, center, trans = _pool(r, index, stats, restrict)

        # ---------- скоры по всему корпусу
        s = {}
        s['t'], n_known = index.title.score(toks)
        s['p'], _ = index.params.score(toks)
        s['d'], _ = index.desc.score(toks)
        s['a'], _ = index.addr.score(toks)
        s['tp'], _ = index.title_params.score(toks)
        s['cov_tp'] = index.title_params.coverage(toks)
        s['cov_d'] = index.desc.coverage(toks)
        s['cov_sp'] = (index.params.coverage(stoks) / max(len(stoks), 1)) if stoks else np.full(index.n, -1, np.float32)
        s['ch'] = index.char_score(r.search_query)
        nidx, nsim = nbrs[qi]
        s['exp'] = _expansion_score(nidx, nsim, index, stats)
        qe = emb_scores[qi] if use_emb else None
        s['emb'] = np.zeros(index.n, np.float32)
        if qe is not None:
            s['emb'][:] = -1
            s['emb'][pool] = qe[pool]

        # объявления, выбранные по похожим запросам (вес — сходство)
        clk = sp.csr_matrix(nsim[None, :].astype(np.float32)) @ stats.group_items[nidx]
        clicked = dict(zip(clk.indices, clk.data))
        # распределение подкатегорий у похожих запросов
        mc_knn = defaultdict(float)
        for gi, sim in zip(nidx, nsim):
            mcs = stats.group_microcats[gi]
            for m in mcs:
                mc_knn[m] += sim / len(mcs)
        mc_total = sum(mc_knn.values()) or 1

        # ---------- кандидаты
        cand = _select_candidates(pool, K, s, len(toks), index, qe is not None)
        in_cand = set(cand)
        extra = [c for c in clicked if c not in in_cand and (restrict is None or restrict[c])]
        if extra:
            cand = np.concatenate([cand, np.array(extra, int)])

        # ---------- признаки
        stage_c = s['t'][cand] + 0.5 * s['p'][cand] + 0.3 * s['d'][cand] + C.EXPANSION_WEIGHT * s['exp'][cand]
        stage_rank = np.argsort(np.argsort(-stage_c))
        # pseudo-relevance feedback: доли подкатегорий среди верхних кандидатов
        prf = pd.Series(index.microcat[cand[stage_rank < C.TOP_FOR_PRF]]).value_counts(normalize=True).to_dict()
        mc = index.microcat[cand]
        il = index.loc[cand]
        trans_total = stats.trans_total.get(sl, 0)
        dist = (haversine(center[0], center[1], index.lat[cand], index.lon[cand])
                if center is not None else np.full(len(cand), np.nan))
        dist_filled = np.nan_to_num(dist, nan=1e4)

        def pool_rank(v):
            """Позиция кандидата среди всего пула по скору v и доля от максимума."""
            srt = np.sort(v[pool])[::-1]
            vc = v[cand]
            return np.searchsorted(-srt, -vc, side='left').astype(np.float32), vc / (srt[0] + 1e-6)

        rk_t, rel_t = pool_rank(s['t'])
        rk_e, rel_e = pool_rank(s['exp'])
        rk_c, rel_c = pool_rank(s['ch'])
        rk_d, rel_d = pool_rank(s['d'])
        rk_n, _ = pool_rank(np.nan_to_num(index.n_reviews))

        qn = r.query_n
        bigrams = [a + ' ' + b for a, b in zip(toks, toks[1:])]
        n_tok = max(len(toks), 1)
        f = {
            'q': qi, 'iidx': cand,
            'phrase': (np.array([qn in index.title_str[c] for c in cand], np.float32) if qn
                       else np.zeros(len(cand), np.float32)),
            'exact': np.array([qn == index.title_str[c] for c in cand], np.float32),
            'bigr': (np.array([sum(b in index.title_params_str[c] for b in bigrams) for c in cand], np.float32) / len(bigrams)
                     if bigrams else np.full(len(cand), -1, np.float32)),
            'dist_rk': np.argsort(np.argsort(dist_filled)).astype(np.float32) / len(cand),
            'dist_rel': (dist_filled / (np.median(dist_filled) + 1)).astype(np.float32),
            'rk_t': rk_t, 'rel_t': rel_t, 'rk_e': rk_e, 'rel_e': rel_e, 'rk_c': rk_c, 'rel_c': rel_c,
            'rk_d': rk_d, 'rel_d': rel_d, 'rk_n': rk_n,
            'mc_ctr': stats.mc_ctr.reindex(mc).values.astype(np.float32),
            's_t': s['t'][cand], 's_p': s['p'][cand], 's_d': s['d'][cand], 's_a': s['a'][cand], 's_tp': s['tp'][cand],
            'cov_tp': s['cov_tp'][cand] / n_tok,
            'cov_d': s['cov_d'][cand] / n_tok,
            'cov_all': np.maximum(s['cov_tp'][cand], s['cov_d'][cand]) / n_tok,
            'cov_sp': s['cov_sp'][cand], 's_ch': s['ch'][cand], 's_exp': s['exp'][cand],
            'clk': np.array([clicked.get(c, 0) for c in cand], np.float32),
            'mc_knn': np.array([mc_knn.get(m, 0) / mc_total for m in mc], np.float32),
            'mc_prf': np.array([prf.get(m, 0) for m in mc], np.float32),
            'stage_rank': stage_rank.astype(np.float32),
            'same_loc': (il == sl).astype(np.float32),
            'trans_p': np.array([trans.get(l, 0) for l in il], np.float32) / max(trans_total, 1),
            'trans_tot': np.float32(trans_total),
            'dist': dist.astype(np.float32),
            'loc_cnt': index.loc_count.reindex(il).values.astype(np.float32),
            'pool': np.float32(len(pool)),
            'ntok': np.float32(len(toks)), 'nt_known': np.float32(n_known),
            'nbr1': np.float32(nsim[0]), 'qcount': np.float32(stats.query_count.get(r.search_query, 0)),
            'title_len': index.title_len[cand], 'desc_len': index.desc_len[cand],
            'rating': index.rating[cand], 'nrev': index.n_reviews[cand], 'price': index.price[cand],
            'phone_h': index.phone_hidden[cand], 'msg_f': index.msg_forbidden[cand],
            'has_sp': np.float32(len(stoks) > 0),
        }
        if use_emb:
            rk_m, _ = pool_rank(s['emb'])
            k50 = min(C.TOP_N, len(pool))
            f.update({'s_emb': s['emb'][cand], 'rk_emb': rk_m,
                      'emb_gap': s['emb'][cand] - np.float32(np.partition(s['emb'][pool], -k50)[-k50])})
        f = pd.DataFrame(f)
        if with_labels:
            f['y'] = np.isin(cand, r.pos).astype(np.int8)
            if neg_keep is not None:
                f = f[(f.y == 1) | (f.stage_rank < 60) | (rs.rand(len(f)) < neg_keep)]
        fc = f.select_dtypes('float64').columns
        f[fc] = f[fc].astype(np.float32)
        frames.append(f)
        if verbose and qi and qi % 1000 == 0:
            print(f'  {qi}/{len(queries)} запросов, {time.time() - t0:.0f} c', flush=True)
    return pd.concat(frames, ignore_index=True)


def feature_columns(df):
    """Список признаков, присутствующих в таблице (с эмбеддингами или без)."""
    return FEATURES + [c for c in EMB_FEATURES if c in df.columns]