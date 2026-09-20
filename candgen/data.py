"""Загрузка и подготовка данных, группировка запросов, разбиение на фолды."""
import numpy as np
import pandas as pd

from . import config as C
from .text import normalize, split_params


# ---------------------------------------------------------------- 
#  подготовка
def prepare():
    """Сырые parquet → нормализованные pickle в WORK_DIR.

    Корпус для поиска = объявления бенчмарка + объявления из трейна, которых
    нет в бенчмарке. Вторые нужны только для валидации: 93% выбранных в трейне
    объявлений отсутствуют в benchmark_items, и без них по трейн-запросам
    нечего было бы искать. При предсказании для бенчмарка поиск ограничен
    объявлениями бенчмарка (колонка in_bench).
    """
    C.WORK_DIR.mkdir(parents=True, exist_ok=True)
    train = pd.read_parquet(C.TRAIN_FILE)
    bench_items = pd.read_parquet(C.BENCH_ITEMS_FILE)
    bench_q = pd.read_parquet(C.BENCH_QUERIES_FILE)
    print(f'train: {len(train)} строк из {C.TRAIN_FILE.name}; корпус: {len(bench_items)}; запросы: {len(bench_q)}')

    train_only = train[bench_items.columns].drop_duplicates('item_id')
    train_only = train_only[~train_only.item_id.isin(bench_items.item_id)]
    items = pd.concat([bench_items.assign(in_bench=1), train_only.assign(in_bench=0)], ignore_index=True)

    params_addr = items.item_infm_params_text.map(split_params)
    items['params_n'] = [normalize(p) for p, _ in params_addr]
    items['addr_n'] = [normalize(a) for _, a in params_addr]
    items['title_n'] = items.item_title_raw.map(normalize)
    items['desc_n'] = items.item_description_raw.map(lambda s: normalize(s, C.DESC_MAX_CHARS))
    for col in ['item_latitude', 'item_longitude', 'item_price']:  # в исходнике это строки-decimal
        items[col] = pd.to_numeric(items[col], errors='coerce')
    items = items.drop(columns=['item_description_raw'])
    items.to_pickle(C.ITEMS_PKL)
    print(f'корпус с train-объявлениями: {len(items)}')

    for df, path in [(train, C.TRAIN_PKL), (bench_q, C.BENCH_PKL)]:
        df = df.copy()
        df['query_n'] = df.search_query.map(normalize)
        df['sparams_n'] = df.search_infm_params_text.map(normalize)
        keep = [c for c in df.columns if c.startswith('search_') or c in ('query_n', 'sparams_n', 'query_id', 'item_id')]
        df[keep].to_pickle(path)


# ----------------------------------------------------------------
#  загрузка
def load():
    """Возвращает (items, train, bench_queries). В train добавлены iidx — номер
    объявления в items — и его локация/подкатегория."""
    items = pd.read_pickle(C.ITEMS_PKL)
    train = pd.read_pickle(C.TRAIN_PKL)
    bench = pd.read_pickle(C.BENCH_PKL)
    id2idx = pd.Series(np.arange(len(items)), index=items.item_id)
    train['iidx'] = id2idx.reindex(train.item_id).values
    train['item_location_id'] = items.item_location_id.values[train.iidx]
    train['item_microcat_id'] = items.item_microcat_id.values[train.iidx]
    return items, train, bench


def group_queries(rows):
    """Строки «запрос—объявление» → одна строка на группу со списком релевантных iidx (pos)."""
    g = rows.groupby(C.QUERY_KEY, sort=False)
    return g.agg(query_n=('query_n', 'first'),
                 sparams_n=('sparams_n', 'first'),
                 search_category=('search_category', 'first'),
                 pos=('iidx', lambda x: sorted(set(x)))).reset_index()


def assign_folds(train):
    """Разбиение групп на фолды по тексту запроса (+15% групп в случайный фолд).

    Разбиение по тексту имитирует главное свойство бенчмарка: большинство его
    запросов в трейне не встречались. Возвращает (groups с колонкой fold,
    train со столбцом fold).
    """
    groups = group_queries(train)
    rng = np.random.RandomState(C.FOLD_SEED)
    uniq = groups.search_query.unique()
    qfold = dict(zip(uniq, rng.randint(0, C.N_FOLDS, len(uniq))))
    fold = groups.search_query.map(qfold).values.copy()
    moved = rng.rand(len(groups)) < C.SEEN_QUERY_SHARE
    fold[moved] = rng.randint(0, C.N_FOLDS, moved.sum())
    groups['fold'] = fold
    train = train.merge(groups[C.QUERY_KEY + ['fold']], on=C.QUERY_KEY)
    return groups, train
