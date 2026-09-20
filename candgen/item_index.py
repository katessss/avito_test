"""Индекс корпуса: разреженные BM25-матрицы по полям, символьный TF-IDF,
геоданные локаций и (опционально) векторные представления.

Все матрицы хранятся в формате CSC: для запроса из 1–5 слов нужно взять
несколько столбцов и сложить их — это быстрее, чем умножать строки пула.
Сборка индекса занимает ~1 минуту.
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from . import config as C
from .text import raw_lower


def bm25_matrix(texts, k1=1.2, b=0.75, min_df=1):
    """Тексты (леммы через пробел) → матрица BM25-весов документ×термин и словарь."""
    cv = CountVectorizer(token_pattern=r'\S+', min_df=min_df, dtype=np.float32)
    X = cv.fit_transform(texts).tocsr().astype(np.float32)
    n_docs = X.shape[0]
    df = np.bincount(X.indices, minlength=X.shape[1])
    idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
    doc_len = np.asarray(X.sum(1)).ravel()
    avg_len = doc_len.mean() if doc_len.mean() > 0 else 1
    rows = np.repeat(np.arange(n_docs), np.diff(X.indptr))
    tf = X.data
    X.data = (tf * (k1 + 1) / (tf + k1 * (1 - b + b * doc_len[rows] / avg_len)) * idf[X.indices]).astype(np.float32)
    return X, cv.vocabulary_


def haversine(lat1, lon1, lat2, lon2):
    """Расстояние в км (векторизовано)."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


class TextField:
    """BM25-матрица одного поля + словарь, с операциями для запроса."""

    def __init__(self, texts, min_df=1):
        X, self.vocab = bm25_matrix(texts, min_df=min_df)
        self.X = X.tocsc()

    def score(self, tokens):
        """BM25-скор всех документов по токенам запроса; второе значение — сколько токенов нашлось в словаре."""
        ids = [self.vocab[t] for t in tokens if t in self.vocab]
        if not ids:
            return np.zeros(self.X.shape[0], np.float32), 0
        return np.asarray(self.X[:, ids] @ np.ones(len(ids), np.float32)).ravel(), len(ids)

    def weighted_score(self, term_ids, weights):
        return np.asarray(self.X[:, term_ids] @ weights.astype(np.float32)).ravel()

    def coverage(self, tokens):
        """Сколько разных токенов запроса встречается в документе."""
        ids = [self.vocab[t] for t in tokens if t in self.vocab]
        if not ids:
            return np.zeros(self.X.shape[0], np.float32)
        sub = self.X[:, ids].copy()
        sub.data[:] = 1
        return np.asarray(sub.sum(1)).ravel()


class ItemIndex:
    def __init__(self, items):
        self.items = items
        self.n = len(items)
        self.in_bench = items.in_bench.values.astype(bool)

        # ---- текстовые поля
        self.title = TextField(items.title_n.values)
        self.params = TextField(items.params_n.values)
        self.desc = TextField(items.desc_n.values, min_df=2)
        self.addr = TextField(items.addr_n.values)
        # «заголовок ×2 + параметры» — основное поле стартового поиска и расширения запроса
        tp_text = (items.title_n + ' ' + items.title_n + ' ' + items.params_n).values
        self.title_params = TextField(tp_text)
        self.title_params_csr = self.title_params.X.tocsr()
        # символьные 3–4-граммы заголовка: ловят опечатки и словоформы, которые лемматизатор не свёл
        self.char_vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 4), min_df=3,
                                        dtype=np.float32, sublinear_tf=True)
        self.char = self.char_vec.fit_transform(items.item_title_raw.str.lower().str.replace('ё', 'е').values).tocsc()
        # для точных фразовых совпадений
        self.title_str = np.array(items.title_n.tolist(), dtype=object)
        self.title_params_str = np.array(list(tp_text), dtype=object)

        # ---- атрибуты объявлений
        self.title_len = np.array([len(s.split()) for s in items.title_n.values], np.float32)
        self.desc_len = np.array([len(s.split()) for s in items.desc_n.values], np.float32)
        self.lat = items.item_latitude.values.astype(np.float64)
        self.lon = items.item_longitude.values.astype(np.float64)
        self.loc = items.item_location_id.values
        self.microcat = items.item_microcat_id.values
        self.n_reviews = items.item_rating_reviews_count.values.astype(np.float32)
        self.rating = items.item_rating.values.astype(np.float32)
        self.price = items.item_price.values.astype(np.float32)
        self.phone_hidden = items.item_is_phone_hidden.values.astype(np.float32)
        self.msg_forbidden = items.item_is_message_forbidden.values.astype(np.float32)

        # ---- локации: список объявлений и медианные координаты
        self.loc_items = items.groupby('item_location_id').indices
        centers = items.groupby('item_location_id')[['item_latitude', 'item_longitude']].median()
        self.loc_ids = centers.index.values
        self.loc_lat = centers.item_latitude.values
        self.loc_lon = centers.item_longitude.values
        self.loc_center = {l: (a, b) for l, a, b in zip(self.loc_ids, self.loc_lat, self.loc_lon)}
        self.loc_count = items.item_location_id.value_counts()

        # ---- векторы (если посчитаны scripts/compute_embeddings.py)
        self.emb, self.query_emb = None, None
        emb_path = C.EMB_DIR / 'items.npy'
        if emb_path.exists():
            self.emb = np.load(emb_path).astype(np.float32)
            assert self.emb.shape[0] == self.n, 'embeddings/items.npy не соответствует items.pkl — пересчитайте эмбеддинги'
            q = np.load(C.EMB_DIR / 'queries.npy').astype(np.float32)
            self.query_emb = dict(zip(pd.read_pickle(C.EMB_DIR / 'queries_text.pkl'), q))
            print(f'векторный поиск включён: {self.emb.shape}', flush=True)

    def char_score(self, query_raw):
        q = self.char_vec.transform([raw_lower(query_raw)])
        return np.asarray(self.char[:, q.indices] @ q.data).ravel()
