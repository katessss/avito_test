"""Шаг 1b. Эмбеддинги объявлений и запросов для векторного канала поиска.

Нужен доступ к HuggingFace и sentence-transformers. Запускать после prepare_data (run_all.sh делает это сам).

    python -m scripts.compute_embeddings --model intfloat/multilingual-e5-base

Результат — в artifacts/embeddings/: items.npy (строки в порядке items.pkl), queries.npy,
queries_text.pkl. Если эти файлы есть, ItemIndex подхватывает их автоматически:
в кандидаты добавляется канал «топ-K/2 по косинусу», в признаки — s_emb, rk_emb, emb_gap.
После этого нужно заново собрать признаки и обучить ранкеры (шаги 2–3):
модели в models/ обучены без векторных признаков.

Модели на выбор: intfloat/multilingual-e5-large, intfloat/multilingual-e5-base (по умолчанию),  
intfloat/multilingual-e5-base (быстро на CPU), deepvk/USER-bge-m3 (ИНОГДА сильнее на русском, медленнее; префиксы не нужны).
"""
import argparse

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from candgen import config as C

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='intfloat/multilingual-e5-base')
    ap.add_argument('--batch', type=int, default=128)
    ap.add_argument('--max-len', type=int, default=256)
    ap.add_argument('--device', default=None, help='cuda, cuda:1, mps или cpu; по умолчанию — GPU, если есть')
    a = ap.parse_args()
    

    # e5 обучены с префиксами «query: » / «passage: »
    q_prefix, d_prefix = ('query: ', 'passage: ') if 'e5' in a.model else ('', '')

    items = pd.read_pickle(C.ITEMS_PKL)
    train = pd.read_pickle(C.TRAIN_PKL)
    bench = pd.read_pickle(C.BENCH_PKL)
    # описание в items.pkl уже лемматизировано — для модели берём исходный текст
    # бенчмарк первым: при дубликатах item_id побеждает его описание
    raw = pd.concat([pd.read_parquet(f, columns=['item_id', 'item_description_raw'])
                     for f in (C.BENCH_ITEMS_FILE, C.TRAIN_FILE)])
    raw = raw.drop_duplicates('item_id')
    desc = items[['item_id']].merge(raw, on='item_id', how='left').item_description_raw
    desc = desc.fillna('').astype(str).str.slice(0, 300)
    params = items.item_infm_params_text.fillna('').astype(str).str.replace(
        r'Место оказания услуг.*?(?= Тип стоимости| Тип услуги| Вид услуги| Куда выезжаете|$)', ' ', regex=True)
    titles = items.item_title_raw.fillna('').astype(str)
    docs = [f"{d_prefix}{t}. {p[:200]}. {d}" for t, p, d in zip(titles, params, desc)]
    del raw, desc

    model = SentenceTransformer(a.model, device=a.device)
    model.max_seq_length = a.max_len
    if str(model.device).startswith('cuda'):
        model.half()  # fp16 на GPU: вдвое быстрее, на качество косинусов не влияет
    print(f'модель {a.model} на {model.device}', flush=True)
    C.EMB_DIR.mkdir(parents=True, exist_ok=True)
    E = model.encode(docs, batch_size=a.batch, normalize_embeddings=True, show_progress_bar=True)
    np.save(C.EMB_DIR / 'items.npy', E.astype(np.float16))

    queries = sorted({str(q) for q in pd.concat([train.search_query, bench.search_query]).dropna()})
    Q = model.encode([q_prefix + q for q in queries], batch_size=a.batch * 2,
                     normalize_embeddings=True, show_progress_bar=True)
    np.save(C.EMB_DIR / 'queries.npy', Q.astype(np.float16))
    pd.to_pickle(queries, C.EMB_DIR / 'queries_text.pkl')
    print('готово:', E.shape, Q.shape)