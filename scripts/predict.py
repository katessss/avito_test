"""Шаг 4. Предсказание для бенчмарка → answer.csv (~20 мин на 2 CPU при K=4000).

    python -m scripts.predict --models models/ranker_bag0.txt,models/ranker_bag1.txt,models/ranker_bag2.txt
"""
import argparse

import pandas as pd

from candgen import config as C
from candgen.pipeline import Context, check_answer, predict_benchmark

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--models', default=','.join(str(C.MODELS_DIR / f'ranker_bag{i}.txt') for i in range(C.N_BAGS)))
    ap.add_argument('--k', type=int, default=C.K_PREDICT)
    ap.add_argument('--out', default='answer.csv')
    a = ap.parse_args()
    ctx = Context()
    ans = predict_benchmark(ctx, a.models.split(','), K=a.k)
    check_answer(ans, ctx.bench, pd.read_parquet(C.BENCH_ITEMS_FILE, columns=['item_id']))
    ans.to_csv(a.out, index=False)
    print(f'{a.out}: {len(ans)} строк, формат проверен')
