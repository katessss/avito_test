"""Шаг 3. Обучение LightGBM LambdaRank на наборе признаков.

    python -m scripts.train_ranker --name bag0 --out models/ranker_bag0.txt
"""
import argparse

from candgen import config as C
from candgen.ranker import train

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True, help='префикс файлов признаков в FEATURES_DIR')
    ap.add_argument('--out', required=True)
    ap.add_argument('--trees', type=int, default=C.N_TREES_TRAIN)
    a = ap.parse_args()
    paths = sorted(C.FEATURES_DIR.glob(f'{a.name}_fold*.parquet'))
    assert paths, f'нет файлов {a.name}_fold*.parquet в {C.FEATURES_DIR}'
    model = train(paths, a.trees)
    model.save_model(a.out)
    print('сохранено', a.out)
