"""Шаг 2. Кандидаты и признаки для обучения/валидации.

Примеры:
    # валидация: 3000 групп из фолда 0
    python -m scripts.build_features --name val --folds 0 --groups 3000
    # «честная» валидация (только объявления бенчмарка)
    python -m scripts.build_features --honest
    # обучающие наборы для бэггинга: по 1750 групп из фолдов 1–4, срезы 0, 1, 2
    python -m scripts.build_features --name bag0 --folds 1,2,3,4 --groups 1750 --bag 0
    # тот же набор, но с большим числом кандидатов (для проверки K при предсказании)
    python -m scripts.build_features --name val_k4000 --folds 0 --groups 700 --k 4000
"""
import argparse

from candgen import config as C
from candgen.pipeline import Context, build_fold_features, build_honest_features

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='val')
    ap.add_argument('--folds', default='0', help='номера фолдов через запятую')
    ap.add_argument('--groups', type=int, default=3000, help='групп из каждого фолда')
    ap.add_argument('--bag', type=int, default=0, help='номер непересекающегося среза групп')
    ap.add_argument('--k', type=int, default=C.K_TRAIN, help='размер списка кандидатов')
    ap.add_argument('--neg-keep', type=float, default=None, help='доля глубоких негативов (по умолчанию все)')
    ap.add_argument('--honest', action='store_true', help='собрать честную валидацию вместо фолдов')
    a = ap.parse_args()

    ctx = Context()
    if a.honest:
        print(build_honest_features(ctx, K=a.k))
    else:
        folds = [int(x) for x in a.folds.split(',')]
        print(build_fold_features(ctx, a.name, folds, a.groups, a.bag, a.k, a.neg_keep))
