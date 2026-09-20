"""Все пути и гиперпараметры проекта в одном месте.

Пути можно переопределить переменными окружения:
    CANDGEN_DATA_DIR  — папка с исходными parquet-файлами (по умолчанию ./data)
    CANDGEN_WORK_DIR  — папка для промежуточных артефактов (по умолчанию ./artifacts)
"""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------------- 
# ПУТИ
DATA_DIR = Path(os.environ.get('CANDGEN_DATA_DIR', ROOT / 'data'))
WORK_DIR = Path(os.environ.get('CANDGEN_WORK_DIR', ROOT / 'artifacts'))
MODELS_DIR = ROOT / 'models'
CUDA_VISIBLE_DEVICES=2

BENCH_ITEMS_FILE = DATA_DIR / 'benchmark_items.parquet'
BENCH_QUERIES_FILE = DATA_DIR / 'benchmark_queries.parquet'
# Если есть полный train.parquet — берём его, иначе вырезку train_demo.parquet.
TRAIN_FILE = DATA_DIR / 'train.parquet'
if not TRAIN_FILE.exists():
    TRAIN_FILE = DATA_DIR / 'train_demo.parquet'

# результаты prepare_data.py
ITEMS_PKL = WORK_DIR / 'items.pkl'          # корпус: объявления бенчмарка + объявления трейна, которых нет в бенчмарке
TRAIN_PKL = WORK_DIR / 'train_queries.pkl'  # строки трейна (признаки запроса + item_id)
BENCH_PKL = WORK_DIR / 'bench_queries.pkl'  # запросы бенчмарка
FEATURES_DIR = WORK_DIR / 'features'
EMB_DIR = WORK_DIR / 'embeddings'           # опционально, см. scripts/compute_embeddings.py

# ---------------------------------------------------------------- 
# ДАННЫЕ
# Ключ «группы» — уникальная поисковая ситуация. Все объявления, выбранные
# в одной группе, считаются релевантными ей одновременно (как в бенчмарке).
QUERY_KEY = ['search_query', 'search_location_id', 'search_infm_params_text']

N_FOLDS = 5
FOLD_SEED = 42
# Доля групп, которые переносятся в случайный фолд независимо от текста запроса.
# Так часть валидационных запросов «уже встречалась» в трейне — как в бенчмарке
# (13.7% запросов бенчмарка есть в train_demo).
SEEN_QUERY_SHARE = 0.15
VAL_FOLD = 0

# ---------------------------------------------------------------- 
# КАНДИДАТЫ
DESC_MAX_CHARS = 1500    # сколько символов описания лемматизировать
GEO_RADIUS_KM = 30       # локации в этом радиусе от центра поиска попадают в пул
MIN_POOL = 200           # если пул меньше — ищем по всему корпусу
N_NEIGHBORS = 30         # число похожих запросов из трейна
EXPANSION_TERMS = 40     # сколько терминов берём в расширение запроса
EXPANSION_WEIGHT = 3.0   # вес расширения в стартовом скоре
TOP_FOR_PRF = 50         # по скольким верхним кандидатам считаем распределение подкатегорий

K_TRAIN = 800            # размер списка кандидатов при обучении ранкера
K_PREDICT = 4000         # размер списка кандидатов при предсказании (см. REPORT.md, раздел 7)
TOP_N = 50               # сколько объявлений отдаём в ответ

# ---------------------------------------------------------------- 
# РАНКЕР
LGB_PARAMS = dict(
    objective='lambdarank',
    learning_rate=0.04,
    num_leaves=31,
    min_data_in_leaf=300,
    feature_fraction=0.6,
    bagging_fraction=0.8,
    bagging_freq=1,
    lambda_l2=10.0,
    lambdarank_truncation_level=60,
    verbose=-1,
    num_threads=32,
)
N_TREES_TRAIN = 700      # сколько деревьев строим
N_TREES_PREDICT = 450    # сколько используем (лучшее число по валидации)
GROUPS_PER_BAG = 10_000    # групп в одном обучающем наборе 
N_BAGS = 3
