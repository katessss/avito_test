"""Шаг 1. Нормализация и лемматизация корпуса и запросов (~2 мин).

    python -m scripts.prepare_data
"""
from candgen.data import prepare

if __name__ == '__main__':
    prepare()
