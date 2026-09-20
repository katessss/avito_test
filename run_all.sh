#!/usr/bin/env bash
# Полный пайплайн с нуля. На 2 CPU / 8 ГБ RAM занимает ~2 часа
# (сборка признаков ~40 мин, обучение 3 моделей ~45 мин, предсказание ~20 мин).
# Чтобы только получить answer.csv готовыми моделями из models/, достаточно шагов 1 и 4.
set -euo pipefail
cd "$(dirname "$0")"

# 1. подготовка данных
python -m scripts.prepare_data

# 1b. эмбеддинги для векторного канала (HuggingFace, GPU)
#     модель и видеокарту можно поменять: EMB_MODEL=... EMB_DEVICE=cuda:1 ./run_all.sh
python -m scripts.compute_embeddings --model "${EMB_MODEL:-intfloat/multilingual-e5-large}" --device "${EMB_DEVICE:-cuda:1}"

# 2. признаки: валидация (фолд 0), честная валидация, 3 непересекающихся обучающих набора (фолды 1–4)
python -m scripts.build_features --name val --folds 0 --groups 3000
python -m scripts.build_features --honest
for b in 0 1 2; do
  python -m scripts.build_features --name bag$b --folds 1,2,3,4 --groups 1750 --bag $b
done

# 3. обучение ранкеров
for b in 0 1 2; do
  python -m scripts.train_ranker --name bag$b --out models/ranker_bag$b.txt
done

# оценка
MODELS=models/ranker_bag0.txt,models/ranker_bag1.txt,models/ranker_bag2.txt
python -m scripts.evaluate --models $MODELS --features val_fold0,honest

# 4. ответ
python -m scripts.predict --models $MODELS --out answer.csv