"""Streamlit에서 재사용하는 단일 이미지 추론 함수입니다."""
import json

import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

from src.config import CLASS_NAMES_PATH, IMAGE_SIZE, MODEL_PATH


def load_artifacts():
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    return model, class_names


def predict_image(model, class_names: list[str], image: Image.Image) -> list[tuple[str, float]]:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = image.resize((IMAGE_SIZE[1], IMAGE_SIZE[0]))
    # 학습 때와 동일하게 0~255 범위를 모델에 전달합니다.
    batch = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)
    probabilities = model(batch, training=False).numpy()[0]
    if len(probabilities) != len(class_names):
        raise ValueError("모델 출력 클래스 수와 class_names.json이 일치하지 않습니다.")
    results = [(name, float(score)) for name, score in zip(class_names, probabilities)]
    return sorted(results, key=lambda item: item[1], reverse=True)