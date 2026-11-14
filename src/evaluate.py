"""학습 및 모델 선택에 사용하지 않은 테스트셋으로 최종 평가합니다."""
import json

import tensorflow as tf

from src.config import CLASS_NAMES_PATH, MODEL_PATH, OUTPUTS_DIR
from src.dataset import load_test_dataset


def main():
    if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
        raise FileNotFoundError("먼저 python -m src.train 명령으로 모델을 학습하세요.")

    class_names = json.loads(CLASS_NAMES_PATH.read_text(encoding="utf-8"))
    test_ds = load_test_dataset(class_names)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    model.compile(
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )
    result = model.evaluate(test_ds, return_dict=True)
    metrics = {k: float(v) for k, v in result.items()}
    metrics["class_names"] = class_names

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    target = OUTPUTS_DIR / "metrics.json"
    target.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"테스트 정확도: {metrics['accuracy']:.2%}")
    print(f"결과 저장: {target}")


if __name__ == "__main__":
    main()