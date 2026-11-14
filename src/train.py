"""기본 전이학습 후, 옵션으로 상위 층을 파인튜닝합니다."""
import argparse
import json

import tensorflow as tf

from src.config import (
    CLASS_NAMES_PATH, FINE_TUNE_EPOCHS, HEAD_EPOCHS, MODEL_PATH, MODELS_DIR, SEED
)
from src.dataset import load_datasets
from src.model import build_model


def compile_model(model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
    )


def early_stopping():
    return tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=2, restore_best_weights=True
    )


def main(fine_tune: bool = False) -> None:
    tf.keras.utils.set_random_seed(SEED)
    train_ds, val_ds, _test_ds, class_names = load_datasets()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    CLASS_NAMES_PATH.write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    model, base_model = build_model(num_classes=len(class_names))
    compile_model(model, learning_rate=1e-3)
    print("[1단계] 사전학습 본체 고정, 분류층 학습")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=HEAD_EPOCHS,
        callbacks=[early_stopping()],
    )
    best_val_loss = float(model.evaluate(val_ds, verbose=0, return_dict=True)["loss"])
    model.save(MODEL_PATH)
    print(f"기본 모델 저장: {MODEL_PATH} (val_loss={best_val_loss:.4f})")

    if fine_tune:
        print("[2단계] 상위 30개 층 파인튜닝")
        base_model.trainable = True
        for layer in base_model.layers[:-30]:
            layer.trainable = False
        for layer in base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False

        # trainable 속성 변경 후에는 반드시 다시 compile 합니다.
        compile_model(model, learning_rate=1e-5)
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=FINE_TUNE_EPOCHS,
            callbacks=[early_stopping()],
        )
        fine_val_loss = float(model.evaluate(val_ds, verbose=0, return_dict=True)["loss"])
        if fine_val_loss < best_val_loss:
            model.save(MODEL_PATH)
            print(f"파인튜닝 모델로 교체 (val_loss={fine_val_loss:.4f})")
        else:
            print("파인튜닝이 더 좋지 않아 기본 모델을 유지합니다.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MobileNetV2 이미지 분류 학습")
    parser.add_argument("--fine-tune", action="store_true", help="기본 학습 후 상위 층 추가 학습")
    args = parser.parse_args()
    main(fine_tune=args.fine_tune)