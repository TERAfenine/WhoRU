"""학습/검증/테스트 폴더를 TensorFlow 데이터셋으로 변환합니다."""
import tensorflow as tf

from src.config import BATCH_SIZE, IMAGE_SIZE, PROCESSED_DIR, SEED


def load_datasets():
    common = dict(
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        color_mode="rgb",
    )
    train_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DIR / "train", shuffle=True, seed=SEED, **common
    )
    class_names = list(train_ds.class_names)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DIR / "val", class_names=class_names, shuffle=False, **common
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DIR / "test", class_names=class_names, shuffle=False, **common
    )
    auto = tf.data.AUTOTUNE
    return train_ds.prefetch(auto), val_ds.prefetch(auto), test_ds.prefetch(auto), class_names


def load_test_dataset(class_names: list[str]):
    ds = tf.keras.utils.image_dataset_from_directory(
        PROCESSED_DIR / "test",
        class_names=class_names,
        image_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="int",
        color_mode="rgb",
        shuffle=False,
    )
    return ds.prefetch(tf.data.AUTOTUNE)