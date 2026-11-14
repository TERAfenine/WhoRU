"""ImageNet 사전학습 MobileNetV2에 새 분류층을 추가합니다."""
import tensorflow as tf

from src.config import IMAGE_SIZE


def build_model(num_classes: int):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(*IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="image")
    # MobileNetV2는 -1~1 범위의 픽셀값을 사용합니다.
    # 전처리를 모델에 포함하므로 웹앱에서는 /255를 따로 하지 않습니다.
    x = tf.keras.layers.Rescaling(1.0 / 127.5, offset=-1.0)(inputs)
    x = base_model(x, training=False)  # BatchNormalization 통계 보호
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs, name="image_classifier")
    return model, base_model