import streamlit as st
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from src.config import MODEL_PATH, CLASS_NAMES_PATH
from src.predict import load_artifacts, predict_image


# 웹앱 설정
st.set_page_config(
    page_title="AI 그림 맞히기",
    page_icon="🎨"
)

st.title("🎨 AI 그림 맞히기")
st.write("아래 그림판에 그림을 그려보세요!")


# 학습된 AI 모델 확인
if not MODEL_PATH.exists() or not CLASS_NAMES_PATH.exists():
    st.error("학습된 모델이 없습니다. 먼저 모델을 학습하세요.")
    st.stop()


# 모델 로드 (캐시 적용)
@st.cache_resource
def load_model(model_version, classes_version):
    return load_artifacts()


model, class_names = load_model(
    MODEL_PATH.stat().st_mtime_ns,
    CLASS_NAMES_PATH.stat().st_mtime_ns
)


# 펜 굵기 조절
stroke_width = st.slider(
    "펜 굵기",
    min_value=1,
    max_value=30,
    value=8
)


# 그림판 생성
canvas_result = st_canvas(
    stroke_width=stroke_width,
    stroke_color="#000000",
    background_color="#FFFFFF",
    drawing_mode="freedraw",
    width=300,
    height=300,
    update_streamlit=True,
    return_image_data=True,
    key="canvas"
)


# AI 예측 버튼
if st.button("AI로 예측하기", type="primary"):

    image_data = canvas_result.image_data
    drawing_data = canvas_result.json_data

    # 아무것도 그리지 않았는지 확인
    if (
        image_data is None
        or not drawing_data
        or not drawing_data.get("objects")
    ):
        st.warning("먼저 그림을 그려주세요!")

    else:
        # 캔버스 데이터를 이미지로 변환
        image_array = np.asarray(
            image_data, dtype=np.uint8
        )

        rgba_image = Image.fromarray(
            image_array, mode="RGBA"
        )

        # 투명한 부분을 흰색으로 합성
        background = Image.new(
            "RGBA", rgba_image.size, "white"
        )
        background.alpha_composite(rgba_image)

        image = background.convert("RGB")

        # AI 예측 실행
        results = predict_image(
            model,
            class_names,
            image
        )

        # 결과 출력
        predicted_class = results[0][0]

        st.subheader("AI 예측 결과")
        st.success(f"AI의 예측: {predicted_class}")

        st.write("클래스별 예측 점수")

        for name, score in results:
            st.write(f"{name}: {score:.1%}")
            st.progress(float(score))
