import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="PotatoVision AI",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "potato_disease_model.keras"
CLASSES_PATH = BASE_DIR / "models" / "class_names.json"

IMAGE_SIZE = (256, 256)

DISEASE_INFO = {
    "Early_Blight": {
        "title": "Early Blight",
        "description": (
            "Early blight is commonly associated with dark lesions "
            "that may develop concentric rings on potato leaves."
        ),
        "icon": "◉",
    },
    "Healthy": {
        "title": "Healthy Leaf",
        "description": (
            "The model identifies this image as belonging to the "
            "healthy-leaf class."
        ),
        "icon": "✓",
    },
    "Late_Blight": {
        "title": "Late Blight",
        "description": (
            "Late blight can cause irregular, water-soaked lesions "
            "and may spread rapidly under favorable conditions."
        ),
        "icon": "◈",
    },
}


# --------------------------------------------------
# DARK THEME
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url(
        'https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap'
    );

    :root {
        --bg: #0b0f0d;
        --panel: #121815;
        --panel-light: #171f1a;
        --border: #28342c;
        --green: #a3e635;
        --green-dark: #263b20;
        --text: #edf4ee;
        --muted: #94a398;
    }

    .stApp {
        background:
            radial-gradient(
                circle at 85% 0%,
                rgba(91, 140, 55, 0.10),
                transparent 30%
            ),
            var(--bg);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    .block-container {
        max-width: 1180px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--text) !important;
    }

    p, label, span {
        color: inherit;
    }

    .brand {
        display: flex;
        align-items: center;
        gap: 12px;
        margin-bottom: 3rem;
    }

    .brand-icon {
        width: 46px;
        height: 46px;
        display: flex;
        align-items: center;
        justify-content: center;
        border-radius: 14px;
        background: var(--green);
        color: #14210e;
        font-size: 25px;
        font-weight: 800;
    }

    .brand-name {
        color: var(--text);
        font-family: 'Manrope', sans-serif;
        font-size: 22px;
        font-weight: 800;
        letter-spacing: -0.8px;
    }

    .brand-caption {
        color: var(--muted);
        font-size: 11px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    .hero-eyebrow {
        color: var(--green);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2.4px;
        text-transform: uppercase;
        margin-bottom: 12px;
    }

    .hero-title {
        font-family: 'Manrope', sans-serif;
        font-size: clamp(38px, 5vw, 62px);
        font-weight: 800;
        line-height: 1.1;
        letter-spacing: -2.5px;
        color: var(--text);
        margin-bottom: 20px;
    }

    .hero-title span {
        color: var(--green);
    }

    .hero-description {
        max-width: 570px;
        color: var(--muted);
        font-size: 16px;
        line-height: 1.8;
        margin-bottom: 28px;
    }

    .status-pill {
        display: inline-block;
        padding: 8px 13px;
        border: 1px solid var(--border);
        border-radius: 30px;
        color: #c9d9cb;
        background: rgba(20, 30, 24, 0.8);
        font-size: 12px;
    }

    .section-label {
        color: var(--green);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .section-heading {
        font-family: 'Manrope', sans-serif;
        font-size: 25px;
        font-weight: 700;
        letter-spacing: -0.7px;
        margin-bottom: 8px;
        color: var(--text);
    }

    .section-description {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.7;
        margin-bottom: 24px;
    }

    .panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 24px;
        margin-bottom: 18px;
    }

    .panel-heading {
        font-family: 'Manrope', sans-serif;
        font-size: 17px;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 7px;
    }

    .panel-caption {
        color: var(--muted);
        font-size: 13px;
        line-height: 1.7;
    }

    .result-label {
        color: var(--muted);
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.7px;
        text-transform: uppercase;
    }

    .result-title {
        color: var(--green);
        font-family: 'Manrope', sans-serif;
        font-size: 30px;
        font-weight: 800;
        letter-spacing: -1px;
        margin: 10px 0;
    }

    .metric-card {
        background: var(--panel-light);
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 17px;
        margin-top: 12px;
    }

    .metric-label {
        color: var(--muted);
        font-size: 12px;
    }

    .metric-value {
        color: var(--text);
        font-family: 'Manrope', sans-serif;
        font-size: 24px;
        font-weight: 800;
        margin-top: 5px;
    }

    .footer {
        border-top: 1px solid var(--border);
        margin-top: 55px;
        padding-top: 22px;
        color: var(--muted);
        font-size: 12px;
        line-height: 1.8;
    }

    div.stButton > button {
        background: var(--green);
        color: #17200f;
        border: 1px solid var(--green);
        border-radius: 11px;
        min-height: 48px;
        font-weight: 800;
        transition: all 0.2s ease;
    }

    div.stButton > button:hover {
        background: #b7f35b;
        border-color: #b7f35b;
        color: #17200f;
        transform: translateY(-1px);
    }

    div.stButton > button:focus {
        color: #17200f;
        border-color: var(--green);
    }

    [data-testid="stFileUploader"] {
        background: var(--panel);
        border: 1px dashed #50634f;
        border-radius: 16px;
        padding: 14px;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: #151d17;
        border: 1px dashed #465b43;
        border-radius: 12px;
    }

    [data-testid="stFileUploaderDropzone"] button {
        color: var(--text);
        border-color: var(--border);
        background: #202b22;
    }

    [data-testid="stImage"] img {
        border-radius: 14px;
    }

    [data-testid="stProgress"] > div > div {
        background-color: var(--green);
    }

    hr {
        border-color: var(--border);
    }

    @media (max-width: 700px) {
        .block-container {
            padding-top: 1.2rem;
        }

        .brand {
            margin-bottom: 2rem;
        }

        .hero-title {
            letter-spacing: -1.5px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


@st.cache_data
def load_class_names():
    with open(CLASSES_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

def predict_image(image, model, class_names):
    image = image.convert("RGB")
    image = image.resize(IMAGE_SIZE)

    image_array = tf.keras.utils.img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)

    # Do not add extra normalization here unless your
    # training pipeline requires it.
    predictions = model.predict(image_array, verbose=0)[0]

    # This application expects a three-class softmax model.
    if len(predictions) != len(class_names):
        raise ValueError(
            "The model output size does not match the class names."
        )

    if not np.all(np.isfinite(predictions)):
        raise ValueError("The model returned invalid prediction values.")

    if np.any(predictions < 0) or np.any(predictions > 1):
        raise ValueError(
            "The model output does not appear to contain probabilities. "
            "Check the final activation and training configuration."
        )

    if not np.isclose(np.sum(predictions), 1.0, atol=1e-3):
        raise ValueError(
            "The model outputs do not sum to approximately 1. "
            "Check whether the final layer uses softmax."
        )

    predicted_index = int(np.argmax(predictions))

    return (
        class_names[predicted_index],
        float(predictions[predicted_index]) * 100,
        predictions,
    )


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="brand">
        <div class="brand-icon">✳</div>
        <div>
            <div class="brand-name">PotatoVision AI</div>
            <div class="brand-caption">Computer Vision · Agriculture</div>
        </div>
    </div>

    <div class="hero-eyebrow">AI-powered plant health analysis</div>

    <div class="hero-title">
        A clearer view of<br>
        <span>plant health.</span>
    </div>

    <div class="hero-description">
        Upload a potato leaf image and explore a CNN-based
        classification of Early Blight, Late Blight, and Healthy leaves.
        Simple, fast, and designed to make machine learning accessible.
    </div>

    <div class="status-pill">
        ● &nbsp; Three-class image classification
    </div>
    """,
    unsafe_allow_html=True,
)

st.write("")
st.write("")
st.write("")


# --------------------------------------------------
# MAIN LAYOUT
# --------------------------------------------------

left, right = st.columns([1.05, 0.95], gap="large")

with left:
    st.markdown(
        """
        <div class="section-label">01 / Image input</div>
        <div class="section-heading">Upload a leaf</div>
        <div class="section-description">
            Choose a clear image of a potato leaf for analysis.
            JPG, JPEG, and PNG images are supported.
        </div>
        """,
        unsafe_allow_html=True,
    )

    uploaded_file = st.file_uploader(
        "Choose a potato leaf image",
        type=["jpg", "jpeg", "png"],
        help="Use a clear image with the leaf visible.",
    )

    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")

            st.image(
                image,
                caption=f"Selected image · {uploaded_file.name}",
                use_container_width=True,
            )

            st.markdown(
                f"""
                <div class="panel">
                    <div class="panel-heading">Image ready</div>
                    <div class="panel-caption">
                        Original dimensions: {image.width} × {image.height}px
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        except Exception:
            st.error("This image could not be opened. Please try another.")
            image = None
    else:
        image = None

        st.markdown(
            """
            <div class="panel">
                <div style="
                    font-size: 42px;
                    color: #a3e635;
                    margin-bottom: 14px;
                ">＋</div>
                <div class="panel-heading">Your image goes here</div>
                <div class="panel-caption">
                    Upload a clear potato leaf photograph to get started.
                    For best results, use a well-lit image with the leaf
                    as the main subject.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    analyze_clicked = st.button(
        "✦  Analyze leaf",
        use_container_width=True,
        disabled=image is None,
    )


with right:
    st.markdown(
        """
        <div class="section-label">02 / Analysis</div>
        <div class="section-heading">Classification results</div>
        <div class="section-description">
            Your model's predicted class and the scores for each category
            will appear here.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if analyze_clicked and image is not None:
        if not MODEL_PATH.exists() or not CLASSES_PATH.exists():
            st.warning(
                "The interface is ready, but the trained model files "
                "have not been found yet."
            )

            st.markdown(
                """
                <div class="panel">
                    <div class="panel-heading">Model setup required</div>
                    <div class="panel-caption">
                        Save your corrected model as
                        <b>models/potato_disease_model.keras</b>
                        and your class ordering as
                        <b>models/class_names.json</b>.
                        Then restart or refresh the application.
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        else:
            try:
                with st.spinner("Analyzing leaf image..."):
                    model = load_model()
                    class_names = load_class_names()

                    predicted_class, confidence, scores = predict_image(
                        image,
                        model,
                        class_names,
                    )

                details = DISEASE_INFO.get(
                    predicted_class,
                    {
                        "title": predicted_class,
                        "description": (
                            "The model predicted this category."
                        ),
                    },
                )

                st.markdown(
                    f"""
                    <div class="panel">
                        <div class="result-label">
                            Predicted classification
                        </div>
                        <div class="result-title">
                            {details["title"]}
                        </div>
                        <div class="panel-caption">
                            {details["description"]}
                        </div>
                        <div class="metric-card">
                            <div class="metric-label">
                                Model confidence score
                            </div>
                            <div class="metric-value">
                                {confidence:.2f}%
                            </div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown(
                    '<div class="panel-heading">'
                    'Class probability scores</div>',
                    unsafe_allow_html=True,
                )

                for class_name, score in sorted(
                    zip(class_names, scores),
                    key=lambda item: item[1],
                    reverse=True,
                ):
                    label = DISEASE_INFO.get(
                        class_name, {}
                    ).get("title", class_name)

                    st.markdown(
                        f"**{label}** · {float(score) * 100:.2f}%"
                    )

                    st.progress(
                        float(np.clip(score, 0.0, 1.0))
                    )

                st.caption(
                    "These scores reflect the model's outputs and "
                    "are not necessarily calibrated probabilities."
                )

                st.info(
                    "This is an AI-generated prediction, not a "
                    "definitive plant-health diagnosis."
                )

            except Exception as error:
                st.error(
                    "Prediction failed. Check the model, class names, "
                    "and preprocessing configuration."
                )

                with st.expander("Technical details"):
                    st.code(str(error))

    else:
        st.markdown(
            """
            <div class="panel">
                <div style="
                    font-size: 34px;
                    margin-bottom: 14px;
                    color: #a3e635;
                ">◎</div>
                <div class="panel-heading">Awaiting analysis</div>
                <div class="panel-caption">
                    Upload a leaf image and select Analyze leaf.
                    The predicted category and class scores will appear
                    in this section.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">SUPPORTED CLASSES</div>
                <div class="metric-value">03</div>
                <div class="panel-caption">
                    Early Blight · Healthy · Late Blight
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown(
    """
    <div class="footer">
        <b style="color: #edf4ee;">PotatoVision AI</b>
        &nbsp; · &nbsp; Deep learning project<br>
        Built with Python, TensorFlow, and Streamlit.
        Predictions are experimental and should not replace
        expert agricultural assessment.
    </div>
    """,
    unsafe_allow_html=True,
)