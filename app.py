import altair as alt
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image

from src.const.maps import IDX_TO_EMOTION
from src.const.paths import DATA_PATHS, MODELS_DIR
from src.const.transforms import VAL_TRANSFORM
from src.models.trainer import EmotionClassifier

st.set_page_config(
    page_title="Affective AI - Degradation Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

EMOTION_COLORS = {
    "angry": "#FF4B4B",
    "disgusted": "#800080",
    "fear": "#000000",
    "happy": "#FFD700",
    "neutral": "#808080",
    "sad": "#4169E1",
    "surprised": "#FFA500",
}


@st.cache_resource
def load_model(ckpt_name):
    ckpt_path = MODELS_DIR / ckpt_name
    if not ckpt_path.exists():
        st.error(f"Coudln't load model: {ckpt_name}")
        return None

    model = EmotionClassifier.load_from_checkpoint(ckpt_path)
    model.eval()
    if torch.cuda.is_available():
        model.cuda()
    return model


def get_prediction(model, img_pil):
    img_tensor = VAL_TRANSFORM(img_pil).unsqueeze(0)

    if torch.cuda.is_available():
        img_tensor = img_tensor.cuda()
        model.cuda()

    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    data = []
    for idx, prob in enumerate(probs):
        emotion = IDX_TO_EMOTION[idx]
        data.append(
            {
                "Emotion": emotion.capitalize(),
                "Probability": prob,
                "Color": EMOTION_COLORS.get(emotion, "#ccc"),
            }
        )

    return pd.DataFrame(data)


def make_chart(df):
    chart = (
        alt.Chart(df)
        .mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3)
        .encode(
            x=alt.X(
                "Probability", axis=None, scale=alt.Scale(domain=[0, 1])
            ),
            y=alt.Y(
                "Emotion", axis=alt.Axis(title=None, labels=True, tickSize=0)
            ),
            color=alt.Color("Color", scale=None),
            tooltip=["Emotion", alt.Tooltip("Probability", format=".1%")],
        )
        .properties(
            height=200
        )
        .configure_view(
            strokeWidth=0
        )
    )
    return chart


def load_image_variants(filename):
    variants = {}
    filename = filename[:4] + "/" + filename

    path_raw = DATA_PATHS["Original"] / filename
    print(path_raw)
    if path_raw.exists():
        variants["Original"] = Image.open(path_raw).convert("RGB")

    path_gray = DATA_PATHS["Grayscale"] / filename
    if path_gray.exists():
        print("exists Grayscale")
        variants["Grayscale"] = Image.open(path_gray).convert("RGB")
    else:
        if "Original" in variants:
            variants["Grayscale"] = variants["Original"].convert("L").convert("RGB")

    path_degraded = DATA_PATHS["Degraded"] / filename
    if path_degraded.exists():
        print("exists Degraded")
        variants["Degraded"] = Image.open(path_degraded).convert("RGB")

    return variants


with st.sidebar:
    st.title("Control Panel")

    all_files = sorted([f.name for f in DATA_PATHS["Original"].rglob("*.JPG")])
    all_files = [f for f in all_files if f[6:8] in ["S.", "HL", "HR"]]
    selected_file = st.selectbox(
        "Select test image:", all_files, index=0 if all_files else None
    )

    st.divider()

    ckpt_files = sorted([f.name for f in MODELS_DIR.glob("*.ckpt")])
    selected_ckpt = st.selectbox(
        "Select Model:",
        ckpt_files,
        index=next((i for i, n in enumerate(ckpt_files) if "unfrozen" in n), 0),
    )

    st.info(
        """
        **Legend:**\n
        Original: Unchanged image.\n
        Grayscale: Luminance channel only.\n
        Degraded: Added Gaussian noise + Blur.
        """
    )

st.title("Impact of Degradation on Emotion Detection")
st.markdown(
    f"Analysis for file: `{selected_file}` using model: `{selected_ckpt}`"
)

if selected_file and selected_ckpt:
    model = load_model(selected_ckpt)
    images = load_image_variants(selected_file)

    if not images:
        st.error("Failed to load images.")
        st.stop()

    col1, col2, col3 = st.columns(3)

    display_cols = [("Original", col1), ("Grayscale", col2), ("Degraded", col3)]

    for domain_name, col in display_cols:
        with col:
            st.subheader(domain_name)

            if domain_name in images:
                img = images[domain_name]

                st.image(img, use_container_width=True)

                if model:
                    df_probs = get_prediction(model, img)

                    top_emotion = df_probs.loc[df_probs["Probability"].idxmax()]

                    st.markdown(
                        f"**Prediction:** {top_emotion['Emotion']} ({top_emotion['Probability']:.1%})"
                    )
                    chart = make_chart(df_probs)
                    st.altair_chart(chart, use_container_width=True)
            else:
                st.warning("Missing file")
else:
    st.info("Select an image and model from the sidebar to get started.")
