import random
import tempfile
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from deepface import DeepFace
from PIL import Image

from src.const.maps import DEEPFACE_MAP, EMOTION_MAP, IDX_TO_EMOTION
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
    "afraid": "#411E1E",
    "happy": "#FFD700",
    "neutral": "#808080",
    "sad": "#4169E1",
    "surprised": "#FFA500",
}


@st.cache_resource
def load_model(ckpt_name):
    ckpt_path = MODELS_DIR / ckpt_name
    if not ckpt_path.exists():
        st.error(f"Couldn't load model: {ckpt_name}")
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
        emotion = EMOTION_MAP[IDX_TO_EMOTION[idx]]
        data.append(
            {
                "Emotion": emotion,
                "Probability": prob,
                "Color": EMOTION_COLORS.get(emotion, "#ccc"),
            }
        )

    return pd.DataFrame(data)


def get_deepface_prediction(img_pil):
    """Run DeepFace analysis on a PIL image."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        img_pil.save(tmp.name)
        tmp_path = tmp.name

    try:
        dfs = DeepFace.analyze(
            img_path=tmp_path,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
        )
        emotions_dict = dfs[0]["emotion"]  # type: ignore
        dominant = dfs[0]["dominant_emotion"]  # type: ignore
        emotions_df = pd.DataFrame.from_dict(
            emotions_dict, orient="index", columns=["Probability"]
        )
        emotions_df.reset_index(inplace=True)
        emotions_df.rename(columns={"index": "Emotion"}, inplace=True)
        emotions_df["Emotion"] = emotions_df["Emotion"].apply(
            lambda x: EMOTION_MAP[DEEPFACE_MAP.get(x)] # type: ignore
        )
        emotions_df["Color"] = emotions_df["Emotion"].map(EMOTION_COLORS)
        emotions_df["Probability"] = (
            emotions_df["Probability"] / emotions_df["Probability"].sum()
        )

        return dominant, emotions_df
    except Exception:
        return None, None
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def make_chart(df):
    base = alt.Chart(df).encode(
        x=alt.X("Probability", axis=None, scale=alt.Scale(domain=[0, 1])),
        y=alt.Y(
            "Emotion",
            axis=alt.Axis(title=None, labels=True, tickSize=0, labelFontSize=18),
        ),
    )

    bars = base.mark_bar(cornerRadiusTopRight=3, cornerRadiusBottomRight=3).encode(
        color=alt.Color("Color", scale=None),
        tooltip=["Emotion", alt.Tooltip("Probability", format=".1%")],
    )

    text = base.mark_text(
        align="left", baseline="middle", dx=3, color="white", fontSize=16
    ).encode(text=alt.Text("Probability", format=".1%"))

    chart = (bars + text).properties(height=200).configure_view(strokeWidth=0)
    return chart


def load_image_variants(filename):
    variants = {}
    subdir = filename[:4]
    rel_path = f"{subdir}/{filename}"

    path_raw = DATA_PATHS["Original"] / rel_path
    if path_raw.exists():
        variants["Original"] = Image.open(path_raw).convert("RGB")

    path_gray = DATA_PATHS["Grayscale"] / rel_path
    if path_gray.exists():
        variants["Grayscale"] = Image.open(path_gray).convert("RGB")
    else:
        if "Original" in variants:
            variants["Grayscale"] = variants["Original"].convert("L").convert("RGB")

    path_degraded = DATA_PATHS["Degraded"] / rel_path
    if path_degraded.exists():
        variants["Degraded"] = Image.open(path_degraded).convert("RGB")

    return variants


def set_random_image(files):
    if files:
        st.session_state.selected_file = random.choice(files)


with st.sidebar:
    st.title("🎛️ Control Panel")

    st.divider()

    all_files = sorted([f.name for f in DATA_PATHS["Original"].rglob("*.JPG")])
    all_files = [f for f in all_files if f[6:8] in ["S.", "HL", "HR"]]

    selected_file = st.selectbox("Select an image:", all_files, key="selected_file")
    st.button("🔀 Random Image", on_click=set_random_image, args=(all_files,))

    st.divider()

    ckpt_files = sorted([f.name for f in MODELS_DIR.glob("*.ckpt")])
    selected_ckpt = st.selectbox(
        "Select Model:",
        ckpt_files,
        index=next((i for i, n in enumerate(ckpt_files) if "unfrozen" in n), 0),
    )

    show_deepface = st.toggle("Compare with DeepFace)", value=True)

if selected_file and selected_ckpt:
    model = load_model(selected_ckpt)
    images = load_image_variants(selected_file)

    if not images:
        st.error(f"Failed to load images for {selected_file}. Check paths.")
        st.stop()

    col1, col2, col3 = st.columns(3, gap="large")
    display_cols = [("Original", col1), ("Grayscale", col2), ("Degraded", col3)]

    for domain_name, col in display_cols:
        with col:
            st.subheader(domain_name, text_alignment="center")

            if domain_name in images:
                img = images[domain_name]

                st.container(horizontal_alignment="center").image(img, width=350)

                if model:
                    df_probs_mobilenet = get_prediction(model, img)
                    top_emotion = df_probs_mobilenet.loc[
                        df_probs_mobilenet["Probability"].idxmax()
                    ]

                    st.markdown(
                        f"#### :violet[**Our Model:** {top_emotion['Emotion'].capitalize()}]",
                        text_alignment="center",
                    )

                    chart_mobilenet = make_chart(df_probs_mobilenet)
                    st.container(horizontal_alignment="center").altair_chart(
                        chart_mobilenet, width=500
                    )

                    if show_deepface:
                        st.divider()
                        with st.spinner("Running DeepFace..."):
                            dom_df, df_probs_deepface = get_deepface_prediction(img)

                        if dom_df:
                            df_match = dom_df == top_emotion["Emotion"]
                            st.markdown(
                                f"#### :blue[**DeepFace:** {dom_df.capitalize()}]",
                                text_alignment="center",
                            )
                            chart_deepface = make_chart(df_probs_deepface)
                            st.container(horizontal_alignment="center").altair_chart(
                                chart_deepface, width=500
                            )
                        else:
                            st.caption("DeepFace failed detection.")

            else:
                st.warning("Missing file")
else:
    st.info("Select an image and model from the sidebar to get started.")
