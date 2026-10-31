import json
import numpy as np
import streamlit as st
from PIL import Image

from search_core import load_model, preprocess_query, embed, search

try:
    from streamlit_drawable_canvas import st_canvas
except Exception:
    st_canvas = None

st.set_page_config(page_title="DoodleDex", layout="wide")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_index():
    embeddings = np.load("index/embeddings.npy")
    images = np.load("index/gallery_images.npy")
    config = json.load(open("index/config.json"))
    return embeddings, images, config


def big(arr, size):
    return Image.fromarray(arr).resize((size, size), Image.NEAREST)


def show_results(image):
    small = preprocess_query(image)
    if small is None:
        st.info("Nothing drawn yet.")
        return
    embeddings, gallery, config = get_index()

    left, right = st.columns(2)
    left.image(image, caption="your input", width=200)
    right.image(big(small, 200), caption="what the model sees (28x28)")

    vec = embed(get_model(), small[None])[0]
    idx, scores = search(vec, embeddings, config["top_k"])

    if scores[0] < config["threshold"]:
        st.warning("Nothing similar found")
        return
    for start in range(0, len(idx), 5):
        cols = st.columns(5)
        for col, j, s in zip(cols, idx[start:start + 5], scores[start:start + 5]):
            col.image(big(gallery[j], 112), caption=f"{s:.2f}")


st.title("DoodleDex")
tab_draw, tab_upload, tab_camera = st.tabs(["Draw", "Upload", "Camera"])

with tab_draw:
    if st_canvas is None:
        st.warning("streamlit-drawable-canvas is not available. Use Upload or Camera.")
    else:
        try:
            canvas = st_canvas(stroke_width=8, stroke_color="#000000", background_color="#FFFFFF",
                               height=280, width=280, drawing_mode="freedraw",
                               update_streamlit=True, return_image_data=True, key="canvas")
            if canvas.image_data is not None:
                show_results(Image.fromarray(canvas.image_data.astype("uint8")))
        except Exception as e:
            st.warning(f"Canvas failed with this Streamlit version ({e}). Use Upload or Camera.")

with tab_upload:
    file = st.file_uploader("Photo or image of a drawing", type=["png", "jpg", "jpeg"])
    if file:
        show_results(Image.open(file))

with tab_camera:
    photo = st.camera_input("Hold a paper drawing up to the camera")
    if photo:
        show_results(Image.open(photo))