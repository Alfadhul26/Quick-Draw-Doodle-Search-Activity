import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter, ImageOps
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


def load_model():
    return MobileNetV2(input_shape=(96, 96, 3), weights="imagenet",
                       include_top=False, pooling="avg")


def to_model_input(images_28, invert=False):
    """(N, 28, 28) doodles -> (N, 96, 96, 3) model-ready float array."""
    x = np.asarray(images_28, dtype="float32")
    if x.ndim == 2:
        x = x[None]
    if invert:
        x = 255.0 - x
    x = tf.image.resize(x[..., None], (96, 96)).numpy()
    x = np.repeat(x, 3, axis=-1)
    return preprocess_input(x)


def embed(model, images_28, invert=False, batch=64):
    out = [model.predict(to_model_input(images_28[i:i + batch], invert), verbose=0)
           for i in range(0, len(images_28), batch)]
    v = np.concatenate(out)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def preprocess_query(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype("uint8"))
    image = ImageOps.exif_transpose(image)                       # upright phone photos

    # 1. flatten transparency onto white, then grayscale
    image = image.convert("RGBA")
    white = Image.new("RGBA", image.size, "white")
    gray = np.array(Image.alpha_composite(white, image).convert("L"))

    # 2. strokes = pixels much darker than the paper (the median)
    is_stroke = gray < np.median(gray) * 0.6
    if is_stroke.sum() < 0.0005 * is_stroke.size:                # nothing drawn
        return None
    strokes = Image.fromarray(np.where(is_stroke, 255, 0).astype("uint8"))

    # 3. crop to the strokes, center in a square with a margin
    strokes = strokes.crop(strokes.getbbox())
    w, h = strokes.size
    side = int(max(w, h) * 1.25)
    square = Image.new("L", (side, side), 0)
    square.paste(strokes, ((side - w) // 2, (side - h) // 2))

    # 4. shrink, thicken, shrink to 28x28
    big = square.resize((112, 112), Image.BOX).filter(ImageFilter.MaxFilter(5))
    small = np.array(big.resize((28, 28), Image.BOX))
    return (small / small.max() * 255).astype("uint8")


def search(query_vector, embeddings, k):
    scores = embeddings @ query_vector
    top = np.argsort(-scores)[:k]
    return top, scores[top]
