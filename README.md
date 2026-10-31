# DoodleDex: Doodle Search Engine
**🔗 Deployed Model:** <https://quick-draw-doodle-search.streamlit.app>

## How it works

1. **Data:** 10 categories (basketball, bread, circle, clock, flower, house, moon, sandwich, square, sun).
2. **Model:** MobileNetV2 (ImageNet weights, no top layer, average pooling). Doodles are resized to 96x96, and converted to 3 channels.
3. **Search:** cosine similarity.
4. **Query preprocessing**

## Results

| Method | precision@5 | precision@25 |
|---|---|---|
| Raw pixels | 0.608 | 0.550 |
| MobileNetV2 embeddings | 0.873 | 0.829 |
| MobileNetV2, inverted colors | 0.878 | 0.849 |

## Files

| File | Purpose |
|---|---|
| `doodle_search.ipynb` | Data sampling, index building, evaluation, threshold |
| `search_core.py` | Model loading, embedding, query preprocessing, search |
| `app.py` | Streamlit app (Draw, Upload, Camera tabs) |

## Credits

Doodles: Quick, Draw! dataset by Google, Inc., licensed CC BY 4.0.
