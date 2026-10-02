# Content-Based Movie Recommender

Recommend movies using textual similarity between genres, keywords, taglines, cast, and director metadata. The notebook combines these features, transforms them with TF-IDF, and ranks films by cosine similarity.

**Stack:** Python · pandas · NumPy · scikit-learn · Jupyter

## How it works

1. Load the included `movies.csv` and replace missing text with empty strings.
2. Combine five metadata fields into a text representation.
3. Build TF-IDF vectors and a pairwise cosine similarity matrix.
4. Match a typed movie title with `difflib.get_close_matches`.
5. Display the highest-ranked similar titles.

## Run locally

```bash
git clone https://github.com/abiy8/Movie-recommendation-system-.git
cd Movie-recommendation-system-
python -m venv .venv
# Activate .venv for your operating system.
pip install jupyter pandas numpy scikit-learn
jupyter notebook
```

Open `MovieRecommendation.ipynb` and run cells in order. The notebook currently reads `/content/movies.csv`; for local use, adjust that path in your local copy to `movies.csv`, or upload the included CSV to `/content` when using Colab. Enter a title present in the dataset when prompted.

## Limitations

This is a content-based learning project, with no user ratings or personalized feedback. The current notebook assumes a close title match exists and may include the input movie among results. The full similarity matrix uses quadratic memory; larger catalogs need a more selective retrieval approach. No recommendation-quality benchmark is claimed.
