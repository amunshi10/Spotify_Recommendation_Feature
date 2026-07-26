# Spotify Song Recommendation System

## What this project does

We're building a **content-based song recommender**: given the name of a song, it suggests other
songs that *sound* similar, based purely on Spotify's audio features (danceability, energy, valence,
tempo, loudness, acousticness, etc.) — no user listening history or collaborative filtering involved.
The dataset covers 170,000+ tracks released between 1921 and 2020.

## Dataset

[Spotify Dataset 1921-2020, 160k+ Tracks](https://www.kaggle.com/datasets/ektanegi/spotifydata-19212020) (Kaggle).

Download `data.csv` from the link above (or the zip export) and place it in the `data/` folder.

## Project steps

1. **EDA** — check shape, dtypes, missing values, and correlations between audio features.
2. **Clean & select features** — drop identifier/non-audio columns (`id`, `name`, `artists`, `release_date`, `year`).
3. **Scale features** — `MinMaxScaler` on the numeric audio-feature columns so no feature dominates by scale.
4. **(Optional) Cluster** — K-Means over the scaled features to add a "sound cluster" label as an extra feature.
5. **Build the recommender** — given a song, compute a distance (Euclidean) to every other song across
   the numeric features and return the closest N.
6. **Evaluate** — spot-check recommendations; no ground truth exists, so this is qualitative.

## Structure

```
data/         raw dataset (data.csv goes here, gitignored)
notebooks/    exploration and model notebook
outputs/      saved plots/results
```

## Setup

```
pip install -r requirements.txt
```

Then open `notebooks/spotify_recommender.ipynb` and run all cells.

## What we did

Following the steps above, we worked through the full pipeline in
[`notebooks/spotify_recommender.ipynb`](notebooks/spotify_recommender.ipynb):

- **Loaded** the 170,653-row dataset and confirmed there were **no missing values** across any column.
- **Checked correlations** between the audio features (see `outputs/correlation_heatmap.png`) after
  dropping the identifier/metadata columns that don't describe a track's sound.
- **Scaled** the 10 continuous audio features (`acousticness`, `danceability`, `duration_ms`, `energy`,
  `instrumentalness`, `liveness`, `loudness`, `speechiness`, `tempo`, `valence`) with `MinMaxScaler` so
  every feature contributes on the same 0–1 scale — `loudness` (measured in dB, roughly -60 to 0) would
  otherwise dominate a raw distance calculation.
- **Clustered** the scaled features into 10 groups with K-Means, adding a `cluster` label as a rough
  proxy for "genre" without ever looking at genre metadata.
- **Built a `SpotifyRecommender` class** that, given a song title, computes the Euclidean distance from
  that song to every other track in one vectorized call (`sklearn.metrics.pairwise.euclidean_distances`)
  and returns the closest matches. This avoids the slow row-by-row Python loop that's common in simpler
  tutorial versions of this project, which matters at 170k+ rows.
- **Tested it** against "Lovers Rock" and got back a top-10 list dominated by reggae/roots artists —
  Jimmy Cliff, Peter Tosh, Burning Spear, UB40 — none of which the model was ever told about the genre.
  That coherence is a good sanity check that the audio-feature distance is actually capturing something
  meaningful about a song's sound, not just noise.
