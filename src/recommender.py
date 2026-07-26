import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances

FEATURE_COLS = ['acousticness', 'danceability', 'duration_ms', 'energy',
                'instrumentalness', 'liveness', 'loudness', 'speechiness',
                'tempo', 'valence']

DEFAULT_WEIGHTS = {'danceability': 1.5, 'energy': 1.5, 'valence': 1.5}


def load_and_prepare(csv_path, n_clusters=10, random_state=42):
    data = pd.read_csv(csv_path)
    scaler = MinMaxScaler()
    data[FEATURE_COLS] = scaler.fit_transform(data[FEATURE_COLS])
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    data['cluster'] = kmeans.fit_predict(data[FEATURE_COLS])
    return data


class SpotifyRecommender:
    def __init__(self, dataset, feature_cols=FEATURE_COLS, weights=None, metric='cosine'):
        self.dataset = dataset
        self.feature_cols = feature_cols
        weights = weights or DEFAULT_WEIGHTS
        self.weight_vector = np.array([weights.get(c, 1.0) for c in feature_cols])
        self.metric = metric

    def recommend(self, song_name, amount=5, same_cluster_only=True, popularity_boost=0.01):
        matches = self.dataset[self.dataset['name'].str.lower() == song_name.lower()]
        if matches.empty:
            raise ValueError(f"'{song_name}' not found in dataset")

        song = matches.iloc[0]
        target = (song[self.feature_cols].values * self.weight_vector).reshape(1, -1)

        candidates = self.dataset[self.dataset['name'].str.lower() != song_name.lower()]
        if same_cluster_only and 'cluster' in self.dataset.columns:
            same_cluster = candidates[candidates['cluster'] == song['cluster']]
            if len(same_cluster) >= amount:
                candidates = same_cluster
        candidates = candidates.copy()

        cand_matrix = candidates[self.feature_cols].values * self.weight_vector
        distance_fn = cosine_distances if self.metric == 'cosine' else euclidean_distances
        candidates['distance'] = distance_fn(target, cand_matrix)[0]

        if popularity_boost:
            pop_norm = candidates['popularity'] / 100
            candidates['score'] = candidates['distance'] - popularity_boost * pop_norm
            sort_col = 'score'
        else:
            sort_col = 'distance'

        return (candidates
                .sort_values(sort_col)[['artists', 'name', 'distance', 'popularity']]
                .head(amount)
                .reset_index(drop=True))
