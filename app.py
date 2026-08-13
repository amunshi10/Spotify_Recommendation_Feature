import os
import streamlit as st

from src.recommender import load_and_prepare, SpotifyRecommender

DATA_PATH = os.path.join(os.path.dirname(__file__), 'data', 'data.csv')
ICON_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'spotify_icon.png')

st.set_page_config(page_title='Spotify Recommender', page_icon=ICON_PATH)


@st.cache_data
def get_data():
    return load_and_prepare(DATA_PATH)


header_col1, header_col2 = st.columns([1, 9], vertical_alignment='center')
with header_col1:
    st.image(ICON_PATH, width=56)
with header_col2:
    st.title('Spotify Song Recommendation System')
st.write('Content-based recommendations using audio features (danceability, energy, valence, tempo, etc.) — no listening history required.')

data = get_data()
recommender = SpotifyRecommender(data)

song_name = st.text_input('Song title', placeholder='e.g. Wonderwall')
amount = st.slider('Number of recommendations', min_value=1, max_value=20, value=10)
popularity_boost = st.slider('Popularity boost', min_value=0.0, max_value=0.1, value=0.01, step=0.005,
                              help='How much to favor popular tracks among similarly-distant candidates. '
                                   'Distances within a cluster are tiny (~0.0005-0.02), so values above '
                                   '~0.02 tend to dominate the ranking.')

if st.button('Recommend', type='primary') and song_name:
    try:
        results = recommender.recommend(song_name, amount=amount, popularity_boost=popularity_boost)
        st.dataframe(results, use_container_width=True)
    except ValueError as e:
        st.error(str(e))
