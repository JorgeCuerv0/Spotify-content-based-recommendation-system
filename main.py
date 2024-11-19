# Module 10
# Author: Jorgé Sandoval
# Our program is a content-based recommendation system for songs
# We will be using the Million Playlist Dataset (MPD) which contains a collection of playlists provided by Spotify users
# The dataset contains information about the playlists, tracks, and artists
# We will build a content-based recommendation system that suggests similar tracks based on the track name, artist name, album name, and playlist name

import json  # Import to read JSON files which our dataset is in
import pandas as pd  # Pandas to manipulate dataframes
import datetime  # Allows us to work with date and times
from sklearn.feature_extraction.text import TfidfVectorizer  # Import for text vectorization
from sklearn.metrics.pairwise import cosine_similarity  # Import to calculate cosine similarity
import re  # Import for regular expressions

# Data Ingestion

# Load the data from JSON file
with open('/Users/jorgesandoval/Desktop/Coding/SMC/Python/SMC CS 87/Module 10/mpd.slice.0-999.json', 'r') as f:
    data = json.load(f)

# Extract the playlists
playlists = data['playlists']
print(f"loaded {len(playlists)} playlists from MPD")

# Create a DataFrame from the playlists
df = pd.DataFrame(playlists)
print(df.head())

# Empty list to store track data
track_df = []

# Loop through each playlist and extract track data
for playlist in playlists:
    for track in playlist['tracks']:
        track_df.append({
            'playlist_id': playlist['pid'],
            'playlist_name': playlist['name'],
            'track_name': track['track_name'],
            'artist_name': track['artist_name'],
            'album_name': track['album_name'],
            'duration_ms': track['duration_ms']
        })

# Create a DataFrame from the track data
track_df = pd.DataFrame(track_df)
print(track_df.head())

# Perform track-level operations
# Make all strings lowercase and remove punctuation in track_df
track_df['track_name'] = track_df['track_name'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
track_df['artist_name'] = track_df['artist_name'].str.lower().str.replace(r'[^\w\s]', '', regex=True)

# Calculate unique artist counts for each playlist
artist_count = track_df.groupby('playlist_id')['artist_name'].nunique().reset_index()
artist_count.columns = ['pid', 'unique_artists']
df = df.merge(artist_count, on='pid', how='left')

# Calculate unique album counts for each playlist
album_count = track_df.groupby('playlist_id')['album_name'].nunique().reset_index()
album_count.columns = ['pid', 'unique_albums']
df = df.merge(album_count, on='pid', how='left')

# Add diversity score
df['diversity_score'] = (df['unique_artists'] + df['unique_albums']) / df['num_tracks']
print("Diversity score added. Head of df:")
print(df[['pid', 'diversity_score']].head())

# Normalize diversity score
df['normalized_diversity_score'] = (df['diversity_score'] - df['diversity_score'].mean()) / df['diversity_score'].std()

# Add playlist age
df['playlist_age_days'] = (datetime.datetime.now() - pd.to_datetime(df['modified_at'], unit='s')).dt.days

# Normalize playlist age
df['normalized_playlist_age'] = (df['playlist_age_days'] - df['playlist_age_days'].mean()) / df['playlist_age_days'].std()

# Merge normalized features with track_df
track_df = track_df.merge(df[['pid', 'normalized_diversity_score', 'normalized_playlist_age']], 
                          left_on='playlist_id', right_on='pid', how='left')

# Combine text and numerical features into a single column
track_df['combined_features'] = (
    track_df['track_name'] + ' ' +
    track_df['artist_name'] + ' ' +
    track_df['album_name'] + ' ' +
    track_df['normalized_diversity_score'].astype(str) + ' ' +
    track_df['normalized_playlist_age'].astype(str)
)

# Apply TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(track_df['combined_features'])

# Compute cosine similarity
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

def recommend_tracks(input_track, track_df, cosine_sim, top_n=10):
    """
    Recommends similar tracks based on cosine similarity.

    Args:
    :param input_track: Name of the input track
    :param track_df: DataFrame containing track details
    :param cosine_sim: Precomputed cosine similarity matrix
    :param top_n: Number of recommendations to return
    :return: DataFrame of recommended tracks
    """
    # Clean the input track name
    input_track_cleaned = re.sub(r'[^\w\s]', '', input_track.lower())

    # Check if the track exists in the DataFrame
    if input_track_cleaned not in track_df['track_name'].values:
        print(f"Error: '{input_track}' not found in the dataset.")
        return pd.DataFrame()

    # Find the index of the input track in the DataFrame
    indx = track_df[track_df['track_name'] == input_track_cleaned].index[0]

    # Get similarity scores for the input track
    sim_scores = list(enumerate(cosine_sim[indx]))

    # Sort the tracks based on similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get indices of the top_n most similar tracks, avoiding duplicate titles
    top_indices = []
    seen_tracks = set()
    for i, score in sim_scores:
        track_name = track_df.iloc[i]['track_name']
        if track_name not in seen_tracks:
            top_indices.append(i)
            seen_tracks.add(track_name)
        if len(top_indices) == top_n:
            break

    # Return the DataFrame of unique recommended tracks
    return track_df.iloc[top_indices][['track_name', 'artist_name', 'album_name']]


# Pretty output
def print_recommendations(recommendations, input_track):
    """
    Pretty-print the recommendations in a structured format.

    Args:
    :param recommendations: DataFrame containing recommended tracks
    :param input_track: The input track name for which recommendations were generated
    """
    print(f"\nRecommendations for: **{input_track}**")
    print("-" * 50)
    for idx, row in recommendations.iterrows():
        print(f"{idx + 1}. Track: {row['track_name']}")
        print(f"   Artist: {row['artist_name']}")
        print(f"   Album: {row['album_name']}")
        print("-" * 50)


# Test the recommendation system
input_track = "Ransom" 
recommendations = recommend_tracks(input_track, track_df, cosine_sim, top_n=5)

if not recommendations.empty:
    print_recommendations(recommendations, input_track)
else:
    print("No recommendations found.")
