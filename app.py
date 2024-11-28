import streamlit as st
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load and preprocess the dataset
@st.cache_data
def load_data():
    movies_df = pd.read_csv("tmdb_5000_movies.csv")
    movies_df = movies_df[['id', 'original_title', 'vote_average', 'vote_count']]
    
    # Normalize review-related columns
    scaler = MinMaxScaler()
    movies_df[['vote_average', 'vote_count']] = scaler.fit_transform(movies_df[['vote_average', 'vote_count']])
    
    return movies_df

movies_df = load_data()

# Recommendation function
def recommend_by_reviews_with_ratings(movie_ratings, num_recommendations=10):
    input_movies = movies_df[movies_df['original_title'].str.lower().isin([title.lower() for title in movie_ratings.keys()])]
    if input_movies.empty:
        return None, "None of the input movies were found in the dataset."

    # Normalize user ratings to a scale of 0-1
    max_rating = max(movie_ratings.values())
    min_rating = min(movie_ratings.values())
    normalized_ratings = {title: (rating - min_rating) / (max_rating - min_rating) for title, rating in movie_ratings.items()}

    # Compute the weighted profile
    input_movies['weight'] = input_movies['original_title'].str.lower().map(
        lambda title: normalized_ratings.get(title, 0)
    )
    weighted_profile = (input_movies[['vote_average', 'vote_count']].T * input_movies['weight']).sum(axis=1)
    weighted_profile = weighted_profile.values.reshape(1, -1)

    # Calculate cosine similarity with all movies
    similarity_scores = cosine_similarity(weighted_profile, movies_df[['vote_average', 'vote_count']])
    similarity_scores = similarity_scores.flatten()

    # Rank movies based on similarity, excluding input movies
    movies_df['similarity'] = similarity_scores
    recommendations = (
        movies_df[~movies_df['original_title'].str.lower().isin([title.lower() for title in movie_ratings.keys()])]
        .sort_values(by='similarity', ascending=False)
        .head(num_recommendations)
    )

    return recommendations['original_title'].tolist(), None

# Streamlit App UI
st.title("Movie Recommendation System 🎥")
st.write("Rate your favorite movies, and we'll recommend similar ones!")

# User Input for Movie Ratings
movie_ratings = {}
with st.form("movie_ratings_form"):
    st.write("Enter your ratings for a few movies:")
    for i in range(5):  # Allow users to rate up to 5 movies
        movie_name = st.text_input(f"Movie {i+1} Name", key=f"movie_{i}_name")
        movie_rating = st.slider(f"Rating for Movie {i+1} (1-5)", 1, 5, key=f"movie_{i}_rating")
        if movie_name.strip():  # Only include movies with a name
            movie_ratings[movie_name] = movie_rating
    
    submitted = st.form_submit_button("Get Recommendations")

# Generate Recommendations
if submitted:
    if not movie_ratings:
        st.error("Please enter at least one movie and its rating.")
    else:
        recommendations, error = recommend_by_reviews_with_ratings(movie_ratings)
        if error:
            st.error(error)
        else:
            st.success("Recommended Movies:")
            for idx, movie in enumerate(recommendations, start=1):
                st.write(f"{idx}. {movie}")
