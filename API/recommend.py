from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load dataset
movies_df = pd.read_csv("tmdb_5000_movies.csv")

# Select relevant columns
movies_df = movies_df[['id', 'original_title', 'vote_average', 'vote_count']]

# Normalize review-related columns
scaler = MinMaxScaler()
movies_df[['vote_average', 'vote_count']] = scaler.fit_transform(movies_df[['vote_average', 'vote_count']])

# Function to compute recommendations based on reviews and user ratings
def recommend_by_reviews_with_ratings(movie_ratings, num_recommendations=10):
    """
    Args:
        movie_ratings: A dictionary where keys are movie titles and values are user ratings.
        num_recommendations: Number of recommendations to return.
        
    Returns:
        A list of recommended movies or an error message.
    """
    # Validate input movies
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

# Flask route for movie recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    """
    POST /recommend
    Payload: {"movie_ratings": {"Movie1": 5, "Movie2": 3}}
    Response: {"recommendations": ["MovieA", "MovieB", ...]}
    """
    try:
        # Parse input JSON
        data = request.get_json()
        movie_ratings = data.get("movie_ratings", {})

        # Validate input
        if not movie_ratings or not isinstance(movie_ratings, dict):
            return jsonify({"error": "Invalid input. Expected a dictionary of movie ratings."}), 400

        # Get recommendations
        recommendations, error = recommend_by_reviews_with_ratings(movie_ratings)
        if error:
            return jsonify({"error": error}), 404

        # Return recommendations
        return jsonify({"recommendations": recommendations}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Run the Flask app
if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
