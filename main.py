import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import difflib
from flask import Flask, render_template, request
import base64
from io import BytesIO

app = Flask(__name__)

def load_data():
    movies = pd.read_csv('ml-25m/movies.csv')
    ratings = pd.read_csv('ml-25m/ratings.csv')
    return movies, ratings

def process_data(movies, ratings):
    # One-hot encoding for genres
    movies['genres'] = movies['genres'].apply(lambda x: x.split('|'))
    genres_encoded = movies['genres'].str.join('|').str.get_dummies()
    movies_encoded = pd.concat([movies, genres_encoded], axis=1)
    
    # Calculate the mean rating for each movie
    mean_ratings = ratings.groupby('movieId')['rating'].mean()
    
    # Calculate the number of ratings for each movie
    num_ratings = ratings.groupby('movieId')['rating'].count()
    
    # Calculate the weighted rating for each movie
    C = 2.5
    m = 3
    weighted_rating = (num_ratings / (num_ratings + m)) * mean_ratings + (m / (num_ratings + m)) * C
    
    return movies_encoded

def recommend_top_5_movies(movies_encoded, movie1, movie2, movie3):
    # Extract the genre encodings for the three movies
    movie1_encoded = movies_encoded[movies_encoded['title'] == movie1].iloc[:, 3:].values
    movie2_encoded = movies_encoded[movies_encoded['title'] == movie2].iloc[:, 3:].values
    movie3_encoded = movies_encoded[movies_encoded['title'] == movie3].iloc[:, 3:].values
    
    # Calculate the cosine similarity between the three movies and all other movies
    similarity1 = cosine_similarity(movies_encoded.iloc[:, 3:], movie1_encoded)
    similarity2 = cosine_similarity(movies_encoded.iloc[:, 3:], movie2_encoded)
    similarity3 = cosine_similarity(movies_encoded.iloc[:, 3:], movie3_encoded)
    
    # Combine the similarity scores
    combined_similarity = similarity1 + similarity2 + similarity3
    
    # Get the indices of the top 5 movies with the highest combined similarity score
    recommended_movie_indices = combined_similarity.reshape(-1).argsort()[-5:][::-1]
    
    # Extract the titles and combined similarity scores of the recommended movies
    recommended_movie_titles = movies_encoded.iloc[recommended_movie_indices]['title'].tolist()
    recommended_movie_scores = combined_similarity[recommended_movie_indices].reshape(-1).tolist()
    
    return recommended_movie_titles, recommended_movie_scores

def find_refined_match(movies_encoded, title):
    all_titles = movies_encoded['title'].tolist()
    
    # Tokenize the input title
    input_tokens = title.split()
    
    max_score = 0
    best_match = ""
    
    for t in all_titles:
        current_score = 0
        
        # For each token in the input title, find its best match in the movie title
        for token in input_tokens:
            token_score = max(difflib.SequenceMatcher(None, token, word).ratio() for word in t.split())
            current_score += token_score
        
        # Additional score if the start of the movie title matches the input title
        if t.startswith(title):
            current_score += 1
        
        # Update the best match if the current score is higher
        if current_score > max_score:
            max_score = current_score
            best_match = t
            
    return best_match

def plot_top_movies_similarity(input_movies, movies_encoded, recommended_titles, recommended_scores, top_n=50):
    plt.figure(figsize=(18, 10))
    
    all_titles = movies_encoded['title'].tolist()
    all_scores = cosine_similarity(movies_encoded.iloc[:, 3:], movies_encoded[movies_encoded['title'].isin(input_movies)].iloc[:, 3:].mean(axis=0).values.reshape(1, -1)).reshape(-1)
    
    # Get top N movies indices based on similarity scores
    top_indices = all_scores.argsort()[-top_n:]
    
    # Extract top N movies titles and scores
    top_titles = [all_titles[i] for i in top_indices]
    top_scores = [all_scores[i] for i in top_indices]
    
    # Bar chart for top N movies
    plt.bar(top_titles, top_scores, color='lightgray', alpha=0.7)
    
    # Highlight the top 5 recommended movies
    for title, score in zip(recommended_titles, recommended_scores):
        if title in top_titles:
            plt.bar(title, score, color='orange', alpha=0.8)
    
    plt.xlabel("Movies")
    plt.ylabel("Combined Similarity Score")
    plt.title(f"Top {top_n} Movies Similar to {', '.join(input_movies)}")
    plt.xticks(rotation=90, ha='center', fontsize=10)
    plt.tight_layout()
    

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        movies, ratings = load_data()
        movies_encoded = process_data(movies, ratings)
        movie1 = request.form['movie1']
        movie2 = request.form['movie2']
        movie3 = request.form['movie3']

        # Match movie titles
        movie1_match = find_refined_match(movies_encoded, movie1)
        movie2_match = find_refined_match(movies_encoded, movie2)
        movie3_match = find_refined_match(movies_encoded, movie3)
        input_movies = [movie1_match, movie2_match, movie3_match]
        
        # Fetch top 5 recommendations
        recommended_titles, recommended_scores = recommend_top_5_movies(movies_encoded, movie1_match, movie2_match, movie3_match)
    

        # Generate plot and convert to an image format suitable for web
        img = BytesIO()
        plot_top_movies_similarity(input_movies, movies_encoded, recommended_titles, recommended_scores, top_n=50)
        plt.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode()

        return render_template('results.html', recommended_titles=recommended_titles, plot_url=plot_url)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, port=8081)







