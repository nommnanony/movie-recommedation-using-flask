from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load movies dataframe and similarity matrix
movies = pickle.load(open("movies.pkl", "rb"))
similarity = pickle.load(open("similarity.pkl", "rb"))

# Get movie list
movie_list = movies["title"].values

def recommend(movie):
    # Find index of the movie
    if movie not in movies["title"].values:
        return []
    idx = movies[movies["title"] == movie].index[0]
    distances = list(enumerate(similarity[idx]))
    # Sort by similarity score
    movies_list = sorted(distances, key=lambda x: x[1], reverse=True)[1:6]
    return [movies.iloc[i[0]].title for i in movies_list]

@app.route("/")
def index():
    return render_template("index.html", movies=movie_list)

@app.route("/recommend", methods=["POST"])
def recommend_movies():
    movie = request.form.get("movie")
    recommendations = recommend(movie)
    return render_template("recommend.html", movie=movie, recommendations=recommendations)

if __name__ == "__main__":
    app.run(debug=True)
