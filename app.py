from flask import Flask, request, jsonify
from surprise import Dataset, Reader, KNNBasic
import pandas as pd

app = Flask(__name__)
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 
# Cargar datos y entrenar modelo al iniciar servidor
movies = pd.read_csv('ml-latest-small/movies.csv')
ratings = pd.read_csv('ml-latest-small/ratings.csv')
reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)
trainset = data.build_full_trainset()
algo = KNNBasic()
algo.fit(trainset)

@app.route('/recommend', methods=['GET'])
def recommend():
    user_id = int(request.args.get('user_id'))
    user_rated_movies = ratings[ratings['userId'] == user_id]['movieId'].tolist()
    all_movies = ratings['movieId'].unique()
    recommendations = []
    for movie in all_movies:
        if movie not in user_rated_movies:
            pred = algo.predict(user_id, movie)
            title = movies[movies['movieId'] == movie]['title'].values[0]
            recommendations.append({
                'movieId': int(movie),
                'title': title,
                'rating': float(pred.est)
            })
    recommendations.sort(key=lambda x: x['rating'], reverse=True)
    top5 = recommendations[:5]
    return jsonify(top5)



if __name__ == '__main__':
    app.run(debug=True)
