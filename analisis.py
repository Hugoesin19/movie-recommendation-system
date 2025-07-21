import pandas as pd

ratings = pd.read_csv('ml-latest-small/ratings.csv')
print(ratings.head())
print(ratings.describe())
print(f"Usuarios únicos: {ratings['userId'].nunique()}")
print(f"Películas únicas: {ratings['movieId'].nunique()}")
ratings = ratings.dropna().drop_duplicates()
print(ratings['rating'].value_counts())
