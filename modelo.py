from surprise import Dataset, Reader, KNNBasic
from surprise.model_selection import train_test_split
from surprise import accuracy

import pandas as pd

ratings = pd.read_csv('ml-latest-small/ratings.csv')

reader = Reader(rating_scale=(0.5, 5))
data = Dataset.load_from_df(ratings[['userId', 'movieId', 'rating']], reader)

trainset, testset = train_test_split(data, test_size=0.25)

algo = KNNBasic()
algo.fit(trainset)

predictions = algo.test(testset)
print("RMSE:", accuracy.rmse(predictions))