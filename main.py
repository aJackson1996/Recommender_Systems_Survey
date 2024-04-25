# This is a sample Python script.
import os
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from scipy import spatial
import sklearn.metrics.pairwise as metrics
import math

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions,
listed_genres = ['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical',
                 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']

path = os.path.join(os.path.dirname(__file__), "recommendations")

def get_user_preference_vector(movies_rated_by_user):
    count_vector = np.zeros_like(movies_rated_by_user['genres'].iloc[0])
    average_rating = movies_rated_by_user['rating'].mean()
    rated_movies_preferred = movies_rated_by_user[movies_rated_by_user['rating'] >= math.floor(average_rating)]
    for idx, movie in rated_movies_preferred.iterrows():
        movie_genres = movie['genres']
        for genre_index in range(len(movie_genres)):
            if movie_genres[genre_index] == 1:
                count_vector[genre_index] += 1
    count_vector = np.expand_dims(count_vector / rated_movies_preferred.shape[0], axis=0)
    return count_vector


def content_based_recommendations(ratings, movies):
    userIds = np.unique(ratings['userId'].values)
    for userId in userIds:
        user_ratings = ratings[ratings['userId'] == userId]
        user_rated_movies = user_ratings.merge(movies, how='inner', on='movieId')
        unrated_movies = movies[~movies['movieId'].isin(user_rated_movies['movieId'])]
        count_vector = get_user_preference_vector(user_rated_movies)
        unrated_movies['genres'] = unrated_movies['genres'].apply(np.array)
        unrated_movies_genre_information = np.stack(unrated_movies['genres'], axis=0)
        predicted_ratings = metrics.cosine_similarity(unrated_movies_genre_information, count_vector) * 5
        predicted_ratings_df = pd.DataFrame(columns=ratings.columns)
        predicted_ratings_df['movieId'] = unrated_movies['movieId']
        predicted_ratings_df['rating'] = [round(prediction[0] * 2) / 2 for prediction in predicted_ratings]
        predicted_ratings_df = predicted_ratings_df.assign(userId=userId)
        ratings = pd.concat([ratings, predicted_ratings_df], ignore_index=True, axis=0)
    ratings = ratings.sort_values('userId')
    ratings.to_csv(os.path.join(path, "content_based.csv"), index=False)



def calculate_error_content_based(ratings, movies):
    user_review_counts = ratings.merge(movies, how='inner', on='movieId').groupby('userId', axis=0).count()
    user_review_counts = user_review_counts[user_review_counts['rating'] > 1]
    user_ids = user_review_counts.index.to_numpy()
    squared_errors = np.zeros_like(user_ids)
    absolute_errors = np.zeros_like(user_ids)
    error_idx = 0
    start_time = time.time()
    for user_id in user_ids:
        user_ratings = ratings[ratings['userId'] == user_id]
        movies_rated_by_user = user_ratings.merge(movies, how='inner', on='movieId')
        movies_rated_by_user_genres = np.stack(movies_rated_by_user['genres'].to_numpy(), axis=0)
        # this preference vector accounts for all movies the user has rated
        user_genre_preferences = np.expand_dims(movies_rated_by_user_genres.sum(axis=0), axis=0)
        # this array of vectors contains the preference vector that would be used for estimating each respective movie
        # the user has rated, by counting the genres of every other movie the user has rated
        adjusted_preference_vectors = (user_genre_preferences - movies_rated_by_user_genres) / (
                    movies_rated_by_user.shape[0] - 1)
        rating_estimations = np.diag(metrics.cosine_similarity(movies_rated_by_user_genres, adjusted_preference_vectors)) * 5
        rating_estimations = np.round(rating_estimations * 2) / 2
        error = (rating_estimations - movies_rated_by_user['rating'].values)
        squared_errors[error_idx] = math.sqrt(np.mean(error ** 2))
        absolute_error = np.absolute(error)
        absolute_errors[error_idx] = np.mean(absolute_error)
        error_idx += 1
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Content-Based metrics. RMSE:{np.mean(squared_errors):.2f} MAE: {np.mean(absolute_errors):.2f} Fit time:{elapsed_time:.2f} seconds")


def collaborative_filtering(ratings, movies, user_user = True, user_k = 10, item_k = 100):
    user_rated_movies = ratings.merge(movies, how='inner', on='movieId')
    user_ids = np.unique(user_rated_movies['userId'].values)
    if user_user:
        user_preferences = []
        for userid in user_ids:
            user_ratings = user_rated_movies[user_rated_movies['userId'] == userid]
            user_preference = get_user_preference_vector(user_ratings)
            user_preferences.append(user_preference)
        user_preferences = np.squeeze(np.stack(user_preferences))
        similarity_matrix = metrics.cosine_similarity(user_preferences)
        similarity_df = pd.DataFrame(data=similarity_matrix, index=user_ids, columns=user_ids)
        for idx, row in similarity_df.iterrows():
            ordered_similarities = row.sort_values(ascending=False)
            ordered_similarities = ordered_similarities.to_frame().iloc[1:]
            ordered_similarities['userId'] = ordered_similarities.index
            my_watched = user_rated_movies[user_rated_movies['userId'] == idx]
            unrated_movies = movies[~movies['movieId'].isin(my_watched['movieId'])]
            ratings_for_unrated_movies = ratings.merge(unrated_movies, how='inner', on='movieId')[['userId', 'movieId', 'rating']]
            ratings_for_unrated_movies = ratings_for_unrated_movies.merge(ordered_similarities, how='inner', on='userId')
            ratings_for_unrated_movies.columns = ratings_for_unrated_movies.columns.astype(str)
            ratings_for_unrated_movies['weighted_rating'] = ratings_for_unrated_movies[str(idx)] * ratings_for_unrated_movies['rating']
            ratings_for_unrated_movies = ratings_for_unrated_movies.sort_values(str(idx), ascending=False)
            ratings_grouped_by_movie = ratings_for_unrated_movies.groupby('movieId').apply(lambda group: group.head(user_k).mean())
            predicted_ratings_df = pd.DataFrame(columns=ratings.columns)
            predicted_ratings_df['movieId'] = ratings_grouped_by_movie['movieId']
            predicted_ratings_df['rating'] = ratings_grouped_by_movie['weighted_rating']
            predicted_ratings_df.assign(userId=idx)
            ratings = pd.concat([ratings, predicted_ratings_df], axis=0, ignore_index=True)
        ratings = ratings.sort_values('userId')
        filename = f"user_user_{user_k}_neighbors.csv"
        ratings.to_csv(os.path.join(path, filename), index=False)
    else:
        predictions = []
        for userId in user_ids:
            user_ratings = user_rated_movies[user_rated_movies['userId'] == userId]
            rated_movie_genres = user_ratings['genres'].apply(np.array).values
            rated_movie_genres = np.stack(rated_movie_genres)
            unrated_movies = movies[~movies['movieId'].isin(user_ratings['movieId'])]
            unrated_movie_genres = unrated_movies['genres'].apply(np.array).values
            unrated_movie_genres = np.stack(unrated_movie_genres)
            movie_similarity_matrix = metrics.cosine_similarity(rated_movie_genres, unrated_movie_genres)
            movie_similarity_df = pd.DataFrame(data=movie_similarity_matrix, index=user_ratings['movieId'], columns=unrated_movies['movieId'])
            movie_ratings = user_ratings.sort_values('movieId')
            movie_ratings.index = user_ratings['movieId']
            weighted_ratings = movie_similarity_df.mul(movie_ratings['rating'].values, axis = 0)
            for unrated_movie_idx, col in movie_similarity_df.items():
                k_neighbors = col.sort_values(ascending=False).iloc[0:item_k]
                movie_ids = k_neighbors.index.values
                better_denom = np.sum(k_neighbors.values)
                denom = better_denom if better_denom != 0 else len(k_neighbors)
                neighbor_weighted_ratings = (weighted_ratings.loc[movie_ids, unrated_movie_idx]).values
                estimated_rating = np.sum(neighbor_weighted_ratings) / denom
                prediction = pd.Series(index =['userId', 'movieId', 'rating'], data=[userId, unrated_movie_idx, estimated_rating])
                predictions.append(prediction)
        prediction_df = pd.DataFrame(predictions)
        predicted_ratings = prediction_df['rating']
        prediction_df['rating'] = (prediction_df['rating'] * 2).round() / 2
        ratings = ratings.sort_values('userId')
        filename = f"item_item_{item_k}_neighbors.csv"
        ratings.to_csv(os.path.join(path, "item_item.csv"), index=False)


def calculate_error_collaborative_filtering(ratings, movies, user_user = True, user_k = 10, item_k = 100):
    user_ratings_for_movies = ratings.merge(movies, how='inner', on='movieId')
    user_ids = np.unique(user_ratings_for_movies['userId'].values)
    squared_errors = np.zeros(shape=(len(user_ids)), dtype=float)
    absolute_errors = np.zeros(shape=(len(user_ids)), dtype=float)
    error_idx = 0
    start_time = time.time()
    if user_user:
        user_preferences = []
        for userid in user_ids:
            user_ratings = user_ratings_for_movies[user_ratings_for_movies['userId'] == userid]
            user_preference = get_user_preference_vector(user_ratings)
            user_preferences.append(user_preference)
        user_preferences = np.squeeze(np.stack(user_preferences))
        similarity_matrix = metrics.cosine_similarity(user_preferences)
        np.fill_diagonal(similarity_matrix, 0)
        similarity_df = pd.DataFrame(data=similarity_matrix, index=user_ids, columns=user_ids)
        for idx, row in similarity_df.iterrows():
            ordered_similarities = row.sort_values(ascending=False)
            ordered_similarities = ordered_similarities.to_frame()
            ordered_similarities['userId'] = ordered_similarities.index
            my_watched = user_ratings_for_movies[user_ratings_for_movies['userId'] == idx]
            rated_movies = movies[movies['movieId'].isin(my_watched['movieId'])]
            ratings_for_rated_movies = ratings.merge(rated_movies, on='movieId', how='inner')[['userId', 'movieId', 'rating']]
            ratings_for_rated_movies = ratings_for_rated_movies.merge(ordered_similarities, how='inner', on='userId')
            ratings_for_rated_movies.columns = ratings_for_rated_movies.columns.astype(str)
            ratings_for_rated_movies['weighted_rating'] = ratings_for_rated_movies[str(idx)] * ratings_for_rated_movies['rating']
            ratings_for_rated_movies = ratings_for_rated_movies.sort_values(str(idx), ascending=False)
            ratings_grouped_by_movie = ratings_for_rated_movies.groupby('movieId').apply(lambda group: group.head(user_k).mean())
            predictions = ratings_grouped_by_movie['rating'].values
            predictions = np.round(predictions * 2) / 2
            labels = my_watched['rating'].values
            error = predictions - labels
            rmse = math.sqrt(np.mean(error ** 2))
            squared_errors[error_idx] = rmse
            mae = np.mean(np.absolute(error))
            absolute_errors[error_idx] = mae
            error_idx += 1
    else:
        for userId in user_ids:
            user_ratings = user_ratings_for_movies[user_ratings_for_movies['userId'] == userId]
            estimated_ratings = []
            rated_movie_genres = user_ratings['genres'].apply(np.array).values
            rated_movie_genres = np.stack(rated_movie_genres)
            movie_similarity_matrix = metrics.cosine_similarity(rated_movie_genres)
            np.fill_diagonal(movie_similarity_matrix, 0)
            movie_similarity_df = pd.DataFrame(data=movie_similarity_matrix, index=user_ratings['movieId'], columns=user_ratings['movieId'])
            movie_ratings = user_ratings.sort_values('movieId')
            movie_ratings.index = user_ratings['movieId']
            weighted_ratings = movie_similarity_df.mul(movie_ratings['rating'].values, axis = 0)
            for rated_movie_idx, col in movie_similarity_df.items():
                k_neighbors = col.sort_values(ascending=False).iloc[0:item_k]
                movie_ids = k_neighbors.index.values
                better_denom = np.sum(k_neighbors.values)
                denom = better_denom if better_denom != 0 else len(k_neighbors)
                neighbor_weighted_ratings = (weighted_ratings.loc[movie_ids, rated_movie_idx]).values
                estimated_rating = np.sum(neighbor_weighted_ratings) / denom
                estimated_ratings.append(estimated_rating)

            estimated_ratings = np.asarray(estimated_ratings)
            estimated_ratings = np.round(estimated_ratings * 2) / 2
            error = user_ratings['rating'].values - estimated_ratings
            rmse = math.sqrt(np.mean((error ** 2)))
            mae = np.mean(np.absolute(error))
            squared_errors[error_idx] = rmse
            absolute_errors[error_idx] = mae
            error_idx += 1
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"{'User-User' if user_user else 'Item-Item'} Collaborative Filtering metrics. K ={user_k if user_user else item_k} RMSE:{np.mean(squared_errors):.2f} MAE:{np.mean(absolute_errors):.2f} Fit time:{elapsed_time:.2f} seconds")


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    movies_df = pd.read_csv('C:/Users/antle/PycharmProjects/CAP5610_Course_Project/movies.csv')
    ratings_df = pd.read_csv('C:/Users/antle/PycharmProjects/CAP5610_Course_Project/ratings.csv').drop('timestamp',
                                                                                                       axis=1)
    movies_df = movies_df[movies_df['genres'] != '(no genres listed)']
    movies_df['genres'] = movies_df['genres'].str.split('|')
    genre_values = []
    for i in movies_df['genres'].values:
        for x in i:
            genre_values.append(x)
    binarizer = MultiLabelBinarizer(sparse_output=True)
    genres = pd.DataFrame.sparse.from_spmatrix(
        binarizer.fit_transform(movies_df.pop('genres').dropna()),
        columns=binarizer.classes_)
    movie_genres = genres[['Action', 'Adventure', 'Animation', 'Children', 'Comedy', 'Crime',
                           'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'IMAX', 'Musical',
                           'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western']].to_numpy().tolist()
    movies_df['genres'] = movie_genres
    calculate_error_content_based(ratings_df, movies_df)
    for i in range(5,20):
        calculate_error_collaborative_filtering(ratings_df, movies_df, user_user=True, user_k=i)
        calculate_error_collaborative_filtering(ratings_df, movies_df, user_user=False, item_k=i)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
