"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import pandas as pd
import numpy as np
import scipy as sp
import pickle
import copy
import operator # <-- Convienient item retrieval during iteration
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer

# Importing data
movies_df = pd.read_csv('resources/data/movies.csv',sep = ',')
ratings_df = pd.read_csv('resources/data/ratings.csv')
ratings_df.drop(['timestamp'], axis=1,inplace=True)

# We make use of an SVD model trained on a subset of the MovieLens 10k dataset.
model=pickle.load(open('resources/models/SVD.pkl', 'rb'))

def prediction_item(item_id):
    """Map a given favourite movie to users within the
       MovieLens dataset with the same preference.

    Parameters
    ----------
    item_id : int
        A MovieLens Movie ID.

    Returns
    -------
    list
        User IDs of users with similar high ratings for the given movie.

    """
    # Data preprosessing
    reader = Reader(rating_scale=(0, 5))
    load_df = Dataset.load_from_df(ratings_df,reader)
    a_train = load_df.build_full_trainset()

    predictions = []
    for ui in a_train.all_users():
        predictions.append(model.predict(iid=item_id,uid=ui, verbose = False))
    return predictions

def pred_movies(movie_list):
    """Maps the given favourite movies selected within the app to corresponding
    users within the MovieLens dataset.

    Parameters
    ----------
    movie_list : list
        Three favourite movies selected by the app user.

    Returns
    -------
    list
        User-ID's of users with similar high ratings for each movie.

    """
    # Store the id of users
    id_store=[]
    # For each movie selected by a user of the app,
    # predict a corresponding user within the dataset with the highest rating
    for i in movie_list:
        predictions = prediction_item(item_id = i)
        predictions.sort(key=lambda x: x.est, reverse=True)
        # Take the top 10 user id's from each movie with highest rankings
        for pred in predictions[:100]:
            id_store.append(pred.uid)
    # Return a list of user id's
    return id_store

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def collab_model(movie_list,top_n=10):
    """Performs Collaborative filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """

    indices = pd.Series(movies_df['title'])
    movie_ids = pred_movies(movie_list)
    df_init_users = ratings_df[ratings_df['userId']==movie_ids[0]]
    for i in movie_ids[1:] :
        # df_init_users=df_init_users.append(ratings_df[ratings_df['userId']==i])
        df_init_users = pd.concat([df_init_users, ratings_df[ratings_df['userId']==i]])
    
    cosine_matrix = df_init_users.pivot_table(index=['userId'], columns=['movieId'], values='rating').fillna(0)
    cosine_matrix_norm = cosine_matrix.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)
    cosine_matrix_norm.fillna(0, inplace=True)
    cosine_matrix_norm = cosine_matrix_norm.T 
    cosine_matrix_norm = cosine_matrix_norm.loc[:, (cosine_matrix_norm != 0).any(axis=0)]
    cosine_matrix_sparse = sp.sparse.csr_matrix(cosine_matrix_norm.values)
    
    # Getting the cosine similarity matrix
    # cosine_sim = cosine_similarity(np.array(df_init_users), np.array(df_init_users))
    cosine_sim = cosine_similarity(cosine_matrix_sparse.T)
    user_cosine_sim_df = pd.DataFrame(cosine_sim, index=cosine_matrix_norm.columns, columns=cosine_matrix_norm.columns)
    
    sim_users = user_cosine_sim_df.index
    favorite_user_items = [] # <-- List of highest rated items gathered from the k users  
    most_common_favorites = {} # <-- Dictionary of highest rated items in common for the k users
    # print(cosine_matrix_norm.head())
    for i in sim_users:
        # Maximum rating given by the current user to an item 
        max_score = cosine_matrix_norm.loc[:, i].max()
        # Save the names of items maximally rated by the current user   
        favorite_user_items.append(cosine_matrix_norm[cosine_matrix_norm.loc[:, i]==max_score].index.tolist())
    
    # Loop over each user's favorite items and tally which ones are 
    # most popular overall.
    for item_collection in range(len(favorite_user_items)):
        for item in favorite_user_items[item_collection]: 
            if item in most_common_favorites:
                most_common_favorites[item] += 1
            else:
                most_common_favorites[item] = 1
    
    # Sort the overall most popular items and return the top-N instances
    sorted_list = sorted(most_common_favorites.items(), key=operator.itemgetter(1), reverse=True)[:top_n]
    
    # top_N = [x[0] for x in sorted_list]
    # top_N = [indices.iloc[index-1] for index, _ in sorted_list]
    top_N = [movies_df[movies_df['movieId'] == idx]['title'] for idx, _ in sorted_list]
    return top_N

    # idx_1 = indices[indices == movie_list[0]].index[0]
    # idx_2 = indices[indices == movie_list[1]].index[0]
    # idx_3 = indices[indices == movie_list[2]].index[0]

    # Creating a Series with the similarity scores in descending order
    # rank_1 = user_cosine_sim_df[idx_1]
    # rank_2 = user_cosine_sim_df[idx_2]
    # rank_3 = user_cosine_sim_df[idx_3]

    # Calculating the scores
    # score_series_1 = pd.Series(rank_1).sort_values(ascending = False)
    # score_series_2 = pd.Series(rank_2).sort_values(ascending = False)
    # score_series_3 = pd.Series(rank_3).sort_values(ascending = False)

     # Appending the names of movies
    # listings = score_series_1.append(score_series_1).append(score_series_3).sort_values(ascending = False)
    # listings = pd.concat([score_series_1,score_series_2,score_series_3]).sort_values(ascending=True)
    # recommended_movies = []
    # Choose top 50
    # top_50_indexes = list(listings.iloc[1:50].index)
    # print(top_50_indexes)
    # Removing chosen movies
    # top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
    # for i in top_indexes[:top_n]:
    #     recommended_movies.append(list(movies_df['title'])[i])
    # # print(recommended_movies)
    # return recommended_movies
