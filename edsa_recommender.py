"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np
from PIL import Image
import seaborn as sns 
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')

# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Solution Overview"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j.iloc[0])
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        # st.write("Describe your winning approach on this page")

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
        submenu = st.sidebar.selectbox("Submenu", ["Overview", "EDA Section", "Model Perfomance"])

        if submenu == "Overview":
            st.write("""
            #### Solution Overview Content
                - Model Definitions: A brief summary of Models experimented
    - EDA Section: Exploratory Data Analysis of Data 
	- Model Perfomance Section: Comparison of models
    - User Feedback: Tell us about your experience using this app. 

			""")
            st.write("#")

            with st.expander("Model Definitions"):
                
                # Display the results side by side
                st.subheader("Non_Negative Matrix Factorization Model:")
                st.write("The Non-Negative Matrix Factorization (NMF) algorithm leverages principles from multivariate analysis \
                        and linear algebra. Its purpose is to break down a given data matrix, represented as M, into the product of two matrices with lower ranks, \
                        namely W and H. The sub-matrix W represents the NMF basis, while the sub-matrix H contains the corresponding coefficients.")
                
                st.subheader("Single Value Decomposition Model:")
                st.write("Single Value Decomposition (SVD) is a fundamental matrix factorization technique used in linear algebra and numerical analysis. \
                        The beauty of SVD lies in its ability to reduce the dimensionality of the original matrix while preserving its essential information. \
                        By truncating the number of singular values and their corresponding vectors, we can approximate the original matrix using a lower-rank approximation.\
                        It can be used to factorize a user-item interaction matrix to identify latent factors (features) that contribute to the preferences of users for different items. \
                        This technique is known as matrix factorization and has been successfully applied in recommender systems to make personalized recommendations.")

                st.subheader("Co-Clustering Model:")
                st.write("Co-Clustering, also known as biclustering or co-clustering, is a machine learning technique used to simultaneously cluster both rows and columns of a data matrix. \
                        It is particularly useful when dealing with data that exhibits natural grouping patterns in both dimensions. The goal of the Co-Clustering algorithm is to find an optimal \
                        partitioning of the rows and columns that maximizes a clustering criterion, such as minimizing the sum of squared differences between elements within each cluster.")
        
            st.write("#")

        elif submenu == "EDA Section":
            
            with st.expander("EDA Definition"):
                st.write("Exploratory Data Analysis refers to the critical process of performing initial investigations on data so as to discover patterns,to spot anomalies,to test \
                        hypothesis and to check assumptions with the help of summary statistics and graphical representations. On the following EDA we will explore the **MovieLens \
                        Dataset** to check for insight. We will be carrying out an extensive data analysis, descriptive statistics and data visualisations, all in the bid to give us \
                        an idea of what useful part of the data will be preprocessed in the Data Processing & feature engineering section in preparation for modeling.")
                
                df = pd.read_csv('resources/data/ratings.csv')
                
                # Create a bar chart for the distribution of ratings
                # def plot_ratings_distribution(dataframe):
            st.subheader("Plots")
            with st.expander("Distribution of Ratings", expanded=True):
                st.subheader("Distribution of Ratings")
                rating_counts = df['rating'].value_counts()
                fig, ax = plt.subplots()
                ax.bar(rating_counts.index, rating_counts.values)
                ax.set_xlabel('Rating')
                ax.set_ylabel('Count')
                ax.set_title('Distribution of Ratings')
                st.pyplot(fig)
                st.write("So we have our ITEM (Movies) rating ranging between `0.5` to `5.0` and most likely to be the target feature for Basic Recommendation i.e. if we are to recommend based on `userId` and `movieId` iteractions alone. \
                            \
    Also as we can see, majority of our Observations fall within the RATING range of `3.0` and `5.0` with `4.0` having the highest occurance with over `2.6 Million rating occurance`. This means that modelling by rating alone will not be entirely be representative of what the viewer may want since majority of ratings fall within the higher rates. Movie Contents/Types, User Preferences and other collaborative options, will have to be called into actions as distinguishing factor to tailoring down a recommendation to a user specification, which is what we want.")
            st.write("#")

            with st.expander("Plot of Key Genres", expanded=True):
                st.subheader("Plot of Key Genress")
                def wordcloud_generator(df, column):  
                    """
                    This function extracts all the unique keywords in a column
                    and counts the number of times each keyword occurs in the column
                    while ignoring words that are not meaningful.
                    these keywords are then used to generate a word cloud 
                    
                    Input: df
                        datatype: DataFrame
                        column
                        datatype: str
                        
                    Output: wordcloud
                            Datatype: None
                            
                    """
                    keyword_counts = {}
                    keyword_pair = []
                    words = dict()
                    
                    # list of words that should be ignored
                    ignore = ['nan', ' nan', 'nan ', 'seefullsummary', ' seefullsummary', 'seefullsummary ']
                    
                    # Extract the unique keywords 
                    for keyword in [keyword for keyword in df[column] if keyword not in ignore]:
                        if keyword in keyword_counts.keys():
                            keyword_counts[keyword] += 1
                        else:
                            keyword_counts[keyword] = 1
                    
                    # Pair the keywords with their frequencies
                    for word,word_freq in keyword_counts.items():
                        keyword_pair.append((word,word_freq))
                    # Sort the keywords accprding to their frequencies
                    keyword_pair.sort(key = lambda x: x[1],reverse=True)
                    
                    # Make it wordcloud-ready
                    for s in keyword_pair:
                        words[s[0]] = s[1]
                        
                    # Create a wordcloud using the top 2000 keywords
                    wordcloud = WordCloud(width=800, 
                                        height=500, 
                                        background_color='black', 
                                        max_words=2000,
                                        max_font_size=110,
                                        scale=3,
                                        random_state=0,
                                        colormap='Greens').generate_from_frequencies(words)

                    return wordcloud 
                movies = pd.read_csv('resources/data/movies.csv')

                movie_ratings = pd.merge(movies, df, on='movieId', how='left')
                plot_genres = wordcloud_generator(movie_ratings,'genres')
                
                # Plot wordcloud
                fig, ax = plt.subplots(figsize=(20, 8))
                ax.imshow(plot_genres, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('Plot Genre\n', fontsize=25)
                st.pyplot(fig)
            st.write("#")

            with st.expander("Distribution of Movie Genres", expanded=True):
                st.subheader("Distribution of Movie Genres")
                movies = pd.read_csv('resources/data/movies.csv')

                movie_genres = pd.DataFrame(movies['genres'].str.split("|").tolist(),
                      index=movies['movieId']).stack()
                movie_genres = movie_genres.reset_index([0, 'movieId'])
                movie_genres.columns = ['movieId', 'Genre']

                
                fig, ax = plt.subplots()
                genre_counts = movie_genres['Genre'].value_counts()
                unique_colors = sns.color_palette('pastel', len(genre_counts))
                ax.bar(genre_counts.index, genre_counts.values, color=unique_colors)
                ax.set_xlabel('Genres')
                ax.set_ylabel('Count')
                ax.set_title('Distribution of Movie Genres')
                ax.tick_params(axis='x', rotation=90)
                st.pyplot(fig)
                st.write("Mere looking at the graph, we can tell that `Drama, Comedy, Thriller and Romance` stand out as the popular movie genres \n\
                    Several factors attributes to why these genres stand out. Hence, Let us get an interesting wordcloud to showcase movie titles and \
                         the count of ratings to see if we could get any further insight on the movies")
            st.write("#")

            with st.expander("Top 10 Titles By Numnber of Ratings", expanded=True):
                st.subheader("Top 10 Titles By Numnber of Ratings")
                movies = pd.read_csv('resources/data/movies.csv')

                movie_ratings = pd.merge(movies, df, on='movieId', how='left')
                
                top_movies = movie_ratings['title'].value_counts().nlargest(10)
                unique_colors = sns.color_palette('pastel', len(top_movies))
                fig, ax = plt.subplots()
                ax.bar(top_movies.index, top_movies.values, color=unique_colors)
                ax.set_xlabel('Movie Title')
                ax.set_ylabel('Number of MovieIds')
                ax.set_title('Top 10 Movies by Number of MovieIds')
                ax.tick_params(axis='x', rotation=90)
                st.pyplot(fig)
                st.write("This reveals that all the movies in the top 10 by Number of Ratings were released in the 90's with only \
                         one Indicating certain likeness for users to this class of classical movies.")
            st.write("#")

        elif submenu == "Model Perfomance":
            with st.expander("Model Performance", expanded=True):
                st.write("RMSE is a popular evaluation metric in various machine learning tasks, particularly in regression problems, where the goal is to predict a continuous numerical output. \
                            It provides a measure of the model's average prediction error, with a lower RMSE indicating better predictive accuracy.")
                rmse_scores = [0.78, 0.81, 0.89]
                trained_models = ['Single Value Decomposition', 'Single Value Decomposition', 'Co-Clustering']

                model_performance = pd.DataFrame({'Model': trained_models, 'RMSE': rmse_scores})

                # Display the model performance table
                st.subheader("Model Performance:")
                st.table(model_performance)

                # Highlight the best model based on RMSE score
                best_model = model_performance.loc[model_performance['RMSE'].idxmin(), 'Model']
                st.subheader(f"The best model is {best_model} with RMSE: {model_performance['RMSE'].min()}")
            st.write("#")      


if __name__ == '__main__':
    main()
