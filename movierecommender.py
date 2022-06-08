import numpy as np
import pandas as pd
import difflib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# loading the data from the csv file to a panda dataframe
movies_data = pd.read_csv('datasheet/movies.csv')
# printing the first five rows of the dataframe
# print(movies_data.head())
# number of rows and columns in the data frame
# print(movies_data.shape)

# selecting the relevant features for recommendation
selected_features = ['genres', 'keywords', 'tagline', 'cast', 'director']
# print(selected_features)

# replacing the null values with string
for feature in selected_features:
    movies_data[feature] = movies_data[feature].fillna('')

# combining all the selected features
combined_features = movies_data['genres']+' '+movies_data['keywords']+' '+movies_data['tagline']+' '+movies_data['cast']+' '+movies_data['director']
# print(combined_features)

# converting the text data to feature vectors
vectorizer = TfidfVectorizer()
feature_vector = vectorizer.fit_transform(combined_features)
# print(feature_vector)

# getting the similarity scores using cosine similarity
similarity = cosine_similarity(feature_vector)
# print(similarity)
# print(similarity.shape)

# getting the movie name from the user
movie_name = input('Enter your favourite movie name: ')

# creating a list with all the movies given in the dataset
title_list = movies_data['title'].tolist()
# print(title_list)

# finding the close match for the movie name given by the user
find_close_match = difflib.get_close_matches(movie_name, title_list)
# print(find_close_match)
close_match = find_close_match[0]
# print(close_match)

# find the index of the movie with title
movie_index = movies_data[movies_data.title == close_match]['index'].values[0]
# print(movie_index)

# getting list of similar movies
similarity_score = list(enumerate(similarity[movie_index]))
# print(similarity_score)
# print(len(similarity_score))

# sorting the movies based on their similarity score
sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)
# print(sorted_similar_movies)

# print the name of similar movies based on the index
print("Movies recommended are: \n")
i = 1
for movie in sorted_similar_movies:
    index = movie[0]
    title_from_index = movies_data[movies_data.index == index]['title'].values[0]
    if i <= 20:
        print(i, '. ', title_from_index)
        i += 1
