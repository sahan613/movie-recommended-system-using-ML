import pickle
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.stem.porter import PorterStemmer
import sklearn
from sklearn.feature_extraction.text import CountVectorizer
import ast
import pandas as pd
import numpy as np
import os

# Read the data
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')

# Merge the data sets
movies = movies.merge(credits, on='title')

# Select relevant columns
movies = movies[['title', 'movie_id', 'genres',
                 'keywords', 'overview', 'cast', 'crew']]

# Drop missing values
movies.dropna(inplace=True)

# Drop duplicates
# Print the number of duplicates before dropping, if necessary for debugging
# print(movies.duplicated().sum())
movies.drop_duplicates(inplace=True)

# Function to extract names from JSON-like strings


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# Apply the conversion functions
movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Function to extract top 3 cast members


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter != 3:
            L.append(i['name'])
            counter += 1
        else:
            break
    return L


movies['cast'] = movies['cast'].apply(convert3)

# Function to extract the director's name


def fetch(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


movies['crew'] = movies['crew'].apply(fetch)

# Split the overview into words
movies['overview'] = movies['overview'].apply(lambda x: x.split())

# Remove spaces from strings in lists
movies['genres'] = movies['genres'].apply(
    lambda x: [i.replace(" ", "") for i in x])
movies['keywords'] = movies['keywords'].apply(
    lambda x: [i.replace(" ", "") for i in x])
movies['cast'] = movies['cast'].apply(
    lambda x: [i.replace(" ", "") for i in x])
movies['crew'] = movies['crew'].apply(
    lambda x: [i.replace(" ", "") for i in x])

# Create 'tags' column by combining genres, overview, cast, crew, and keywords
movies['tags'] = movies['genres'] + movies['overview'] + \
    movies['cast'] + movies['crew'] + movies['keywords']

# Create a new DataFrame with selected columns
new_df = movies[['movie_id', 'title', 'tags']].copy()

# Join the tags into a single string
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))

# Convert tags to lowercase
new_df['tags'] = new_df['tags'].apply(lambda x: x.lower())

# Check the head of the new DataFrame
# print(new_df.head())


# print(vectors)

# porter stemmer used for devided same meaning words in one word(loving , loved,lover-love)
ps = PorterStemmer()


def stem(text):
    y = []
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)


# print(ps.stem('coding'))

# calling the create def
# print(stem('i would like to coding'))

# apply out tags column to get the words in same words
new_df['tags'] = new_df['tags'].apply(stem)

cv = CountVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(new_df['tags']).toarray()

similarities = cosine_similarity(vectors)

# print(list(enumerate(similarities[0])))

# print(new_df['title'])
# print(new_df[new_df['title'] == 'Shanghai Calling'].index[0])

'''print(sorted(list(enumerate(similarities[0])),
reverse = True, key = lambda x: x[1])[1:6])'''


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarities[movie_index]
    movies_list = sorted(list(enumerate(distances)),
                         reverse=True, key=lambda x: x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# recommend('Batman Begins')

pickle.dump(new_df.to_dict(), open('movies_dict.pkl', 'wb'))

pickle.dump(similarities, open('Similariti.pkl', 'wb'))
