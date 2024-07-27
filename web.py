import streamlit as st
import pickle
import pandas as pd


def recommend(movie):
    movie_index = movies[movies['title'] == movie].index[0]
    distances = similarity[movie_index]
    movie_list = sorted(list(enumerate(distances)),reverse=True, key=lambda x : x[1])[1:6]

    recommended_movies=[]
    for i in movie_list:
        recommended_movies.append(movies.iloc[i[0]].title)
    return  recommended_movies

similarity = pickle.load(open('Similariti.pkl','rb'))

movies_dict=pickle.load(open('movies_dict.pkl','rb'))
movies=pd.DataFrame(movies_dict)

st.title('Movie Recommendation System')
selected_movie=st.selectbox(
          '*Select a film*',
           movies['title'].values

)

if st.button('recommend'):
    recommendations = recommend(selected_movie)
    for i in recommendations:
        st.write(i)
