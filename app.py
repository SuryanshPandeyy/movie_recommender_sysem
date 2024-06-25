import streamlit as st
import pickle
import requests
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000, stop_words='english')


def fetch_poster(movie_id):
    url = "https://api.themoviedb.org/3/movie/{}?api_key=8296b3f86ba0102b47a0a703fc0082bb".format(movie_id)

    response = requests.get(url)
    data = response.json()
    return "https://image.tmdb.org/t/p/original" + data['poster_path']


st.title('Movie Recommender System')
movie_list = pickle.load(open('movies.pkl', 'rb'))
movies = movie_list['title'].values
selected_movie_name = st.selectbox('Select your movie', movies)

vectors = cv.fit_transform(movie_list['tags']).toarray()
similarity = cosine_similarity(vectors)


def recommend(movie):
    movie_index = movie_list[movie_list['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommended_movies = []
    recommended_movies_posters = []
    for j in movies_list:
        movie_id = movie_list.iloc[j[0]].movie_id
        recommended_movies_posters.append(fetch_poster(movie_id))
        recommended_movies.append(movie_list.iloc[j[0]].title)
    return recommended_movies, recommended_movies_posters


if st.button('Recommend'):
    names, posters = recommend(selected_movie_name)
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.header(names[0])
        st.image(posters[0])
    with col2:
        st.header(names[1])
        st.image(posters[1])
    with col3:
        st.header(names[2])
        st.image(posters[2])
    with col4:
        st.header(names[3])
        st.image(posters[3])
    with col5:
        st.header(names[4])
        st.image(posters[4])
