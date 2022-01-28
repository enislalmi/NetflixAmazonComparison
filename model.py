import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import streamlit as st
import plotly.graph_objs as go
import plotly as py
import plotly.express as px
import plotly.io as pio
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

def fix_modelling_data(concat_df1, concat_df2):
    all_imdb = pd.concat([concat_df1, concat_df2])
    all_imdb_up =  all_imdb[['title','director', 'type', 'duration', 'listed_in', 'vote_average','cast']]
    all_imdb_up.drop_duplicates(subset='title', keep='first')
    #all_imdb_up['cast'] = all_imdb_up['cast'].str.split(",", n = 1, expand = True)
    #all_imdb_up = all_imdb_up.rename(columns={'cast': 'leading_actor'})
    all_imdb_up['director'].replace(np.nan, 'Multiple Directors', inplace=True)
    #leading actor null rate is 2.14 so im dropping those columns
    all_imdb_up = all_imdb_up.dropna()
    all_imdb_up['leading_actor'] = all_imdb_up.cast.apply(lambda x : x.split(",")[0])
    
    movies_df = all_imdb_up[all_imdb_up['type'].str.contains('Movie')==True]

    movies_df['duration'] = movies_df['duration'].apply(lambda x : x.split(" ")[0])
    movies_df['duration'].astype(int)

    shows_df = all_imdb_up[all_imdb_up['type'].str.contains('Movie')==False]

    shows_df['duration']= shows_df['duration'].apply(lambda x : x.split(" ")[0])
    shows_df['duration'].astype(int)
    return movies_df,shows_df

def records_for_prediction(df):
    # all_imdb[['title','director', 'type', 'duration', 'listed_in', 'vote_average','cast']]
   
    #type, listedin, director, vote_average, duration, title
    title = df['title']
    type_of_listing = df['type']
    listed_in = df ['listed_in']
    director = df['director']
    leading_actor = df['leading_actor']
    duration = df['duration']
    vote_average = df['vote_average']


    fig = go.Figure(data=[go.Table(
    header=dict(
        values=['<b>Title</b>','<b>Type of Listing</b>','<b>Listed in</b>','<b>Director</b>','<b>Leading Actor</b>','<b>Duration</b>','<b>Vote average</b>'],
        line_color='rgb(18,139,181)', fill_color='rgb(222,181,34)',
        align='center',font=dict(color='rgb(12,11,0)', size=12)
    ),
    cells=dict(
        values=[title, type_of_listing, listed_in, director, leading_actor, duration, vote_average],
        line_color='rgb(18,139,181)',
        fill_color='rgb(222,181,34)',
        align='center', font=dict(color='white', size=11)
        ))
    ])
    st.plotly_chart(fig)


def merge_on_title(df1, df2):
    return pd.merge(df1, df2, how='inner', on=['title'])

netflix_df = pd.read_csv("netflix_titles.csv")
amazon_df = pd.read_csv("amazon_prime_titles.csv")
imdb_df = pd.read_csv('imdb_titles.csv')    
amazon_imdb_df = merge_on_title(amazon_df, imdb_df)
netflix_imdb_df = merge_on_title(netflix_df, imdb_df)
model_df = fix_modelling_data(amazon_imdb_df, netflix_imdb_df)

movies_df , shows_df  = fix_modelling_data(amazon_imdb_df, netflix_imdb_df)

#Movies dataset

X_movies = movies_df[['title','director','leading_actor','listed_in']]
y_movies = movies_df['vote_average']


le = preprocessing.LabelEncoder()

X_encoded = X_movies.apply(le.fit_transform)
X_encoded = pd.concat([X_encoded, movies_df['duration']], axis = 1)

#X_encoded.to_csv("movies.csv")
#movies_df.to_csv("movies_enc.csv")

X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_movies, test_size=0.2, random_state=42)

rf_reg = RandomForestRegressor(n_estimators=100,
                               criterion="mse",
                               n_jobs=5,
                               random_state=42)

rf_reg.fit(X_train, y_train)

train_pred_y = rf_reg.predict([X_train])
test_pred_y = rf_reg.predict(X_test)

print(f"train_MAE = {mean_absolute_error(y_train, train_pred_y)}")
print(f"test_MAE = {mean_absolute_error(y_test, test_pred_y)}")
