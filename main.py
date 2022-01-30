import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objs as go
import plotly as py
import plotly.express as px
import plotly.io as pio
from model import run_model
from utils import *

train_plot, test_plot = run_model()



st.title("Netflix VS Amazon - Comparison - From a Data Science Point of View")
st.header("Seeing two giants of streaming platforms through data")
st.sidebar.markdown(
    "A comparison between Netflix and Amazon Prime as Streaming Platforms")
st.sidebar.markdown("On this website you will be able to see the main differences between these two giants in the listings they offer. We will try to answer some of your questions using their databases and visualizing the most important features. Enjoy!")


netflix_df = pd.read_csv("netflix_titles.csv")
amazon_df = pd.read_csv("amazon_prime_titles.csv")
imdb_df = pd.read_csv('imdb_titles.csv')


#Some cleaup
netflix_df = delete_ids(netflix_df)
amazon_df = delete_ids(amazon_df)
netflix_df = fill_missing_data(netflix_df)
netflix_df = change_date_format(netflix_df)
amazon_df = change_date_format(amazon_df)


country_frequencies_netflix = country_frequencies(netflix_df)
country_frequencies_netflix = cleanup_multiple_listings(
    country_frequencies_netflix)
geo_netflix_df = pd.DataFrame(
    list(country_frequencies_netflix.items()), columns=['country', 'freq'])


country_frequencies_amazon = country_frequencies(amazon_df)
country_frequencies_amazon = cleanup_multiple_listings(
    country_frequencies_amazon)
geo_amazon_df = pd.DataFrame(
    list(country_frequencies_amazon.items()), columns=['country', 'freq'])

st.subheader("Where were Netflix movies shot?")
st.plotly_chart(geo_map(geo_netflix_df))

st.subheader("Where were Amazon movies shot?")
st.plotly_chart(geo_map(geo_amazon_df))


st.subheader("What kind of shows are we watching on Netfix?")

netflix_shows_freq = show_frequencies(netflix_df)
netflix_shows_freq = cleanup_multiple_listings(netflix_shows_freq)
shows_netflix_df = pd.DataFrame(list(netflix_shows_freq.items()), columns=[
                                'Type of Listing', 'Frequency'])
                              
st.plotly_chart(listings_frequencies(shows_netflix_df))


st.subheader("What kind of shows are we watching on Amazon?")

amazon_shows_freq = show_frequencies(amazon_df)
amazon_shows_freq = cleanup_multiple_listings(amazon_shows_freq)
shows_amazon_df = pd.DataFrame(list(amazon_shows_freq.items()), columns=[
                               'Type of Listing', 'Frequency'])
st.plotly_chart(listings_frequencies(shows_amazon_df))



st.subheader(
    "When were the listings released and when did they become available to us?")
st.markdown("Netflix")
st.plotly_chart(released_available(netflix_df))


st.subheader(
    "How are the two platforms divided in the type of listings they have?")
st.plotly_chart(bar_frequencies_both(shows_amazon_df, shows_netflix_df))


organize_ratings(amazon_df)
st.subheader(
    "Which is the target audience of Netflix? How is the content rating of its shows? ")
st.plotly_chart(content_rating(netflix_df))



st.subheader(
    "Which is the target audience of Amazon? How is the content rating of its shows? ")
st.plotly_chart(content_rating(amazon_df))



st.subheader("What are the oldest and newest listings?")
st.plotly_chart(newest_oldest_listing())


st.subheader("How are the movies added by years in Netflix?")
st.plotly_chart(listings_added_by_year(netflix_df))


st.subheader("What are the titles in common between Netflix and Amazon?")
common_titles = merge_on_title(netflix_df, amazon_df)
st.plotly_chart(titles_in_common(common_titles))


st.subheader("Show me some statistics!")
st.plotly_chart(title_statistics(netflix_df, amazon_df))


st.subheader("Let's see a comparison between distribution of movies")
st.markdown("Netflix")

year_freq_netflix = year_frequencies(netflix_df)
year_freq_netflix_dict = count_values(year_freq_netflix)
year_freq_netflix_df = pd.DataFrame(
    list(year_freq_netflix_dict.items()), columns=['Year Released', 'Frequency'])


year_freq_amazon = year_frequencies(amazon_df)
year_freq_amazon_dict = count_values(year_freq_amazon)
year_freq_amazon_df = pd.DataFrame(list(year_freq_amazon_dict.items()), columns=[
                                   'Year Released', 'Frequency'])

st.plotly_chart(listings_by_year(year_freq_netflix_df))


st.markdown("What about Amazon?")
st.plotly_chart(listings_by_year(year_freq_amazon_df))

amazon_imdb_df = merge_on_title(amazon_df, imdb_df)
netflix_imdb_df = merge_on_title(netflix_df, imdb_df)
model_df = fix_modelling_data(amazon_imdb_df, netflix_imdb_df)

st.write("Modelling based on what?")
records_for_prediction(model_df)

st.subheader("Modelling of the Training Data")
st.plotly_chart(train_plot)


st.subheader("Modelling of the Testing Data")
st.plotly_chart(test_plot)



