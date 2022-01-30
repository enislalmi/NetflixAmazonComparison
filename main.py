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

st.subheader('First things first - Cleanup')
st.caption('The first thing we will do is cleanup the dataset. Such actions can be seen in the code by the usage of functions: ')
st.caption('1. Delete IDs was the first action we took. The IDs were redundant values, as they were giving no extra information. In merging functions or even searching the title will be serving as an id.')
st.caption('2. Next step was to fill the missing data. The null values were in the columns: Country, Director and Cast.')
st.caption('We determined the null values in Country as the mode. For Directors the null values were replaced with Multiple Directors, and for the cast it was fit to just say No data')
st.caption('3. The third action taken was changing the date format. We made sure that the date format was the same for both databases')
st.caption('4. Then another action was taken was to organize the ratings in the Amazon DB. By using iloc we made sure that the rating on Amazon and Netflix was on the same intervals.')
st.caption('5. Lastly, some of records as in columns: Cast and Country had multiple listings. Like: UK, US or Kristen Stewart, Robert Pattison. We saw fit to use only the first of such lists. The country as we needed to plot them in the map and needed the frequency. The cast as the first actor was used as the Leading Actor in the modelling below.')
st.caption('Remark: If you want to check the missing data, we have prepared a check_missing_data function for you. However, I made sure that the Null Rates are 0.')

netflix_df = pd.read_csv("netflix_titles.csv")
amazon_df = pd.read_csv("amazon_prime_titles.csv")
imdb_df = pd.read_csv('imdb_titles.csv')


st.subheader('Second thing? Exploring the dataset.')
st.caption('Below you will find the dataset we have been working on.')
st.subheader('Netflix')

# Some cleaup
netflix_df = delete_ids(netflix_df)
amazon_df = delete_ids(amazon_df)
netflix_df = fill_missing_data(netflix_df)
netflix_df = change_date_format(netflix_df)
amazon_df = change_date_format(amazon_df)

st.plotly_chart(show_dataframe_netflix(netflix_df))

st.caption('By exploring the dataframe, understanding our records, we came up with some interesting questions as you can see below.')

amzon_df_na = amazon_df.copy()
amazon_df_na = fill_missing_data(amzon_df_na)
st.plotly_chart(show_dataframe_amazon(amazon_df_na))



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
st.caption("This plot is done using choropleth-maps. We created a dataframe that counts the frequency of each country, and plotted them on the map.")
st.subheader("Where were Amazon movies shot?")
st.plotly_chart(geo_map(geo_amazon_df))


st.subheader("What kind of shows are we watching on Netfix?")

netflix_shows_freq = show_frequencies(netflix_df)
netflix_shows_freq = cleanup_multiple_listings(netflix_shows_freq)
shows_netflix_df = pd.DataFrame(list(netflix_shows_freq.items()), columns=[
                                'Type of Listing', 'Frequency'])

st.plotly_chart(listings_frequencies(shows_netflix_df))
st.caption('The main aim of this plot is to understand what kind of shows are we watching on both platforms. Mainly, we are focusing on the Genre. ')

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
st.caption('An interesting aspect to look into is how soon movies become available to us. Of course, we do want to subsribe to a platform that only shows us movies from the 2000s. Further, this chart makes available to us also the Duration and Type of the movie.')


st.subheader(
    "How are the two platforms divided in the type of listings they have?")
st.plotly_chart(bar_frequencies_both(shows_amazon_df, shows_netflix_df))

st.caption('Some people are more interested in Action, and some others in Drama. That is completely fine, as long as you know which platform is worth giving your money to.')

organize_ratings(amazon_df)
st.subheader(
    "Which is the target audience of Netflix? How is the content rating of its shows? ")
st.plotly_chart(content_rating(netflix_df))
st.caption('It does not come as a suprise sharing our accounts with SOs, friends and family. Knowing how the rating of the content is on the platforms is nice to understand if it can fill your family needs. None want just R movies, right?')

st.subheader(
    "Which is the target audience of Amazon? How is the content rating of its shows? ")
st.plotly_chart(content_rating(amazon_df))


st.subheader("What are the oldest and newest listings?")
st.plotly_chart(newest_oldest_listing())
st.caption('This is a very easy graph, showing which is the oldest and newest listing in the platform.')

st.subheader("How are the movies added by years in Netflix?")
st.plotly_chart(listings_added_by_year(netflix_df))
st.caption('Ever wondered what was the year Netflix added more movies? Or even how it was in the begging days of the platform?')

st.subheader("What are the titles in common between Netflix and Amazon?")
common_titles = merge_on_title(netflix_df, amazon_df)
st.plotly_chart(titles_in_common(common_titles))
st.caption('Sometimes these streaming giants coincide to have the same titles among each other. Which are those titles?')

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

st.caption('Another interesting question is the distribution of listings between Amazon and Netflix. Is Amazon focusing more on 2000-2010 movies and Netflix on the 2010-2020? Or is that just a myth? ')

st.markdown("What about Amazon?")
st.plotly_chart(listings_by_year(year_freq_amazon_df))

amazon_imdb_df = merge_on_title(amazon_df, imdb_df)
netflix_imdb_df = merge_on_title(netflix_df, imdb_df)
model_df = fix_modelling_data(amazon_imdb_df, netflix_imdb_df)

st.write("Modelling based on what?")
records_for_prediction(model_df)
st.caption('Lastly we are going to present some Modelling. The Modelling is based on the afore mentioned table.')
st.caption('The modelling is using OHE-Encoding, best for Categorical Data.')
st.caption('The Model will use Random Forest Regressor to give us a prediction based on our records.')

st.subheader("Modelling of the Training Data")
st.plotly_chart(train_plot)
st.caption('This model gives an accuracy of 89 percent. Keeping in mind the data used has been around 2500 records, we are quite pleased with the result.')



st.subheader("Modelling of the Testing Data")
st.plotly_chart(test_plot)
st.caption('The test data has more outliers than the train data, but we decided to keep it on these parameters as to exclude data overfitting.')


st.markdown(
    """<a style='display: block; text-align: center;' >Thank you.</a>
    """,
    unsafe_allow_html=True,
)
st.markdown(
    """<a style='display: block; text-align: center;' href="https://www.linkedin.com/in/enislalmi/">Enis Lalmi</a>
    """,
    unsafe_allow_html=True,
)