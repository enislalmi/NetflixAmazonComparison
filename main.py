import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objs as go
import plotly as py
import plotly.express as px
import plotly.io as pio

st.header("Netflix VS Amazon - Comparison - From a Data Science Point of View")
st.write("Seeing two giants of streaming platforms through data")

netflix_df = pd.read_csv("netflix_titles.csv")
amazon_df = pd.read_csv("amazon_prime_titles.csv")


    #''' Function to delete the ids, as the title is a unique value and having the ids is just
#unneccesary'''


def delete_ids(df):
    df = df.drop(["show_id"], axis=1)
    return df


def check_missing_data(df):
    for i in df.columns:
        null_rate = df[i].isna().sum() / len(df) * 100
        if null_rate > 0:
            print(f"{i} null rate: {round(null_rate,2)}%")
        else:
            print(f"Column {i} has no missing data.")


def fill_missing_data(df):
    #'''For the country missing data the most common country could be used to fill null values'''
    df['country'] = df['country'].fillna(df['country'].mode()[0])

    #'''Something that was seen was that most of the TV Shows did not have a director, that is
   # because most of them have multiple directors'''
    df['cast'].replace(np.nan, 'Null values', inplace=True)
    df['director'].replace(np.nan, 'Multiple Directors', inplace=True)

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df


def change_date_format(df):
    df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
    #df["date_added"] = df['date_added'].dt.strftime('%m-%Y')
    df["date_added"] = df['date_added'].dt.strftime('%Y')

    return df


#For the movies/tv shows that have more than country, I will be countring only the first country
#which coincides with the country where the movie was filmed the most

def cleanup_multiple_listings(df):

    dict = df.to_dict()

    return {key.split(',')[0]: (value+dict.get(key.split(',')[0], 0)) if ',' in key else value
            for key, value in dict.items()}


def country_frequencies(df):
    return df['country'].value_counts()

def show_frequencies(df):
    return df['listed_in'].value_counts()


def information(df):
    return df.info()

def description(df):
    return df.describe()

def correlation(df):
    return df.corr()

country_frequencies_netflix = country_frequencies(netflix_df)
country_frequencies_netflix = cleanup_multiple_listings(country_frequencies_netflix)
geo_netflix_df =  pd.DataFrame(list(country_frequencies_netflix.items()),columns = ['country','freq'])
country_frequencies_amazon = country_frequencies(amazon_df)
country_frequencies_amazon = cleanup_multiple_listings(country_frequencies_amazon)
geo_amazon_df =  pd.DataFrame(list(country_frequencies_amazon.items()),columns = ['country','freq'])


def geo_map(df):
    data = dict (
    type = 'choropleth',
    locations = df['country'],
    locationmode='country names',
    z=df['freq'])
    map = go.Figure(data=[data])
    st.plotly_chart(map)
  

def organize_ratings(df):
        df.loc[df['rating'] == "13+", 'rating'] = "PG-13"
        df.loc[df['rating'] == "16+", 'rating'] = 'NC-17'
        df.loc[df['rating'] == "18+", 'rating'] = 'R'
        df.loc[df['rating'] == "7+", 'rating'] = 'TV-Y7'
        df.loc[df['rating'] == "UNRATED", 'rating'] = 'NR'
        df.loc[df['rating'] == "NOT_RATE", 'rating'] = 'NR'
        df.loc[df['rating'] == "AGES_18_", 'rating'] = 'R'
        df.loc[df['rating'] == "AGES_16_", 'rating'] = 'NC-17'
        df.loc[df['rating'] == "ALL", 'rating'] = 'G'
        df.loc[df['rating'] == "ALL_AGES", 'rating'] = 'G'
        df.loc[df['rating'] == "16", 'rating'] = 'NC-17'


def listings_added_by_year(df):
    df = change_date_format(df)
    year_added_frequencies = df['date_added'].value_counts().to_dict()
    year_added_frequencies = pd.DataFrame(year_added_frequencies.items(), columns=['Year Added', 'Frequency'])
    fig = px.line(year_added_frequencies, x='Year Added', y='Frequency',
              title='Movies added by year',width=800, height=400)

    fig.update_xaxes(
        dtick="M1",
        tickformat="%b\n%Y")
    st.plotly_chart(fig)




st.write("Where were Netflix movies shot?")
geo_map(geo_netflix_df)  
st.write("Where were Amazon movies shot?")
geo_map(geo_amazon_df) 

def listings_frequencies(df):
    fig = px.scatter(df, y=df['Type of Listing'], x=df["Frequency"])
    fig.update_traces(marker_size=10)
    st.plotly_chart(fig)


st.write("What kind of shows are we watching on Netfix?")
netflix_shows_freq = show_frequencies(netflix_df)
netflix_shows_freq = cleanup_multiple_listings(netflix_shows_freq)
shows_netflix_df =  pd.DataFrame(list(netflix_shows_freq.items()),columns = ['Type of Listing','Frequency'])
listings_frequencies(shows_netflix_df)


st.write("What kind of shows are we watching on Amazon?")
amazon_shows_freq = show_frequencies(amazon_df)
amazon_shows_freq = cleanup_multiple_listings(amazon_shows_freq)
shows_amazon_df =  pd.DataFrame(list(amazon_shows_freq.items()),columns = ['Type of Listing','Frequency'])
listings_frequencies(shows_amazon_df)


st.write("How are the two platforms divided in the type of listings they have?")
def bar_frequencies_both(shows_amazon_df, shows_netflix_df):
    amazon_platform = 'Amazon'
    netflix_platform = 'Netflix'
    shows_amazon_df['Streaming'] = amazon_platform
    shows_netflix_df['Streaming'] = netflix_platform
    frames = [shows_amazon_df, shows_netflix_df]

    result = pd.concat(frames)
    fig = px.bar(result, x="Streaming", y="Frequency", color="Type of Listing", title="Comparison between listings in the two streaming platforms")
    
    st.plotly_chart(fig)

bar_frequencies_both(shows_amazon_df, shows_netflix_df)
organize_ratings(amazon_df)
def content_rating(df):
    
    fig = px.pie(df, values=df['rating'].value_counts(), names=df['rating'].dropna().unique(),
                title='Ratings of the listings:')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig)

st.write("Which is the target audience of Netflix? How is the content rating of its shows? ")
content_rating(netflix_df)

st.write("Which is the target audience of Amazon? How is the content rating of its shows? ")
content_rating(amazon_df)

def newest_oldest_listing():
    platform = ['Netflix', 'Amazon', 'Netflix', 'Amazon']
    year = [netflix_df['release_year'].max(), amazon_df['release_year'].max(),netflix_df['release_year'].min(),amazon_df['release_year'].min()]

    data = [dict(
    type = 'scatter',
    x = platform,
    y = year,
    mode = 'markers',
    
    )]

    layout = dict(
        title = 'Newest and Oldest movies across both platforms'
    )

    fig_dict = dict(data=data, layout=layout)

    st.plotly_chart(fig_dict)

st.write("What are the oldest and newest listings?")

newest_oldest_listing()


st.write("How are the movies added by years in Netflix?")
listings_added_by_year(netflix_df)


#if __name__ == '__main__':

    #netflix_df = delete_ids(netflix_df)
    #amazon_df = delete_ids(amazon_df)
    #netflix_df = fill_missing_data(netflix_df)
    #amazon_df = fill_missing_data(amazon_df)
    #netflix_df = change_date_format(netflix_df)
    #amazon_df = change_date_format(amazon_df)
    #netflix_country_freq = country_frequencies(netflix_df)
    #cleanup_countries(netflix_country_freq)
    #amazon_country_freq = country_frequencies(amazon_df)
    #cleanup_countries(amazon_country_freq)
    #organize_ratings(amazon_df)
    
