from bleach import clean
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objs as go
import plotly as py


st.header("Netflix VS Amazon - Comparison")
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
    df["date_added"] = df['date_added'].dt.strftime('%m-%Y')

    return df


#For the movies/tv shows that have more than country, I will be countring only the first country
#which coincides with the country where the movie was filmed the most

def cleanup_countries(df):

    countries_dict = df.to_dict()

    return {key.split(',')[0]: (value+countries_dict.get(key.split(',')[0], 0)) if ',' in key else value
            for key, value in countries_dict.items()}


def country_frequencies(df):
    return df['country'].value_counts()


def information(df):
    return df.info()

def description(df):
    return df.describe()

def correlation(df):
    return df.corr()

country_frequencies_netflix = country_frequencies(netflix_df)
country_frequencies_netflix = cleanup_countries(country_frequencies_netflix)
geo_netflix_df =  pd.DataFrame(list(country_frequencies_netflix.items()),columns = ['country','freq'])
country_frequencies_amazon = country_frequencies(amazon_df)
country_frequencies_amazon = cleanup_countries(country_frequencies_amazon)
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






st.write("Where were Netflix movies shot?")
geo_map(geo_netflix_df)  
st.write("Where were Amazon movies shot?")
geo_map(geo_amazon_df)  

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
    
