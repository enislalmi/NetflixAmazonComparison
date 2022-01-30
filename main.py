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

train_plot,test_plot  = run_model()



st.title("Netflix VS Amazon - Comparison - From a Data Science Point of View")
st.header("Seeing two giants of streaming platforms through data")
st.sidebar.markdown(
    "A comparison between Netflix and Amazon Prime as Streaming Platforms")
st.sidebar.markdown("On this website you will be able to see the main differences between these two giants in the listings they offer. We will try to answer some of your questions using their databases and visualizing the most important features. Enjoy!")


netflix_df = pd.read_csv("netflix_titles.csv")
amazon_df = pd.read_csv("amazon_prime_titles.csv")
imdb_df = pd.read_csv('imdb_titles.csv')

# ''' Function to delete the ids, as the title is a unique value and having the ids is just
# unneccesary'''


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

    # '''Something that was seen was that most of the TV Shows did not have a director, that is
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


# For the movies/tv shows that have more than country, I will be countring only the first country
# which coincides with the country where the movie was filmed the most

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


def geo_map(df):
    data = dict(
        type='choropleth',
        locations=df['country'],
        locationmode='country names',
        z=df['freq'])
    map = go.Figure(data=[data])
    map.add_annotation(dict(font=dict(color='yellow', size=15),
                            x=0,
                            y=-0.12,
                            showarrow=False,
                            text="As we can see from this graph the United States and India have the greatest distribution of movies.",
                            textangle=0,
                            xanchor='left',
                            xref="paper",
                            yref="paper"))
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

# the difference between added and released is that added is when the movies are added to the platform
# released is when the movie is released


def listings_added_by_year(df):
    df = change_date_format(df)
    year_added_frequencies = df['date_added'].value_counts().to_dict()
    year_added_frequencies = pd.DataFrame(
        year_added_frequencies.items(), columns=['Year Added', 'Frequency'])
    fig = px.line(year_added_frequencies, x='Year Added', y='Frequency',
                  title='Movies added by year', width=800, height=400)

    fig.update_xaxes(
        dtick="M1",
        tickformat="%b\n%Y")

    st.plotly_chart(fig)


def filling_missing_data_model(df):
    #'''For the country missing data the most common country could be used to fill null values'''
    df['country_x'] = df['country_x'].fillna(df['country_x'].mode()[0])

    # '''Something that was seen was that most of the TV Shows did not have a director, that is
   # because most of them have multiple directors'''
    df['director_x'].replace(np.nan, 'Multiple Directors', inplace=True)

    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    return df


def listings_frequencies(df):
    fig = px.scatter(df, y=df['Type of Listing'], x=df["Frequency"])
    fig.update_traces(marker_size=10)

    st.plotly_chart(fig)


def bar_frequencies_both(shows_amazon_df, shows_netflix_df):
    amazon_platform = 'Amazon'
    netflix_platform = 'Netflix'
    shows_amazon_df['Streaming'] = amazon_platform
    shows_netflix_df['Streaming'] = netflix_platform
    frames = [shows_amazon_df, shows_netflix_df]

    result = pd.concat(frames)
    fig = px.bar(result, x="Streaming", y="Frequency", color="Type of Listing",
                 title="Comparison between listings in the two streaming platforms")

    st.plotly_chart(fig)


def content_rating(df):

    fig = px.pie(df, values=df['rating'].value_counts(), names=df['rating'].dropna().unique(),
                 title='Ratings of the listings:')
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.add_annotation(dict(font=dict(color='yellow', size=15),
                            x=0,
                            y=-0.12,
                            showarrow=False,
                            text="If you are sharing your account with your family, kids or friends?",
                            textangle=0,
                            xanchor='left',
                            xref="paper",
                            yref="paper"))
    st.plotly_chart(fig)


def released_available(df):
    df = change_date_format(df)
    fig = px.scatter(df.dropna(), x="date_added",
                     y='release_year', color="duration", symbol="type")

    st.plotly_chart(fig)


def newest_oldest_listing():
    platform = ['Netflix', 'Amazon', 'Netflix', 'Amazon']
    year = [netflix_df['release_year'].max(), amazon_df['release_year'].max(
    ), netflix_df['release_year'].min(), amazon_df['release_year'].min()]

    data = [dict(
        type='scatter',
        x=platform,
        y=year,
        mode='markers',

    )]

    layout = dict(
        title='Newest and Oldest movies across both platforms'
    )

    fig_dict = dict(data=data, layout=layout)

    st.plotly_chart(fig_dict)


def merge_on_title(df1, df2):
    return pd.merge(df1, df2, how='inner', on=['title'])


def only_needed_columns_model(df):
    df = df[['title', 'type_x', 'director_x',
             'duration_x', 'listed_in_x', 'rating_x']]
    return df.rename(columns={'type_x': 'type', 'director_x': 'director', 'duration_x': 'duration', 'listed_in_x': 'listed_in', 'rating_x': 'rating'})


def titles_in_common(df):

    titles = df['title']
    type_of_listing = df['type_x']

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Titles in common between Amazon and Netflix</b>',
                    '<b>Type of Listing</b>'],
            line_color='white', fill_color='white',
            align='center', font=dict(color='black', size=12)
        ),
        cells=dict(
            values=[titles, type_of_listing],
            line_color='black',
            fill_color='rgb(49, 130, 189)',
            align='center', font=dict(color='white', size=11)
        ))
    ])
    fig.add_annotation(dict(font=dict(color='yellow', size=15),
                            x=0,
                            y=-0.12,
                            showarrow=False,
                            text="How many listings can you watch on both platforms? Is it worth having both of them?",
                            textangle=0,
                            xanchor='left',
                            xref="paper",
                            yref="paper"))
    st.plotly_chart(fig)


def title_statistics(df1, df2):
    netflix_titles = set(df1['title'])
    amazon_titles = set(df2['title'])
    merged_titles = merge_on_title(df1, df2)
    common_titles = set(merged_titles['title'])
    unique_titles_netflix = netflix_titles.difference(common_titles)
    unique_titles_amazon = amazon_titles.difference(common_titles)
    fig = px.bar(x=["Netflix", "Amazon", "Common"], y=[len(unique_titles_netflix), len(
        unique_titles_amazon), len(common_titles)], title="Listings Statistics", color=['red', 'blue', 'orange'])
    fig.update_layout(
        title="Netflix and Amazon in numbers",
        xaxis_title="Platforms",
        yaxis_title="Number of Listings",
        font=dict(
            family="Arial, monospace",
            size=10,
            color="white"
        )
    )

    st.plotly_chart(fig)


def fix_modelling_data(concat_df1, concat_df2):
    all_imdb = pd.concat([concat_df1, concat_df2])
    all_imdb_up = all_imdb[['title', 'director', 'type',
                            'duration', 'listed_in', 'vote_average', 'cast']]
    all_imdb_up.drop_duplicates(subset='title', keep='first')
    all_imdb_up['cast'] = all_imdb_up['cast'].str.split(",", n=1, expand=True)
    all_imdb_up = all_imdb_up.rename(columns={'cast': 'leading_actor'})
    all_imdb_up['director'].replace(np.nan, 'Multiple Directors', inplace=True)
    # leading actor null rate is 2.14 so im dropping those columns
    all_imdb_up = all_imdb_up.dropna()
    all_imdb_up['leading_actor'] = all_imdb_up.leading_actor.apply(
        lambda x: x.split(",")[0])
    return all_imdb_up


def records_for_prediction(df):
    # all_imdb[['title','director', 'type', 'duration', 'listed_in', 'vote_average','cast']]

    #type, listedin, director, vote_average, duration, title
    title = df['title']
    type_of_listing = df['type']
    listed_in = df['listed_in']
    director = df['director']
    leading_actor = df['leading_actor']
    duration = df['duration']
    vote_average = df['vote_average']

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Title</b>', '<b>Type of Listing</b>', '<b>Listed in</b>',
                    '<b>Director</b>', '<b>Leading Actor</b>', '<b>Duration</b>', '<b>Vote average</b>'],
            line_color='rgb(18,139,181)', fill_color='rgb(222,181,34)',
            align='center', font=dict(color='rgb(12,11,0)', size=12)
        ),
        cells=dict(
            values=[title, type_of_listing, listed_in, director,
                    leading_actor, duration, vote_average],
            line_color='rgb(18,139,181)',
            fill_color='rgb(222,181,34)',
            align='center', font=dict(color='white', size=11)
        ))
    ])
    st.plotly_chart(fig)


def year_frequencies(df):
    return df['release_year'].value_counts()


def count_values(df):

    dict = df.to_dict()

    return {key: (value+dict.get(key))
            for key, value in dict.items()}


def listings_by_year(df):
    fig = px.scatter(df, x=df['Year Released'], y=df['Frequency'],
                     size=year_freq_netflix_df['Frequency'], color=year_freq_netflix_df['Year Released'],
                     log_x=True, size_max=60)
    st.plotly_chart(fig)


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
geo_map(geo_netflix_df)
st.subheader("Where were Amazon movies shot?")
geo_map(geo_amazon_df)
st.subheader("What kind of shows are we watching on Netfix?")
netflix_shows_freq = show_frequencies(netflix_df)
netflix_shows_freq = cleanup_multiple_listings(netflix_shows_freq)
shows_netflix_df = pd.DataFrame(list(netflix_shows_freq.items()), columns=[
                                'Type of Listing', 'Frequency'])
listings_frequencies(shows_netflix_df)


st.subheader("What kind of shows are we watching on Amazon?")
amazon_shows_freq = show_frequencies(amazon_df)
amazon_shows_freq = cleanup_multiple_listings(amazon_shows_freq)
shows_amazon_df = pd.DataFrame(list(amazon_shows_freq.items()), columns=[
                               'Type of Listing', 'Frequency'])
listings_frequencies(shows_amazon_df)

st.subheader(
    "When were the listings released and when did they become available to us?")
st.markdown("Netflix")
released_available(netflix_df)
st.subheader(
    "How are the two platforms divided in the type of listings they have?")
bar_frequencies_both(shows_amazon_df, shows_netflix_df)
organize_ratings(amazon_df)
st.subheader(
    "Which is the target audience of Netflix? How is the content rating of its shows? ")
content_rating(netflix_df)

st.subheader(
    "Which is the target audience of Amazon? How is the content rating of its shows? ")
content_rating(amazon_df)
st.subheader("What are the oldest and newest listings?")

newest_oldest_listing()


st.subheader("How are the movies added by years in Netflix?")
listings_added_by_year(netflix_df)

st.subheader("What are the titles in common between Netflix and Amazon?")
common_titles = merge_on_title(netflix_df, amazon_df)
titles_in_common(common_titles)
st.subheader("Show me some statistics!")
title_statistics(netflix_df, amazon_df)


st.subheader("Let's see a comparison between distribution of movies")
st.markdown("Netflix")
year_freq_netflix = year_frequencies(netflix_df)
year_freq_netflix_dict = count_values(year_freq_netflix)
year_freq_netflix_df = pd.DataFrame(
    list(year_freq_netflix_dict.items()), columns=['Year Released', 'Frequency'])
listings_by_year(year_freq_netflix_df)

st.markdown("What about Amazon?")

year_freq_amazon = year_frequencies(amazon_df)
year_freq_amazon_dict = count_values(year_freq_amazon)
year_freq_amazon_df = pd.DataFrame(list(year_freq_amazon_dict.items()), columns=[
                                   'Year Released', 'Frequency'])
listings_by_year(year_freq_netflix_df)

amazon_imdb_df = merge_on_title(amazon_df, imdb_df)
netflix_imdb_df = merge_on_title(netflix_df, imdb_df)
model_df = fix_modelling_data(amazon_imdb_df, netflix_imdb_df)
st.write("Modelling based on what?")
records_for_prediction(model_df)

st.subheader("Modelling of the Training Data")

st.plotly_chart(train_plot)

st.subheader("Modelling of the Testing Data")

st.plotly_chart(test_plot)


# if __name__ == '__main__':

#netflix_df = delete_ids(netflix_df)
#amazon_df = delete_ids(amazon_df)
#netflix_df = fill_missing_data(netflix_df)
#amazon_df = fill_missing_data(amazon_df)
#netflix_df = change_date_format(netflix_df)
#amazon_df = change_date_format(amazon_df)
#netflix_country_freq = country_frequencies(netflix_df)
# cleanup_countries(netflix_country_freq)
#amazon_country_freq = country_frequencies(amazon_df)
# cleanup_countries(amazon_country_freq)
# organize_ratings(amazon_df)
