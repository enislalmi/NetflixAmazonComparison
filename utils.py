import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objs as go
import plotly as py
import plotly.express as px
import plotly.io as pio
# ''' Function to delete the ids, as the title is a unique value and having the ids is just
# unneccesary'''


netflix_df = pd.read_csv("netflix_titles.csv")
amazon_df = pd.read_csv("amazon_prime_titles.csv")
imdb_df = pd.read_csv('imdb_titles.csv')


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
    return map


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

    return fig


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

    return fig


def bar_frequencies_both(shows_amazon_df, shows_netflix_df):
    amazon_platform = 'Amazon'
    netflix_platform = 'Netflix'
    shows_amazon_df['Streaming'] = amazon_platform
    shows_netflix_df['Streaming'] = netflix_platform
    frames = [shows_amazon_df, shows_netflix_df]

    result = pd.concat(frames)
    fig = px.bar(result, x="Streaming", y="Frequency", color="Type of Listing",
                 title="Comparison between listings in the two streaming platforms")

    return fig


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
    return fig


def released_available(df):
    df = change_date_format(df)
    fig = px.scatter(df.dropna(), x="date_added",
                     y='release_year', color="duration", symbol="type")

    return fig


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

    return fig_dict


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
    return fig


#color palette differs in the functions
def show_dataframe_netflix(df):
    
    
#type	title	director	cast	country	date_added	release_year	rating	duration	listed_in	description

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Title</b>',
                    '<b>Type of Listing</b>',
                    '<b>Director</b>','<b>Cast</b>','<b>Country</b>','<b>Date added</b>',
                    '<b>Release Year</b>','<b>Rating</b>','<b>Duration</b>','<b>Listed in</b>'],
            line_color='rgb(86,77,77)', fill_color='rgb(131,16,16)',
            align='center', font=dict(color='black', size=12)
        ),
        cells=dict(
            values=[df['title'], df['type'], df['director'], df['cast'], df['country'], df['date_added'], df['release_year'], df['rating'], df['duration'], df['listed_in']],
            line_color='rgb(86,77,77)',
            fill_color='rgb(131,16,16)',
            align='center', font=dict(color='white', size=10)
        ))
    ])
    fig.add_annotation(dict(font=dict(color='yellow', size=17),
                            x=0,
                            y=-0.12,
                            showarrow=False,
                            text="What dataframes are we using?",
                            textangle=0,
                            xanchor='left',
                            xref="paper",
                            yref="paper"))
    return fig


def show_dataframe_amazon(df):
    
    
#type	title	director	cast	country	date_added	release_year	rating	duration	listed_in	description

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=['<b>Title</b>',
                    '<b>Type of Listing</b>',
                    '<b>Director</b>','<b>Cast</b>','<b>Country</b>','<b>Date added</b>',
                    '<b>Release Year</b>','<b>Rating</b>','<b>Duration</b>','<b>Listed in</b>'],
            line_color='rgb(206, 169, 104)', fill_color=' rgb(45, 191, 248)',
            align='center', font=dict(color='black', size=12)
        ),
        cells=dict(
            values=[df['title'], df['type'], df['director'], df['cast'], df['country'], df['date_added'], df['release_year'], df['rating'], df['duration'], df['listed_in']],
            line_color='rgb(206, 169, 104)',
            fill_color='rgb(228, 192, 131)',
            align='center', font=dict(color='white', size=10)
        ))
    ])
    fig.add_annotation(dict(font=dict(color='yellow', size=17),
                            x=0,
                            y=-0.12,
                            showarrow=False,
                            text="What dataframes are we using?",
                            textangle=0,
                            xanchor='left',
                            xref="paper",
                            yref="paper"))
    return fig

def title_statistics(df1, df2):
    netflix_titles = set(df1['title'])
    amazon_titles = set(df2['title'])
    merged_titles = merge_on_title(df1, df2)
    common_titles = set(merged_titles['title'])
    unique_titles_netflix = netflix_titles.difference(common_titles)
    unique_titles_amazon = amazon_titles.difference(common_titles)
    fig = px.bar(x=["Amazon", "Netflix", "Common"], y=[len(unique_titles_amazon), len(
        unique_titles_netflix), len(common_titles)], title="Listings Statistics", color=["Amazon", "Netflix", 'In Common'])
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

    return fig


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
    return fig


def year_frequencies(df):
    return df['release_year'].value_counts()


def count_values(df):

    dict = df.to_dict()

    return {key: (value+dict.get(key))
            for key, value in dict.items()}


def listings_by_year(df):
    fig = px.scatter(df, x=df['Year Released'], y=df['Frequency'],
                     size=df['Frequency'], color=df['Year Released'],
                     log_x=True, size_max=60)
    return fig
