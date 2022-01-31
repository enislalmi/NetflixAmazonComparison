import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objs as go
import plotly.express as px
import plotly.io as pio
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')

#the fixed data that is going to be used for prediction
def fix_modelling_data(concat_df1, concat_df2):
    all_imdb = pd.concat([concat_df1, concat_df2])
    all_imdb_up = all_imdb[['title', 'director', 'type',
                            'duration', 'listed_in', 'vote_average', 'cast']]
    all_imdb_up.drop_duplicates(subset='title', keep='first')
    #all_imdb_up['cast'] = all_imdb_up['cast'].str.split(",", n = 1, expand = True)
    #all_imdb_up = all_imdb_up.rename(columns={'cast': 'leading_actor'})
    all_imdb_up['director'].replace(np.nan, 'Multiple Directors', inplace=True)
    # leading actor null rate is 2.14 so im dropping those columns
    all_imdb_up = all_imdb_up.dropna()
    all_imdb_up['leading_actor'] = all_imdb_up.cast.apply(
        lambda x: x.split(",")[0])

    movies_df = all_imdb_up[all_imdb_up['type'].str.contains('Movie') == True]

    movies_df['duration'] = movies_df['duration'].apply(
        lambda x: x.split(" ")[0])
    movies_df['duration'].astype(int)

    shows_df = all_imdb_up[all_imdb_up['type'].str.contains('Movie') == False]

    shows_df['duration'] = shows_df['duration'].apply(
        lambda x: x.split(" ")[0])
    shows_df['duration'].astype(int)
    return movies_df, shows_df


def records_for_prediction(df):
   
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


def merge_on_title(df1, df2):
    return pd.merge(df1, df2, how='inner', on=['title'])


#evaluate is used to print the performance. Numbers I got Train: 0.8, Test:1.2
def evaluate(test_pred_y, y_test):
    errors = abs(test_pred_y - y_test)
    mape = 100 * (np.mean(errors/test_pred_y))
    accuracy = 100 - mape
    print('Performance')
    print('Accuracy = {:0.2f}%.'.format(accuracy))

# ordinary least square is used 
def measured_predicted_plot(y_test, test_pred_y):
    df = pd.DataFrame({'Real Values': y_test, 'Predicted Values': test_pred_y})
    fig = px.scatter(df, x="Real Values", y="Predicted Values",
                     trendline="ols",
                     title="Graph of the model prediction")

    return fig


# Reference - Towards Data Science Article: Random Forest in Python Author: Will Koehrson
# Evaluating the data:
# print("Input data", X_test.iloc[0])
# print("Expected output",y_test.iloc[0]) #Tr.6.7 Te. 6.6
# print("Predicted output", rf_reg.predict(X_test.iloc[0].values.reshape(1,-1))) #Tr6.31 Te.6.5


def run_model():

    netflix_df = pd.read_csv("netflix_titles.csv")
    amazon_df = pd.read_csv("amazon_prime_titles.csv")
    imdb_df = pd.read_csv('imdb_titles.csv')
    amazon_imdb_df = merge_on_title(amazon_df, imdb_df)
    netflix_imdb_df = merge_on_title(netflix_df, imdb_df)

    movies_df, shows_df = fix_modelling_data(amazon_imdb_df, netflix_imdb_df)

    # Movies dataset
    X_movies = movies_df[['title', 'director', 'leading_actor', 'listed_in']]
    y_movies = movies_df['vote_average']

    ohe = preprocessing.OneHotEncoder(handle_unknown='ignore')

    ohe_encoded = pd.DataFrame()

    # one hot encoding, each records gets a unique 01 number
    for column in ['title', 'director', 'leading_actor', 'listed_in']:

        data = ohe.fit_transform(
            X_movies[column].values.reshape(-1, 1)).toarray()

        column_names = ohe.get_feature_names([column])

        ohe_encoded = pd.concat(
            [pd.DataFrame(data, columns=column_names), ohe_encoded], axis=1)

    # IndexError, therefore reset_index
    ohe_encoded = ohe_encoded.reset_index(drop=True)
    duration = movies_df['duration'].reset_index(drop=True)
    #duration is not a categorical feature therefore no encoding needed
    X_encoded = pd.concat([ohe_encoded, duration], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y_movies, test_size=0.2, random_state=42)

    # Grid Search Function gave me this parameters as the best parameters to use

    rf_reg = RandomForestRegressor(bootstrap=True, n_estimators=300,
                                   max_depth=60, max_features=50, min_samples_split=2)

    rf_reg.fit(X_train, y_train)
    train_pred_y = rf_reg.predict(X_train)
    test_pred_y = rf_reg.predict(X_test)

    # Train data
    train_plot = measured_predicted_plot(y_train, train_pred_y)

    # Test data
    test_plot = measured_predicted_plot(y_test, test_pred_y)

    return train_plot, test_plot


if __name__ == '__main__':
    run_model()
