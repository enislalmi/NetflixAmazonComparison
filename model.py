import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pytest import param
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
import streamlit as st
import plotly.graph_objs as go
import plotly as py
import plotly.express as px
import plotly.io as pio
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
import warnings
from sympy import E

warnings.filterwarnings('ignore')


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


def merge_on_title(df1, df2):
    return pd.merge(df1, df2, how='inner', on=['title'])


netflix_df = pd.read_csv("netflix_titles.csv")
amazon_df = pd.read_csv("amazon_prime_titles.csv")
imdb_df = pd.read_csv('imdb_titles.csv')
amazon_imdb_df = merge_on_title(amazon_df, imdb_df)
netflix_imdb_df = merge_on_title(netflix_df, imdb_df)
model_df = fix_modelling_data(amazon_imdb_df, netflix_imdb_df)

movies_df, shows_df = fix_modelling_data(amazon_imdb_df, netflix_imdb_df)

# Movies dataset

X_movies = movies_df[['title', 'director', 'leading_actor', 'listed_in']]
y_movies = movies_df['vote_average']


ohe = preprocessing.OneHotEncoder(handle_unknown='ignore')

ohe_encoded = pd.DataFrame()


# one hot encoding, each records gets a unique 01 number
for column in ['title', 'director', 'leading_actor', 'listed_in']:

    data = ohe.fit_transform(X_movies[column].values.reshape(-1, 1)).toarray()

    column_names = ohe.get_feature_names([column])

    ohe_encoded = pd.concat(
        [pd.DataFrame(data, columns=column_names), ohe_encoded], axis=1)


#IndexError, therefore reset_index
ohe_encoded = ohe_encoded.reset_index(drop=True)
duration = movies_df['duration'].reset_index(drop=True)
# duration is not a categorical feature therefore no encoding needed
X_encoded = pd.concat([ohe_encoded, duration], axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y_movies, test_size=0.2, random_state=42)

#Grid Search Function gave me this parameters as the best parameters to use

rf_reg = RandomForestRegressor(bootstrap=True, n_estimators=300,
                               max_depth=60, max_features=50, min_samples_split=2, verbose=1, n_jobs=2)

#train_MSE = 0.6535916391533556
#test_MSE = 1.2269102747161142




rf_reg.fit(X_train, y_train)
train_pred_y = rf_reg.predict(X_train)
test_pred_y = rf_reg.predict(X_test)

print(X_train.shape)
print(X_train)
print(y_train.shape)
feature_imp = pd.Series(plt.clf.feature_importances_,index=movies_df.feature_names).sort_values(ascending=False)
print(feature_imp)
#(1941, 5655)
#(1941,)
# plt.figure()
# plt.scatter(X_train, y_train, s=20, edgecolor="black", c="darkorange", label="data")
# plt.plot(X_test, y_test, color="cornflowerblue", label="max_depth=2", linewidth=2)

# plt.xlabel("data")
# plt.ylabel("target")
# plt.title("Decision Tree Regression")
# plt.legend()
# plt.show()
# print(f"train_MSE = {mean_squared_error(y_train, train_pred_y)}")
# print(f"test_MSE = {mean_squared_error(y_test, test_pred_y)}")




#Reference - Towards Data Science Article: Random Forest in Python Author: Will Koehrson



def evaluate (test_pred_y, y_test):
    errors = abs(test_pred_y - y_test)
    mape = 100 * (np.mean(errors/test_pred_y))
    accuracy = 100 - mape
    print('Model Performance')
    print('Average Error: {:0.4f} degrees.'.format(np.mean(errors)))
    print('Accuracy = {:0.2f}%.'.format(accuracy))



#Evaluating the data:
# print("Input data", X_test.iloc[0])
# print("Expected output",y_test.iloc[0]) #Tr.6.7 Te. 6.6
# print("Predicted output", rf_reg.predict(X_test.iloc[0].values.reshape(1,-1))) #Tr6.31 Te.6.5