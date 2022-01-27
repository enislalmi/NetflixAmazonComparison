import pandas as pd
import numpy as np

netflix_df = pd.read_csv("netflix_titles.csv")
amazon_df = pd.read_csv("amazon_prime_titles.csv")


''' Function to delete the ids, as the title is a unique value and having the ids is just
unneccesary'''

def delete_ids(df):
    df = df.drop(["show_id"], axis=1)
    return df


def check_missing_data(df):
    for i in df.columns:
        null_rate = df[i].isna().sum() / len(df) * 100 
        if null_rate > 0 :
            print(f"{i} null rate: {round(null_rate,2)}%")
        else:
            print(f"Column {i} has no missing data.")

def fill_missing_data(df):

    '''For the country missing data the most common country could be used to fill null values'''
    df['country'] = df['country'].fillna(df['country'].mode()[0])

    '''Something that was seen was that most of the TV Shows did not have a director, that is
    because most of them have multiple directors'''
    df['cast'].replace(np.nan, 'Null values',inplace  = True)
    df['director'].replace(np.nan, 'Multiple Directors',inplace  = True)

  
    df.dropna(inplace=True)
    df.drop_duplicates(inplace= True)
    return df


#netflix_df = delete_ids(netflix_df)
#amazon_df = delete_ids(amazon_df)
#netflix_df = fill_missing_data(netflix_df)
#amazon_df = fill_missing_data(amazon_df)






