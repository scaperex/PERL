import os
import pickle
import pandas as pd
import csv
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
import preprocessor as p
def analyze_stance_data():
    df['Target'].value_counts().plot.bar(rot=45)
    plt.savefig('counts_per_domain.png', bbox_inches='tight')

    # df_train.drop(['Tweet', 'Stance'], axis=1).groupby(by=['Target', 'Sentiment']).size().plot.bar()
    pd.pivot_table(df.drop(['Tweet', 'Stance'], axis=1), index='Target',
                   columns='Sentiment', aggfunc='size').plot.bar(rot=45)
    plt.title("#Records per sentiment and domain")
    plt.ylabel("Count")
    plt.xlabel("Domain")

    pd.pivot_table(df_train.drop(['Tweet', 'Stance'], axis=1), index = 'Target',
                   columns = 'Sentiment',aggfunc ='size').plot.bar()
    plt.savefig('counts_per_sentiment_per_domain.png', bbox_inches='tight')
    plt.show()

def open_blitzer_data():
    src = 'books'

    src_path = "data/" + src + os.sep
    with open(src_path + "train", 'rb') as f:
        (train, train_labels) = pickle.load(f)

    with open(src_path + "unlabeled", 'rb') as f:
        unlabeled = pickle.load(f)

    with open(src_path + "dev", 'rb') as f:
        dev = pickle.load(f)

    with open(src_path + "test", 'rb') as f:
        test = pickle.load(f)



def filter_hashtags(tweet: str, hashtags: str):
    """
    Remove all hashtags that were used in the filtering collection data.
    Remove urls
    :param tweet:
    :param hashtags:
    :return:
    """
    for hashtag in hashtags.split():
        tweet = tweet.replace("#" + hashtag, "")
    tweet = re.sub(r"http\S+", "", tweet) # TODO change per decision
    return tweet

def preprocess_stance_data(data_path, is_labeled):
    if is_labeled:
        df = pd.read_excel(data_path)
        df = df.drop('Opinion Towards', axis=1)

        df['Sentiment'].replace({'pos': 1,
                                 'neg': 0,
                                 'other': None}, inplace=True)
        df.dropna(inplace=True)
        df['Sentiment'] = df['Sentiment'].astype(int)

    else:
        df = pd.read_csv('StanceDataset/domain_tweets_all.txt', sep='\t', header=None)
        df.columns = ['tweet_ID', 'date', 'Tweet', 'hashtags', 'Target']
        # remove query hashtags and urls from tweets

        df['Tweet'] = df.apply(lambda row: filter_hashtags(row['Tweet'], row['hashtags']), axis=1)
        # Remove tweets with multiple classes
        df = df[df["Target"].str.contains(",") == False]

    df['Target'].replace({'Hillary Clinton': 'hillary',
                          'Feminist Movement': 'feminist',
                          'Legalization of Abortion': 'abortion',
                          'Donald Trump': 'trump',
                          'Climate Change is a Real Concern': 'climate',
                          'Atheism': 'atheism'}, inplace=True)

    # TODO choose clean method
    # df['Tweet'].replace({'#': '', "@": ''}, inplace=True, regex=True)

    # Save files seperately per class
    for _class in ['hillary', 'feminist', 'abortion', 'trump',
                   'climate', 'atheism']:
        save_path = os.path.join('data', _class)
        os.makedirs(save_path, exist_ok=True)
        x = df.loc[df["Target"] == _class, 'Tweet'].tolist()

        print(f"Saving {len(x)} is_labeled={is_labeled} {_class} tweets to {save_path}")
        if is_labeled:
            y = df.loc[df["Target"] == _class, 'Sentiment'].tolist()
            with open(os.path.join(save_path, 'test'), 'wb') as f:
                pickle.dump((x, y), f)
            x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.15, stratify=y, random_state=42)
            with open(os.path.join(save_path, 'train'), 'wb') as f:
                pickle.dump((x_train, y_train), f)
            with open(os.path.join(save_path, 'dev'), 'wb') as f:
                pickle.dump((x_dev, y_dev), f)
        else:
            with open(os.path.join(save_path, 'unlabeled'), 'wb') as f:
                pickle.dump(x, f)

if __name__ == '__main__':
    preprocess_stance_data("StanceDataset/full_data.xlsx", is_labeled=True)
    preprocess_stance_data('StanceDataset/domain_tweets_all.txt', is_labeled=False)

