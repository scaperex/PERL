import os
import pickle
import pandas as pd
import csv
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split, StratifiedKFold

import preprocessor as p
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import mutual_info_score
import numpy as np

sns.set_style("ticks")


def analyze_stance_data(data_path):
    df = pd.read_excel(data_path)
    df['Target'].replace({'Hillary Clinton': 'Hillary',
                          'Feminist Movement': 'Feminist',
                          'Legalization of Abortion': 'Abortion',
                          'Donald Trump': 'Trump',
                          'Climate Change is a Real Concern': 'Climate Change',
                          'Atheism': 'Atheism'}, inplace=True)

    df['Sentiment'].replace({'pos': 'Positive',
                             'neg': 'Negative',
                             'other': 'Neutral'}, inplace=True)
    df['Stance'].replace({'FAVOR': 'Favor',
                          'AGAINST': 'Against',
                          'NONE': 'Neutral'}, inplace=True)
    # df['Target'].value_counts().plot.bar(rot=45)
    # plt.savefig('counts_per_domain.png', bbox_inches='tight')

    # df.drop(['Tweet', 'Stance'], axis=1).groupby(by=['Target', 'Sentiment']).size().plot.bar()
    pd.pivot_table(df.drop(['Tweet', 'Stance'], axis=1), index='Target',
                   columns='Sentiment', aggfunc='size').plot.bar(rot=45, color={'Positive': 'tab:green',
                                                                                'Negative': 'tab:red',
                                                                                'Neutral': 'tab:gray'})
    # plt.title("Number of Instances per Domain, by Sentiment Label")
    plt.ylabel("Count")
    plt.xlabel("Domain")
    plt.savefig('counts_per_sentiment_per_domain.png', bbox_inches='tight')
    plt.show()

    pd.pivot_table(df.drop(['Tweet', 'Sentiment'], axis=1), index='Target',
                   columns='Stance', aggfunc='size').plot.bar(rot=45, color={'Favor': 'tab:green',
                                                                             'Against': 'tab:red',
                                                                             'Neutral': 'tab:gray'})
    # plt.title("Number of Instances per Domain, by Stance Label")
    plt.ylabel("Count")
    plt.xlabel("Domain")

    plt.savefig('counts_per_stance_per_domain.png', bbox_inches='tight')
    plt.show()
    # y_sentiment = df["Sentiment"]
    # y_stance = df["Stance"]
    # y_sentiment.replace({'Positive': 1,
    #                          'Negative': 0,
    #                          'Neutral': 2}, inplace=True)
    # y_stance.replace({'Favor': 1,
    #                           'Against': 0,
    #                           'Neutral': 2}, inplace=True)
    # ax = sns.heatmap(data=confusion_matrix(y_sentiment.values, y_stance.values, labels=[0,1,2]),
    #                  annot=True,
    #                  fmt='d',
    #                  yticklabels=['negative', 'positive', 'other'],
    #                  xticklabels=['against', 'favor', 'none'])
    # plt.savefig('labels_corrs.png', bbox_inches='tight')
    # plt.show()
    #
    # print(f"MI: {mutual_info_score(y_stance, y_sentiment)}")


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
    return tweet


def preprocess_stance_data(data_path, label_name=None):
    if label_name:
        df = pd.read_excel(data_path)
        df = df.drop('Opinion Towards', axis=1)
        if label_name == 'Sentiment':
            df['Sentiment'].replace({'pos': 1,
                                     'neg': 0,
                                     'other': None}, inplace=True)

        elif label_name == 'Stance':
            df['Stance'].replace({'FAVOR': 1,
                                  'AGAINST': 0,
                                  'NONE': None}, inplace=True)
        df.dropna(inplace=True)
    else:
        df = pd.read_csv('RawStanceDataset/domain_tweets_all.txt', sep='\t', header=None)
        df.columns = ['tweet_ID', 'date', 'Tweet', 'hashtags', 'Target']
        # remove query hashtags and urls from tweets

        df['Tweet'] = df.apply(lambda row: filter_hashtags(row['Tweet'], row['hashtags']), axis=1)
        # Remove tweets with multiple classes
        df = df[df["Target"].str.contains(",") == False]

    # p.OPT.NUMBER, p.OPT.HASHTAG
    p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.RESERVED, p.OPT.EMOJI, p.OPT.SMILEY)
    df['Tweet'] = df.apply(lambda row: p.clean(row['Tweet']), axis=1)
    df['Tweet'].replace({'#': ' '}, inplace=True, regex=True)

    df['Tweet'] = df['Target'] + " " + df['Tweet']

    df['Target'].replace({'Hillary Clinton': 'hillary',
                          'Feminist Movement': 'feminist',
                          'Legalization of Abortion': 'abortion',
                          'Donald Trump': 'trump',
                          'Climate Change is a Real Concern': 'climate',
                          'Atheism': 'atheism'}, inplace=True)

    # Save files seperately per class
    for _class in ['hillary', 'feminist', 'abortion', 'trump',
                   'climate', 'atheism']:
        save_path = os.path.join(DATA_DIR, _class)
        os.makedirs(save_path, exist_ok=True)
        x = df.loc[df["Target"] == _class, 'Tweet'].tolist()

        if label_name:
            y = df.loc[df["Target"] == _class, label_name].tolist()
            print(f'Saving {len(x)} label_name={label_name} {_class} tweets to {save_path}. Positive/Favor Proportion: {sum(y) / len(y)}')
            with open(os.path.join(save_path, 'test'), 'wb') as f:
                pickle.dump((x, y), f)

            fold_test_save_path = os.path.join("5-fold_" + save_path, 'test')
            os.makedirs("5-fold_" + save_path, exist_ok=True)
            with open(fold_test_save_path, 'wb') as f:
                pickle.dump((x, y), f)

            skf = StratifiedKFold(n_splits=5)
            x_array = np.array(x)
            y_array = np.array(y)
            for fold_index, (train_index, test_index) in enumerate(skf.split(x_array, y_array)):
                x_train, x_test = x_array[train_index].tolist(), x_array[test_index].tolist()
                y_train, y_test = y_array[train_index].tolist(), y_array[test_index].tolist()
                fold_save_path = os.path.join("5-fold_" + DATA_DIR, _class, f'fold-{fold_index + 1}')
                os.makedirs(fold_save_path, exist_ok=True)
                with open(os.path.join(fold_save_path, 'train'), 'wb') as f:
                    pickle.dump((x_train, y_train), f)
                with open(os.path.join(fold_save_path, 'dev'), 'wb') as f:
                    pickle.dump((x_test, y_test), f)

            x_train, x_dev, y_train, y_dev = train_test_split(x, y, test_size=0.15, stratify=y, random_state=42)
            with open(os.path.join(save_path, 'train'), 'wb') as f:
                pickle.dump((x_train, y_train), f)
            with open(os.path.join(save_path, 'dev'), 'wb') as f:
                pickle.dump((x_dev, y_dev), f)
        else:
            num_examples = 20000
            print(f"Saving {num_examples} label_name={label_name} {_class} tweets to {save_path}")
            with open(os.path.join(save_path, 'unlabeled'), 'wb') as f:
                pickle.dump(x[:num_examples], f)


if __name__ == '__main__':
    # label = 'Stance'
    label = 'Sentiment'
    DATA_DIR = 'stancedata' if label == 'Stance' else 'data'
    # preprocess_stance_data("RawStanceDataset/full_data.xlsx", label_name=label)
    # preprocess_stance_data('RawStanceDataset/domain_tweets_all.txt')
    analyze_stance_data("RawStanceDataset/full_data.xlsx")
