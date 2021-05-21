import os
import pickle
import pandas as pd
import csv
import matplotlib.pyplot as plt

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


df = pd.read_excel("StanceDataset/full_data.xlsx").drop('Opinion Towards', axis=1)

df['Target'].replace({'Hillary Clinton': 'hillary',
            'Feminist Movement': 'feminist',
            'Legalization of Abortion':'abortion',
            'Donald Trump':'trump',
            'Climate Change is a Real Concern':'climate',
            'Atheism':'atheism'}, inplace=True)

df['Target'].value_counts().plot.bar(rot=45)
plt.savefig('counts_per_domain.png', bbox_inches='tight')


# df_train.drop(['Tweet', 'Stance'], axis=1).groupby(by=['Target', 'Sentiment']).size().plot.bar()
pd.pivot_table(df.drop(['Tweet', 'Stance'], axis=1), index='Target',
               columns='Sentiment', aggfunc='size').plot.bar(rot=45)
plt.title("#Records per sentiment and domain")
plt.ylabel("Count")
plt.xlabel("Domain")

# pd.pivot_table(df_train.drop(['Tweet', 'Stance'], axis=1), index = 'Target',
#                columns = 'Sentiment',aggfunc ='size').plot.bar()
plt.savefig('counts_per_sentiment_per_domain.png', bbox_inches='tight')
plt.show()

pass
