# %%

import snscrape.modules.twitter as sntwitter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import chain
import re


# %%
query = '(from:Dior)'
tweets= []
count=0
for t in sntwitter.TwitterSearchScraper(query).get_items():
    if (count<1000):
        tweets.append([t.content])
    else:
        break
    count+=1

print(len(tweets))


# %%
df = pd.DataFrame(tweets, columns=['content'])
print("TASK1: Scrapped Data")
print(df.head(10))
df.to_csv('csvs/Dior_task1.csv', encoding='utf-8' ,index=False)


# %%
from string import punctuation
df = pd.read_csv('E:\D drive\Fall 2022\Machine Learning\project\phase 1\csvs\phase1.csv', index_col=False)
f_stop_words = open("stop_words.txt")
stop_words=f_stop_words.read()
stop_words = stop_words.translate(str.maketrans('', '', punctuation))
stop_words = stop_words.split("\n")
cleaned=[]
bow = []
for i,row in enumerate(df.iterrows()):
    data = row[1][0]
    data = data.lower()
    data=data.replace('\n',' ')
    '''
    remove emojis, emoticons, map and flags
    
    for unicodes used source: https://stackoverflow.com/questions/33404752/removing-emojis-from-a-string-in-python
    '''
    emoji_pattern = re.compile("["u"\U0001F600-\U0001F64F" u"\U0001F300-\U0001F5FF"  u"\U0001F680-\U0001F6FF" u"\U0001F1E0-\U0001F1FF""]+", flags=re.UNICODE)
    data = emoji_pattern.sub(r'', data)
    data = re.sub("\d+", "", data) # remove digits
    for s in stop_words:
        if (s in data):
            data.replace(s,' ')
    data=data.split(' ')
    d=[]
    for word in data:
        if (len(word)>1):
            if (word[0]!='@' and word[0]!='#' and word[0:5]!='https' ):  #remove hashtags, mentions and urls
                temp = word.translate(str.maketrans("","",punctuation))
                bow.append(temp)
                d.append(temp)
    temp = " ".join(d)
    
    if (len(temp)>1):
        cleaned.append(temp)


# %%
print(len(cleaned))
print(len(bow))

# %%
df2 = pd.DataFrame(cleaned, columns=['content'])
print("TASK2: Cleaned Data")
print(df2.head(10))
df2.to_csv('csvs/Dior_task2.csv', encoding='utf-8',index=False)

# %%
def bow(csv):
    all = []
    for i,row in enumerate(csv.iterrows()):
        try:
            data = row[1][0]
            data=data.split(' ')
            all.append(data)
        except:
            print(i)
            print(row[1][0])
    all = list(chain.from_iterable(all))
    un=set(all)
    un = list(un)
    return sorted(un)


        

# %%
def one_hot(csv, train):
    all_words = bow(train)
    print(len(all_words))
    encode=list()
    for i,row in enumerate(csv.iterrows()):
        word_vector =  [1 for _ in range(len(all_words))]
        data = row[1][0]
        data=data.split(' ')
        for d in data:
            try:
                index = all_words.index(d)
                word_vector[index]+=1
            except:
                continue
        encode.append(word_vector)
    return encode

# %%
train_data = df2.sample(frac = 0.8 ,random_state=1)
test_data = df2.drop(train_data.index)

one_hot_train = one_hot(train_data,train_data)
one_hot_test = one_hot(test_data,train_data)
df_train = pd.DataFrame(one_hot_train)
df_test = pd.DataFrame(one_hot_test)

print("SAMPLE FEATURE VECTOR OF TRAINING DATA")
df_train.head(10)


# %%

print("SAMPLE FEATURE VECTOR OF TESTING DATA")
df_test.head(10)

# %%



