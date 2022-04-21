"""
This file was created for the purpose of testing out hyperparameters (especially for the feed-forward neural network)
through the Ohio supercomputer center.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import csv
import nltk
import string
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score

import numpy as np
from collections import Counter
import time

from FeedForwardNN import *

df = pd.read_csv(r'data/spam.csv', encoding="latin-1")

## Another dataset, may be we could use different dataset just to add complexity in our project and compare the results between them ?
# This file is in the spambase folder, data description can be found in the same folder in README file.

spambase_dataset = pd.read_csv(r'data/spambase_csv.csv', encoding="latin-1")

df = df.dropna(axis=1)
df = df.rename(columns={"v1": "label", "v2": "text"})
df['label'].unique()

## number of spam labeled text
len(df[df.label == "spam"])

## number of ham labeled text
len(df[df.label == "ham"])

# missing values
df.isnull().sum()

"""**Feature Engineering**"""


# Text length
def text_length(text):
    return len(text) - text.count(" ")


df['text_length'] = df.text.apply(lambda x: text_length(x))


# Count Punctuation percentage

def count_punct(text):
    count = sum([1 for char in text if char in string.punctuation])
    return round(count / (len(text) - text.count(" ")), 3) * 100


df['punctuation'] = df.text.apply(lambda x: count_punct(x))
df['text_length'].hist(bins=100)

"""**Analyzing data**"""

# convert text to lower case, later can be tokenized
df['processed_text'] = df['text'].str.lower()


def remove_punctuations(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text


df["processed_text"] = df['processed_text'].apply(remove_punctuations)

# remove the stopwords, may need to define our own stopwords later
from nltk.corpus import stopwords

# import nltk
nltk.download('stopwords')
stop = stopwords.words('english')

df["processed_text"] = df["processed_text"].apply(
    lambda words: ' '.join(word.lower() for word in words.split() if word not in stop))

other_stopwords = ['u', 'im', '2', 'ur', 'ill', '4', 'lor', 'r', 'n', 'da', 'oh']

df["processed_text"] = df["processed_text"].apply(
    lambda words: ' '.join(word.lower() for word in words.split() if word not in other_stopwords))

# tokenize text
import nltk

nltk.download('punkt')
df['processed_text'] = df.apply(lambda row: nltk.word_tokenize(row['processed_text']), axis=1)

# Collect ham words
ham_words = list(df.loc[df.label == 'ham', 'processed_text'])

# Flatten list of lists
ham_words = list(np.concatenate(ham_words).flat)

# Create dictionary to store word frequency
ham_words = Counter(ham_words)

# Collect spam words
spam_words = list(df.loc[df.label == 'spam', 'processed_text'])

# Flatten list of lists
spam_words = list(np.concatenate(spam_words).flat)

# Create dictionary to store word frequency
spam_words = Counter(spam_words)

# 50 most common ham words
ham_words = list(df.loc[df.label == 'ham', 'processed_text'])
ham_words = list(np.concatenate(ham_words).flat)
ham_words = Counter(ham_words)
ham_words = pd.DataFrame(ham_words.most_common(50), columns=['ham_word', 'frequency'])

# 50 most common spam words
spam_words = list(df.loc[df.label == 'spam', 'processed_text'])
spam_words = list(np.concatenate(spam_words).flat)
spam_words = Counter(spam_words)
spam_words = pd.DataFrame(spam_words.most_common(50), columns=['spam_word', 'frequency'])

"""Plot most common ham and spam words"""

ham_words.plot(x="ham_word", y="frequency", kind="bar", width=0.7, align='center')

plt.title("50 most common ham words")
plt.xlabel("words")
plt.ylabel("frequency")

spam_words.plot(x="spam_word", y="frequency", kind="bar", width=0.7, align='center')

plt.title("50 most common spam words")
plt.xlabel("words")
plt.ylabel("frequency")

result = pd.concat([ham_words, spam_words], axis=1, join='inner')

"""# **Make data ready for modeling**"""

x_train, x_test, y_train, y_test = train_test_split(df[['text', 'text_length', 'punctuation']], df.label, test_size=0.2, random_state=42)

# TfidfVectorizer, can be used other vectorization process
tfidf_vect = TfidfVectorizer()
tfidf_vect_fit = tfidf_vect.fit(x_train['text'])

tfidf_train = tfidf_vect.transform(x_train['text'])
tfidf_test = tfidf_vect.transform(x_test['text'])

# Recombine transformed body text with body_len and punct% features
x_train = pd.concat(
    [x_train[['text_length', 'punctuation']].reset_index(drop=True), pd.DataFrame(tfidf_train.toarray())], axis=1)
x_test = pd.concat([x_test[['text_length', 'punctuation']].reset_index(drop=True), pd.DataFrame(tfidf_test.toarray())], axis=1)


# **Gradient Boosting**

## rather than using just one iteration, using 5 iteration and average the accuracy result for better result

gb_accuracy = []
total_time = []
for i in range(5):
    model = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0).fit(x_train,
                                                                                                             y_train)

    start_time = time.time()
    model.score(x_train, y_train)
    end_time = time.time()

    total_time.append(end_time - start_time)

    y_pred = model.predict(x_test)
    gb_accuracy.append(accuracy_score(y_test, y_pred) * 100)

avg_gb_accuracy = np.mean(gb_accuracy)
avg_time = np.mean(total_time)

print(classification_report(y_test, y_pred))

print("accuracy of gradient boosting is: ", avg_gb_accuracy)
print("total run time: ", avg_time)

# **Random Forest Classifier**

rf_accuracy = []
total_time = []
for i in range(5):
    model = RandomForestClassifier(max_depth=100, random_state=42)

    start_time = time.time()
    model.fit(x_train, y_train)
    end_time = time.time()

    total_time.append(end_time - start_time)

    y_pred = model.predict(x_test)
    rf_accuracy.append(accuracy_score(y_test, y_pred) * 100)

avg_rf_accuracy = np.mean(rf_accuracy)
avg_time = np.mean(total_time)

print(classification_report(y_test, y_pred))

print("accuracy of random forest is: ", avg_rf_accuracy)
print("total run time: ", avg_time)

# **Support Vector Machine (SVM)**

svm_accuracy = []
total_time = []
for i in range(5):
    model = svm.SVC()

    start_time = time.time()
    model.fit(x_train, y_train)
    end_time = time.time()

    total_time.append(end_time - start_time)
    y_pred = model.predict(x_test)
    svm_accuracy.append(accuracy_score(y_test, y_pred) * 100)

avg_svm_accuracy = np.mean(svm_accuracy)
avg_time = np.mean(total_time)

print(classification_report(y_test, y_pred))

print("accuracy of svm is: ", avg_svm_accuracy)
print("total run time: ", avg_time)



# start training rnn
# Note that x_train at this point is roughly 275 MB, could change to sparse vector
print("===Training Feed Forward Neural Network===")
x_train = torch.tensor(x_train.values.tolist())
x_test = torch.tensor(x_test.values.tolist())
y = []
for li in y_train.values.tolist():
    y += [(0 if li == "ham" else 1)]
y_train = torch.tensor(y)
y = []
for li in y_test.values.tolist():
    y += [(0 if li == "ham" else 1)]
y_test = torch.tensor(y)

a_file = open("accuracy.csv", 'w+', newline='')
a_writer = csv.writer(a_file)
f1_file = open("f1.csv", 'w+', newline='')
f1_writer = csv.writer(f1_file)
# train the nn with different hyperparameters
for n_epochs in [5, 10, 20, 60, 120, 200]:
    a_row = []
    f1_row = []
    for lr in [0.0002, 0.0005, 0.0007, 0.001, 0.0012]:
        start_time = time.time()
        ffnn = train_feed_forward_classifier(x_train, y_train, n_epochs)
        end_time = time.time()

        accuracy, f1 = eval_nn(ffnn, x_test, y_test)
        a_row += [accuracy]
        f1_row += [f1]
    a_writer.writerow(a_row)
    f1_writer.writerow(f1_row)
a_file.close()
f1_file.close()

# note - best found was
