# Import required functions from scikit-learn
import copy

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import *

# Import data for classification
df_silv = pd.read_csv('silvics_edited_data.csv')
df_wiki = pd.read_csv('wikipedia_edited_data.csv')
df_merg = pd.read_csv('merged_edited_data.csv')

# Classification parameters (used for all trials)
field = 'category'
n = 10  # only consider the n most frequent classes in field
maxFeatures = 5000
test_set = 0.10
prt_class_rept = True

### Try all 3 NB models across the Silvics Manual data set ###
### Run three versions of NB: Gaussian, Complement, and Multinomial  ###

print('\n### Results for Silvics Manual Dataset ###\n')

# Silvics parameters
labels = df_silv[field].value_counts()
top_labels = labels.index.tolist()[:n]
top_df_temp = df_silv[df_silv[field].isin(top_labels)]
top_df = copy.copy(top_df_temp)
df_text = top_df['text']
df_field = top_df[field]

# Gaussian Naive Bayes model
model = GaussianNB()
model_text = 'GaussianNB'

# Create a bag of words
cv = TfidfVectorizer(stop_words='english', max_features=maxFeatures)
bag = cv.fit_transform(df_text)
bag = np.array(bag.todense())

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(bag, df_field, test_size=test_set, random_state=0)

# Train and test
y_pred = model.fit(X_train, y_train).predict(X_test)

# Performance Evaluation
F1_macro = f1_score(y_test, y_pred, average='macro')
print(
    f'F1 score: {F1_macro:.3f}\n   {model_text}\n   macro averaging\n   n = {n}\n   max_features = {maxFeatures}\n   test set = {test_set * 100}%\n')
if prt_class_rept:
    print(f'Classification Report: {model_text}')
    print((classification_report(y_test, y_pred, labels=top_labels)))

# Complement Naive Bayes model
model = ComplementNB()
model_text = 'ComplementNB'

# Train and test
y_pred = model.fit(X_train, y_train).predict(X_test)

# Performance Evaluation
F1_macro = f1_score(y_test, y_pred, average='macro')
print(
    f'F1 score: {F1_macro:.3f}\n   {model_text}\n   macro averaging\n   n = {n}\n   max_features = {maxFeatures}\n   test set = {test_set * 100}%\n')
if prt_class_rept:
    print(f'Classification Report: {model_text}')
    print(classification_report(y_test, y_pred, labels=top_labels))

# Multinomial Naive Bayes model
model = MultinomialNB()
model_text = 'MultinomialNB'

# Train and test
y_pred = model.fit(X_train, y_train).predict(X_test)

# Performance Evaluation
F1_macro = f1_score(y_test, y_pred, average='macro')
print(
    f'F1 score: {F1_macro:.3f}\n   {model_text}\n   macro averaging\n   max_features = {maxFeatures}\n   test set = {test_set * 100}%\n')
if prt_class_rept:
    print(f'Classification Report: {model_text}')
    print(classification_report(y_test, y_pred, labels=top_labels))

### Try all 3 NB models across the Wikipedia data set ###
### Run three versions of NB: Gaussian, Complement, and Multinomial  ###

print('\n### Results for Wikipedia Dataset ###\n')

# Wikipedia parameters
labels = df_wiki[field].value_counts()
top_labels = labels.index.tolist()[:n]
top_df_temp = df_wiki[df_wiki[field].isin(top_labels)]
top_df = copy.copy(top_df_temp)
df_text = top_df['text']
df_field = top_df[field]

# Gaussian Naive Bayes model
model = GaussianNB()
model_text = 'GaussianNB'

# Create a bag of words
cv = TfidfVectorizer(stop_words='english', max_features=maxFeatures)
bag = cv.fit_transform(df_text)
bag = np.array(bag.todense())

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(bag, df_field, test_size=test_set, random_state=0)

# Train and test
y_pred = model.fit(X_train, y_train).predict(X_test)

# Performance Evaluation
F1_macro = f1_score(y_test, y_pred, average='macro')
print(
    f'F1 score: {F1_macro:.3f}\n   {model_text}\n   macro averaging\n   n = {n}\n   max_features = {maxFeatures}\n   test set = {test_set * 100}%\n')
if prt_class_rept:
    print(f'Classification Report: {model_text}')
    print(classification_report(y_test, y_pred, labels=top_labels))

# Complement Naive Bayes model
model = ComplementNB()
model_text = 'ComplementNB'

# Train and test
y_pred = model.fit(X_train, y_train).predict(X_test)

# Performance Evaluation
F1_macro = f1_score(y_test, y_pred, average='macro')
print(
    f'F1 score: {F1_macro:.3f}\n   {model_text}\n   macro averaging\n   n = {n}\n   max_features = {maxFeatures}\n   test set = {test_set * 100}%\n')
if prt_class_rept:
    print(f'Classification Report: {model_text}')
    print(classification_report(y_test, y_pred, labels=top_labels))

# Multinomial Naive Bayes model
model = MultinomialNB()
model_text = 'MultinomialNB'

# Train and test
y_pred = model.fit(X_train, y_train).predict(X_test)

# Performance Evaluation
F1_macro = f1_score(y_test, y_pred, average='macro')
print(
    f'F1 score: {F1_macro:.3f}\n   {model_text}\n   macro averaging\n   n = {n}\n   max_features = {maxFeatures}\n   test set = {test_set * 100}%\n')
if prt_class_rept:
    print(f'Classification Report: {model_text}')
    print(classification_report(y_test, y_pred, labels=top_labels))

### Try all 3 NB models across the Merged data set ###
### Run three versions of NB: Gaussian, Complement, and Multinomial  ###

print('\n### Results for Merged Dataset ###\n')

# Merged parameters
labels = df_merg[field].value_counts()
top_labels = labels.index.tolist()[:n]
top_df_temp = df_merg[df_merg[field].isin(top_labels)]
top_df = copy.copy(top_df_temp)
df_text = top_df['text']
df_field = top_df[field]

# Gaussian Naive Bayes model
model = GaussianNB()
model_text = 'GaussianNB'

# Create a bag of words
cv = TfidfVectorizer(stop_words='english', max_features=maxFeatures)
bag = cv.fit_transform(df_text)
bag = np.array(bag.todense())

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(bag, df_field, test_size=test_set, random_state=0)

# Train and test
y_pred = model.fit(X_train, y_train).predict(X_test)

# Performance Evaluation
F1_macro = f1_score(y_test, y_pred, average='macro')
print(
    f'F1 score: {F1_macro:.3f}\n   {model_text}\n   macro averaging\n   n = {n}\n   max_features = {maxFeatures}\n   test set = {test_set * 100}%\n')
if prt_class_rept:
    print(f'Classification Report: {model_text}')
    print(classification_report(y_test, y_pred, labels=top_labels))

# Complement Naive Bayes model
model = ComplementNB()
model_text = 'ComplementNB'

# Train and test
y_pred = model.fit(X_train, y_train).predict(X_test)

# Performance Evaluation
F1_macro = f1_score(y_test, y_pred, average='macro')
print(
    f'F1 score: {F1_macro:.3f}\n   {model_text}\n   macro averaging\n   n = {n}\n   max_features = {maxFeatures}\n   test set = {test_set * 100}%\n')
if prt_class_rept:
    print(f'Classification Report: {model_text}')
    print(classification_report(y_test, y_pred, labels=top_labels))

# Multinomial Naive Bayes model
model = MultinomialNB()
model_text = 'MultinomialNB'

# Train and test
y_pred = model.fit(X_train, y_train).predict(X_test)

# Performance Evaluation
F1_macro = f1_score(y_test, y_pred, average='macro')
accuracy = accuracy_score(y_test, y_pred)
print(
    f'F1 score: {F1_macro:.3f}\nAccuracy: {accuracy:0.3f}\n   {model_text}\n   macro averaging\n   n = {n}\n   max_features = {maxFeatures}\n   test set = {test_set * 100}%\n')
if prt_class_rept:
    print(f'Classification Report: {model_text}')
    print(classification_report(y_test, y_pred, labels=top_labels))