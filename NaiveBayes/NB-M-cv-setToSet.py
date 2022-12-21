import pandas as pd
import numpy as np
import copy
import warnings

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix


# Stop Classification Report from issuing warnings (probably other stuff, too)
def warn(*args, **kwargs):
    pass


warnings.warn = warn

# Load data for classification
df_silv = pd.read_csv('silvics_edited_data.csv')
df_wiki = pd.read_csv('wikipedia_edited_data.csv')
df_merg = pd.read_csv('merged_edited_data.csv')

# Classification parameters (used for all trials)
field = 'category'
n = 20  # only consider the n most frequent classes in field
maxFeatures = 2000
test_set1 = 0.01
test_set2 = 0.99
prt_reports = False
SilvicsLabels = df_silv[field].value_counts().index.tolist()[:n]

'''
###################################################
### Create a model from the Silvics Manual data ###
### Test the model on the Wikipedia data        ###
###################################################
'''

print('\n### Train on Silvics Manual, Test on Wikipedia ###\n')

# Silvics parameters
labels1 = df_silv[field].value_counts()
top_labels1 = labels1.index.tolist()[:n]
top_df_temp = df_silv[df_silv[field].isin(top_labels1)]
top_df1 = copy.copy(top_df_temp)
df_txt1 = top_df1['text']
df_fld1 = top_df1[field]

# Create a bag of words
cv1 = CountVectorizer(stop_words='english', max_features=maxFeatures)
bag1 = cv1.fit_transform(df_txt1)
bag1 = np.array(bag1.todense())

# Split the data into a training set and a test set
X_train1, X_test1, y_train1, y_test1 = train_test_split(bag1, df_fld1, test_size=test_set1, random_state=0)

# Multinomial Naive Bayes model
model = MultinomialNB()
model_text = 'MultinomialNB'

# Train and test
y_mod1 = model.fit(X_train1, y_train1)
y_prd1 = y_mod1.predict(X_test1)

# Performance Evaluation
F1_macro1 = f1_score(y_test1, y_prd1, average='macro')
print('Evaluation using Silvics Manual dataset to predict on Silvics Manual dataset')
print(
    f'F1 score: {F1_macro1:.3f}\n   {model_text}\n   macro averaging\n   max_features = {maxFeatures}\n   '
    f'test set = {test_set1 * 100}%\n')
if prt_reports:
    print(f'Classification Report: {model_text}')
    print(classification_report(y_test1, y_prd1, labels=top_labels1))
    print('Confusion Matrix')
    print(confusion_matrix(y_test1, y_prd1))

# Use Silvics-based model to predict for Wikipedia data

# Restrict Wikipedia data
top_df_temp = df_wiki[df_wiki[field].isin(top_labels1)]
top_df2 = copy.copy(top_df_temp)
df_txt2 = top_df2['text']
df_fld2 = top_df2[field]

# Create bag of words from Wikipedia data
cv2 = CountVectorizer(stop_words='english', max_features=maxFeatures)
bag2 = cv2.fit_transform(df_txt2)
bag2 = np.array(bag2.todense())

# Split Wikipedia data into training and test sets
X_train2, X_test2, y_train2, y_test2 = train_test_split(bag2, df_fld2, test_size=test_set2, random_state=0)

# Uset the Silvics model to classify the Wikipedia data
y_prd2 = y_mod1.predict(X_test2)

# Performance Evaluation
F1_macro2 = f1_score(y_test2, y_prd2, average='macro')
print('\nEvaluation using Silvics Manual dataset to predict on Wikipedia dataset')
print(
    f'F1 score: {F1_macro2:.3f}\n   {model_text}\n   macro averaging\n   max_features = {maxFeatures}\n   '
    f'test set = {test_set2 * 100}%\n')
if prt_reports:
    print(f'Classification Report: {model_text}')
    print(classification_report(y_test2, y_prd2, labels=top_labels1))
    print(confusion_matrix(y_test2, y_prd2))

'''
#################################################
### Create a model from the Wikipedia data    ###
### Test the model on the Silvics Manual data ###
#################################################
'''

print('\n### Train on Wikipedia, Test on Silvics Manual ###\n')

# Wikipedia parameters
# labels1 = df_wiki[field].value_counts()
# top_labels1 = labels1.index.tolist()[:n]
top_df_temp = df_wiki[df_wiki[field].isin(SilvicsLabels)]
top_df1 = copy.copy(top_df_temp)
df_txt1 = top_df1['text']
df_fld1 = top_df1[field]

# Create a bag of words
cv1 = CountVectorizer(stop_words='english', max_features=maxFeatures)
bag1 = cv1.fit_transform(df_txt1)
bag1 = np.array(bag1.todense())

# Split the data into a training set and a test set
X_train1, X_test1, y_train1, y_test1 = train_test_split(bag1, df_fld1, test_size=test_set1, random_state=0)

# Multinomial Naive Bayes model
model = MultinomialNB()
model_text = 'MultinomialNB'

# Train and test
y_mod1 = model.fit(X_train1, y_train1)
y_prd1 = y_mod1.predict(X_test1)

# Performance Evaluation
F1_macro1 = f1_score(y_test1, y_prd1, average='macro')
print('Evaluation using Wikipedia dataset to predict on Wikipedia dataset')
print(
    f'F1 score: {F1_macro1:.3f}\n   {model_text}\n   macro averaging\n   max_features = {maxFeatures}\n   '
    f'test set = {test_set1 * 100}%\n')
if prt_reports:
    print(f'Classification Report: {model_text}')
    print(classification_report(y_test1, y_prd1, labels=SilvicsLabels))
    print(confusion_matrix(y_test1, y_prd1))

# Use Wikipedia-based model to predict for Silvics Manual data

# Restrict Silvics data
top_df_temp = df_silv[df_silv[field].isin(top_labels1)]
top_df2 = copy.copy(top_df_temp)
df_txt2 = top_df2['text']
df_fld2 = top_df2[field]

# Create bag of words from Wikipedia data
cv2 = CountVectorizer(stop_words='english', max_features=maxFeatures)
bag2 = cv2.fit_transform(df_txt2)
bag2 = np.array(bag2.todense())

# Split Wikipedia data into training and test sets
X_train2, X_test2, y_train2, y_test2 = train_test_split(bag2, df_fld2, test_size=test_set2, random_state=0)

# Uset the Silvics model to classify the Wikipedia data
y_prd2 = y_mod1.predict(X_test2)

# Performance Evaluation
F1_macro2 = f1_score(y_test2, y_prd2, average='macro')
print('\nEvaluation using Wikipedia dataset to predict on Silvics Manual dataset')
print(
    f'F1 score: {F1_macro2:.3f}\n   {model_text}\n   macro averaging\n   max_features = {maxFeatures}\n   '
    f'test set = {test_set2 * 100}%\n')
if prt_reports:
    print(f'Classification Report: {model_text}')
    print(classification_report(y_test2, y_prd2, labels=top_labels1))
    print(confusion_matrix(y_test2, y_prd2))

'''
#################################################
### Create a model from the Merged data       ###
### Test the model on the Silvics Manual data ###
#################################################
'''

print('\n### Train on Merged, Test on Silvics ###\n')

# Merged parameters
# labels1 = df_wiki[field].value_counts()
# top_labels1 = labels1.index.tolist()[:n]
top_df_temp = df_merg[df_merg[field].isin(SilvicsLabels)]
top_df1 = copy.copy(top_df_temp)
df_txt1 = top_df1['text']
df_fld1 = top_df1[field]

# Create a bag of words
cv1 = CountVectorizer(stop_words='english', max_features=maxFeatures)
bag1 = cv1.fit_transform(df_txt1)
bag1 = np.array(bag1.todense())

# Split the data into a training set and a test set
X_train1, X_test1, y_train1, y_test1 = train_test_split(bag1, df_fld1, test_size=test_set1, random_state=0)

# Multinomial Naive Bayes model
model = MultinomialNB()
model_text = 'MultinomialNB'

# Train and test
y_mod1 = model.fit(X_train1, y_train1)
y_prd1 = y_mod1.predict(X_test1)

# Performance Evaluation
F1_macro1 = f1_score(y_test1, y_prd1, average='macro')
print('Evaluation using Merged dataset to predict on Merged dataset')
print(
    f'F1 score: {F1_macro1:.3f}\n   {model_text}\n   macro averaging\n   max_features = {maxFeatures}\n   '
    f'test set = {test_set1 * 100}%\n')
if prt_reports:
    print(f'Classification Report: {model_text}')
    print(classification_report(y_test1, y_prd1, labels=SilvicsLabels))
    print(confusion_matrix(y_test1, y_prd1))

# Use Merged-based model to predict for Silvics Manual data

# Restrict Silvics Manual data
top_df_temp = df_silv[df_silv[field].isin(SilvicsLabels)]
top_df2 = copy.copy(top_df_temp)
df_txt2 = top_df2['text']
df_fld2 = top_df2[field]

# Create bag of words from Silvics Manual data
cv2 = CountVectorizer(stop_words='english', max_features=maxFeatures)
bag2 = cv2.fit_transform(df_txt2)
bag2 = np.array(bag2.todense())

# Split Silvics Manual data into training and test sets
X_train2, X_test2, y_train2, y_test2 = train_test_split(bag2, df_fld2, test_size=test_set2, random_state=0)

# Use the Merged model to classify the Silvics data
y_prd2 = y_mod1.predict(X_test2)

# Performance Evaluation
F1_macro2 = f1_score(y_test2, y_prd2, average='macro')
print('\nEvaluation using Merged dataset to predict on Silvics Manual dataset')
print(
    f'F1 score: {F1_macro2:.3f}\n   {model_text}\n   macro averaging\n   max_features = {maxFeatures}\n   '
    f'test set = {test_set2 * 100}%\n')
if prt_reports:
    print(f'Classification Report: {model_text}')
    print(classification_report(y_test2, y_prd2, labels=SilvicsLabels))
    print(confusion_matrix(y_test2, y_prd2))

'''
############################################
### Create a model from the Merged data  ###
### Test the model on the Wikipedia data ###
############################################
'''

print('\n### Train on Merged, Test on Wikipedia ###\n')

# Merged parameters
labels1 = df_merg[field].value_counts()
top_labels1 = labels1.index.tolist()[:n]
top_df_temp = df_merg[df_merg[field].isin(top_labels1)]
top_df1 = copy.copy(top_df_temp)
df_txt1 = top_df1['text']
df_fld1 = top_df1[field]

# Create a bag of words
cv1 = CountVectorizer(stop_words='english', max_features=maxFeatures)
bag1 = cv1.fit_transform(df_txt1)
bag1 = np.array(bag1.todense())

# Split the data into a training set and a test set
X_train1, X_test1, y_train1, y_test1 = train_test_split(bag1, df_fld1, test_size=test_set1, random_state=0)

# Multinomial Naive Bayes model
model = MultinomialNB()
model_text = 'MultinomialNB'

# Train and test
y_mod1 = model.fit(X_train1, y_train1)
y_prd1 = y_mod1.predict(X_test1)

# Performance Evaluation
F1_macro1 = f1_score(y_test1, y_prd1, average='macro')
print('Evaluation using Merged dataset to predict on Merged dataset')
print(
    f'F1 score: {F1_macro1:.3f}\n   {model_text}\n   macro averaging\n   max_features = {maxFeatures}\n   '
    f'test set = {test_set1 * 100}%\n')
if prt_reports:
    print(f'Classification Report: {model_text}')
    print(classification_report(y_test1, y_prd1, labels=top_labels1))
    print(confusion_matrix(y_test1, y_prd1))

# Use Merged-based model to predict for Wikipedia data

# Restrict Wikipedia data
top_df_temp = df_wiki[df_wiki[field].isin(top_labels1)]
top_df2 = copy.copy(top_df_temp)
df_txt2 = top_df2['text']
df_fld2 = top_df2[field]

# Create bag of words from Wikipedia data
cv2 = CountVectorizer(stop_words='english', max_features=maxFeatures)
bag2 = cv2.fit_transform(df_txt2)
bag2 = np.array(bag2.todense())

# Split Wikipedia data into training and test sets
X_train2, X_test2, y_train2, y_test2 = train_test_split(bag2, df_fld2, test_size=test_set2, random_state=0)

# Use the Merged model to classify the Wikipedia data
y_prd2 = y_mod1.predict(X_test2)

# Performance Evaluation
F1_macro2 = f1_score(y_test2, y_prd2, average='macro')
print('\nEvaluation using Merged dataset to predict on Wikipedia dataset')
print(
    f'F1 score: {F1_macro2:.3f}\n   {model_text}\n   macro averaging\n   max_features = {maxFeatures}\n   '
    f'test set = {test_set2 * 100}%\n')
if prt_reports:
    print(f'Classification Report: {model_text}')
    print(classification_report(y_test2, y_prd2, labels=top_labels1))
    print(confusion_matrix(y_test2, y_prd2))
