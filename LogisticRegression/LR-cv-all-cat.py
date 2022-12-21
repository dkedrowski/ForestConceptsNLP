# Import required functions from scikit-learn
import pandas as pd
import numpy as np
import copy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

# Import data for classification
df_silv = pd.read_csv('silvics_edited_data.csv')
df_wiki = pd.read_csv('wikipedia_edited_data.csv')
df_merg = pd.read_csv('merged_edited_data.csv')

# Classification parameters (used for all trials)
field = 'category'
n = 19                # only consider the n most frequent classes in field
maxFeatures = 2000
test_set = 0.1
solv = 'liblinear'    # choices include: lbfgs [quasi-Newton, recommended for small datasets],
                      #                  liblinear [coordinate descent, not truly multinomial (one-vs-rest), penalizes the intercept],
                      #                  sag [stochastic gradient descent, faster for large datasets],
                      #                  newton-cg,
                      #                  saga [sag variant, good choice for sparse multinomial regression, faster for large datasets]
pen = 'l2'            # choices include: l1 [liblinear, saga],
                      #                  l2 [any],
                      #                  none [any but liblinear],
                      #                  elasticnet [saga]
tl = 0.0001
iter_max = 1000
mc = 'auto'           # choices include: auto, ovr, multinomial [not available for liblinear]
prt_class_rept = False

### Silvics Manual dataset ###
### Logistic Regression per above choices  ###

print('\n### Results for Silvics Manual Dataset ###\n')

# Silvics parameters
labels = df_silv[field].value_counts()
top_labels = labels.index.tolist()[:n]
top_df_temp = df_silv[df_silv[field].isin(top_labels)]
top_df = copy.copy(top_df_temp)
df_text = top_df['text']
df_field = top_df[field]

# Logistic Regression model
model = LogisticRegression(solver=solv, penalty=pen, tol=tl, max_iter=iter_max, multi_class=mc)
model_text = 'Logistic Regression'

# Create a bag of words
cv = CountVectorizer(stop_words='english', max_features=maxFeatures)
bag = cv.fit_transform(df_text)
bag = np.array(bag.todense())

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(bag, df_field, test_size=test_set, random_state=0)

# Train and test
y_pred = model.fit(X_train, y_train).predict(X_test)

# Performance Evaluation
F1_macro = f1_score(y_test, y_pred, average='macro')
print(f'F1 score: {F1_macro:.3f}\n   {model_text}: {solv}, {pen}, {mc}\n   macro averaging\n   n = {n}\n   max_features = {maxFeatures}\n   test set = {test_set * 100}%\n')
if prt_class_rept:
    print(f'Classification Report: {model_text}: {solv}, {pen}, {mc}')
    print((classification_report(y_test, y_pred, labels=top_labels)))

### Wikipedia dataset ###
### Logistic Regression per above choices  ###

print('\n### Results for Wikipedia Dataset ###\n')

# Wikipedia parameters
labels = df_wiki[field].value_counts()
top_labels = labels.index.tolist()[:n]
top_df_temp = df_wiki[df_wiki[field].isin(top_labels)]
top_df = copy.copy(top_df_temp)
df_text = top_df['text']
df_field = top_df[field]

# Logistic Regression model
model = LogisticRegression(solver=solv, penalty=pen, tol=tl, max_iter=iter_max, multi_class=mc)
model_text = 'Logistic Regression'

# Create a bag of words
cv = CountVectorizer(stop_words='english', max_features=maxFeatures)
bag = cv.fit_transform(df_text)
bag = np.array(bag.todense())

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(bag, df_field, test_size=test_set, random_state=0)

# Train and test
y_pred = model.fit(X_train, y_train).predict(X_test)

# Performance Evaluation
F1_macro = f1_score(y_test, y_pred, average='macro')
print(f'F1 score: {F1_macro:.3f}\n   {model_text}: {solv}, {pen}, {mc}\n   macro averaging\n   n = {n}\n   max_features = {maxFeatures}\n   test set = {test_set * 100}%\n')
if prt_class_rept:
    print(f'Classification Report: {model_text}: {solv}, {pen}, {mc}')
    print((classification_report(y_test, y_pred, labels=top_labels)))

### Merged dataset ###
### Logistic Regression per above choices  ###

print('\n### Results for Merged Dataset ###\n')

# Merged parameters
labels = df_merg[field].value_counts()
top_labels = labels.index.tolist()[:n]
top_df_temp = df_merg[df_merg[field].isin(top_labels)]
top_df = copy.copy(top_df_temp)
df_text = top_df['text']
df_field = top_df[field]

# Logistic Regression model
model = LogisticRegression(solver=solv, penalty=pen, tol=tl, max_iter=iter_max, multi_class=mc)
model_text = 'Logistic Regression'

# Create a bag of words
cv = CountVectorizer(stop_words='english', max_features=maxFeatures)
bag = cv.fit_transform(df_text)
bag = np.array(bag.todense())

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(bag, df_field, test_size=test_set, random_state=0)

# Train and test
y_pred = model.fit(X_train, y_train).predict(X_test)

# Performance Evaluation
F1_macro = f1_score(y_test, y_pred, average='macro')
print(f'F1 score: {F1_macro:.3f}\n   {model_text}: {solv}, {pen}, {mc}\n   macro averaging\n   n = {n}\n   max_features = {maxFeatures}\n   test set = {test_set * 100}%\n')
if prt_class_rept:
    print(f'Classification Report: {model_text}: {solv}, {pen}, {mc}')
    print((classification_report(y_test, y_pred, labels=top_labels)))