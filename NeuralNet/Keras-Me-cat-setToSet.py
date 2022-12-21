import copy

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras import layers
from keras.backend import clear_session
from keras.models import Sequential
from keras.utils import np_utils
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

clear_session()
plt.style.use('ggplot')

# Import data for classification
df_merg = pd.read_csv('merged_edited_data.csv')
df_silv = pd.read_csv('silvics_edited_data.csv')
df_wiki = pd.read_csv('wikipedia_edited_data.csv')

field = 'category_int'
n = 12
maxFeatures = 2000
test_set = 0.25
layer1 = 20
layer2 = n
epochs = 5
batch = 5

labels = df_silv[field].value_counts()
top_labels = labels.index.tolist()[:n]

df_merg[field] = pd.to_numeric(df_merg[field])

top_df_temp = df_merg[df_merg[field].isin(top_labels)]
top_df = copy.copy(top_df_temp)
X = top_df['text'].values
y = top_df[field].values

encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
dummy_y = np_utils.to_categorical(encoded_Y)

cv = CountVectorizer(stop_words='english', max_features=maxFeatures)
bag = cv.fit_transform(X)
bag = np.array(bag.todense())

X_train, X_test, y_train, y_test = train_test_split(bag, dummy_y, test_size=test_set, random_state=0)

input_dim = X_train.shape[1]
model = Sequential()
model.add(layers.Dense(layer1, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(layer2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'AUC', 'Precision', 'Recall'])
model.summary()

model.fit(X_train, y_train,
          epochs=epochs,
          verbose=0,
          validation_data=(X_test, y_test),
          batch_size=batch)

loss, accuracy, auc, precision, recall = model.evaluate(X_train, y_train, verbose=0)
print("Training Accuracy: {:.4f}".format(accuracy))
print("Training AUC: {:.4f}".format(auc))
print("Training Precision: {:.4f}".format(precision))
print("Training Recall: {:.4f}".format(recall))
print("Training F1 (macro): {:.4f}".format(2 * precision * recall / (precision + recall)))
print()
loss, accuracy, auc, precision, recall = model.evaluate(X_test, y_test, verbose=0)
print("Testing Accuracy:  {:.4f}".format(accuracy))
print("Testing AUC: {:.4f}".format(auc))
print("testing Precision: {:.4f}".format(precision))
print("Testing Recall: {:.4f}".format(recall))
print("Testing F1: {:.4f}".format(2 * precision * recall / (precision + recall)))

df_silv[field] = pd.to_numeric(df_silv[field])
top_df_temp = df_silv[df_silv[field].isin(top_labels)]
top_df = copy.copy(top_df_temp)
X1 = top_df['text'].values
y1 = top_df[field].values
encoder.fit(y1)
encoded_Y1 = encoder.transform(y1)
dummy_y1 = np_utils.to_categorical(encoded_Y1)
bag1 = cv.fit_transform(X1)
bag1 = np.array(bag1.todense())

loss, accuracy, auc, precision, recall = model.evaluate(bag1, dummy_y1, verbose=0)
print()
print("Silvics Accuracy:  {:.4f}".format(accuracy))
print("Silvics AUC: {:.4f}".format(auc))
print("Silvics Precision: {:.4f}".format(precision))
print("Silvics Recall: {:.4f}".format(recall))
print("Silvics F1: {:.4f}".format(2 * precision * recall / (precision + recall)))

df_wiki[field] = pd.to_numeric(df_wiki[field])
top_df_temp = df_wiki[df_wiki[field].isin(top_labels)]
top_df = copy.copy(top_df_temp)
X2 = top_df['text'].values
y2 = top_df[field].values
encoder.fit(y2)
encoded_Y2 = encoder.transform(y2)
dummy_y2 = np_utils.to_categorical(encoded_Y2)
bag2 = cv.fit_transform(X2)
bag2 = np.array(bag2.todense())

loss, accuracy, auc, precision, recall = model.evaluate(bag2, dummy_y2, verbose=0)
print()
print("Wikipedia Accuracy:  {:.4f}".format(accuracy))
print("Wikipedia AUC: {:.4f}".format(auc))
print("Wikipedia Precision: {:.4f}".format(precision))
print("Wikipedia Recall: {:.4f}".format(recall))
print("Wikipedia F1: {:.4f}".format(2 * precision * recall / (precision + recall)))
