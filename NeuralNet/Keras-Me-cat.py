import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

from keras.models import Sequential
from keras import layers
from scikeras.wrappers import KerasClassifier
from keras.utils import np_utils

from keras.backend import clear_session

clear_session()
plt.style.use('ggplot')

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

# Import data for classification
df_merg = pd.read_csv('merged_edited_data.csv')

# Classification parameters (used for all trials)
field = 'category_int'
prt_class_rept = False
n = 10  # only consider the n most frequent classes in field
maxFeatures = 2000
test_set = 0.25
layer1 = 10
layer2 = n
epochs = 50
batch = 1

print('\n### Results for Merged Dataset ###\n')

df_merg[field] = pd.to_numeric(df_merg[field])

# Merged parameters
labels = df_merg[field].value_counts()
top_labels = labels.index.tolist()[:n]
top_df_temp = df_merg[df_merg[field].isin(top_labels)]
top_df = copy.copy(top_df_temp)
# df_text = top_df['text']
# df_field = top_df[field]
X = top_df['text'].values
y = top_df[field].values

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)
# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)

# Create a bag of words
cv = CountVectorizer(stop_words='english', max_features=maxFeatures)
# bag = cv.fit_transform(df_text)
bag = cv.fit_transform(X)
bag = np.array(bag.todense())

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(bag, dummy_y, test_size=test_set, random_state=0)

input_dim = X_train.shape[1]
model = Sequential()
model.add(layers.Dense(layer1, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(layer2, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'AUC', 'Precision', 'Recall'])
model.summary()

history = model.fit(X_train, y_train,
                    epochs=epochs,
                    verbose=0,
                    validation_data=(X_test, y_test),
                    batch_size=batch)

loss, accuracy, auc, precision, recall = model.evaluate(X_train, y_train, verbose=0)
print("Training Accuracy: {:.4f}".format(accuracy))
print("Training AUC: {:.4f}".format(auc))
print("Training Precision: {:.4f}".format(precision))
print("Training Recall: {:.4f}".format(recall))

loss, accuracy, auc, precision, recall = model.evaluate(X_test, y_test, verbose=0)
print("Testing Accuracy:  {:.4f}".format(accuracy))
print("Testing AUC: {:.4f}".format(auc))
print("testing Precision: {:.4f}".format(precision))
print("Testing Recall: {:.4f}".format(recall))

plot_history(history)

# estimator = KerasClassifier(model=model, epochs=epochs, batch_size=batch, verbose=0)
# kfold = KFold(n_splits=10, shuffle=True)
# results = cross_val_score(estimator, X_train, y_train, cv=kfold)
# print("Baseline: %.2f%% (%.2f%%" % (results.mean()*100, results.std()*100))


# Import data for classification
df_silv = pd.read_csv('silvics_edited_data.csv')
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

print(f'X_test shape: {X_test.shape}, y_test shape: {y_test.shape}')
print(f'bag1 shape: {bag1.shape}, dummy_y1 shape: {dummy_y1.shape}')

# loss, accuracy, auc, precision, recall = model.evaluate(bag1, dummy_y1, verbose=0)
# print("Silvics Accuracy:  {:.4f}".format(accuracy))
# print("Silvics AUC: {:.4f}".format(auc))
# print("Silvics Precision: {:.4f}".format(precision))
# print("Silvics Recall: {:.4f}".format(recall))


# Import data for classification
df_wiki = pd.read_csv('wikipedia_edited_data.csv')
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
print("Wikipedia Accuracy:  {:.4f}".format(accuracy))
print("Wikipedia AUC: {:.4f}".format(auc))
print("Wikipedia Precision: {:.4f}".format(precision))
print("Wikipedia Recall: {:.4f}".format(recall))