import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.metrics import classification_report
# from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import *

from keras.models import Sequential
from keras import layers
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
n = 10  # only consider the n most frequent classes in field
maxFeatures = 2000
test_set = 0.25
prt_class_rept = False
epochs = 25

print('\n### Results for Merged Dataset ###\n')

df_merg[field] = pd.to_numeric(df_merg[field])

# Merged parameters
labels = df_merg[field].value_counts()
top_labels = labels.index.tolist()[:n]
top_df_temp = df_merg[df_merg[field].isin(top_labels)]
top_df = copy.copy(top_df_temp)
df_text = top_df['text']
df_field = top_df[field]

# Create a bag of words
cv = CountVectorizer(stop_words='english', max_features=maxFeatures)
bag = cv.fit_transform(df_text)
bag = np.array(bag.todense())

# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(bag, df_field, test_size=test_set, random_state=0)

input_dim = X_train.shape[1]

model = Sequential()
model.add(layers.Dense(10, input_dim=input_dim, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train,
                    epochs=epochs,
                    verbose=False,
                    validation_data=(X_test, y_test),
                    batch_size=10)

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))

plot_history(history)