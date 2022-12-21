import copy
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from keras.models import Sequential
from keras import layers
from keras.utils import np_utils

from keras.backend import clear_session

clear_session()

df_merg = pd.read_csv('merged_edited_data_sents.csv')
field = 'category_int'
df_merg[field] = pd.to_numeric(df_merg[field])
encoder = LabelEncoder()

data_df = {
    "num_categories": [],
    "max_features": [],
    "test_set": [],
    "layer1_neurons": [],
    "layer2_neurons": [],
    "epochs": [],
    "batch_size": [],
    "parameters": [],
    "train_loss": [],
    "train_accuracy": [],
    "train_auc": [],
    "train_precision": [],
    "train_recall": [],
    "train_f1": [],
    "test_loss": [],
    "test_accuracy": [],
    "test_auc": [],
    "test_precision": [],
    "test_recall": [],
    "test_f1": []
}

iter = 1
for n in [10]:
    layer2 = n

    labels = df_merg[field].value_counts()
    top_labels = labels.index.tolist()[:n]
    top_df_temp = df_merg[df_merg[field].isin(top_labels)]
    top_df = copy.copy(top_df_temp)
    X = top_df['sent'].values
    y = top_df[field].values

    encoder.fit(y)
    encoded_Y = encoder.transform(y)
    dummy_y = np_utils.to_categorical(encoded_Y)

    for maxFeatures in [5000]:
        cv = CountVectorizer(stop_words='english', max_features=maxFeatures)
        bag = cv.fit_transform(X)
        bag = np.array(bag.todense())

        for test_set in [0.10]:
            X_train, X_test, y_train, y_test = train_test_split(bag, dummy_y, test_size=test_set, random_state=0)
            input_dim = X_train.shape[1]

            for layer1 in [10]:
                clear_session()
                model = Sequential()
                model.add(layers.Dense(layer1, input_dim=input_dim, activation='relu'))
                model.add(layers.Dense(layer2, activation='softmax'))
                model.compile(loss='categorical_crossentropy',
                              optimizer='adam',
                              metrics=['accuracy',
                                       'AUC',
                                       'Precision',
                                       'Recall'])
                for epochs in [5]:
                    for batch in [1, 5, 10]:
                        history = model.fit(X_train, y_train,
                                            epochs=epochs,
                                            verbose=0,
                                            validation_data=(X_test, y_test),
                                            batch_size=batch)
                        loss, accuracy, auc, precision, recall = model.evaluate(X_train, y_train, verbose=0)

                        data_df["num_categories"].append(n)
                        data_df["max_features"].append(maxFeatures)
                        data_df["test_set"].append(test_set)
                        data_df["layer1_neurons"].append(layer1)
                        data_df["layer2_neurons"].append(layer2)
                        data_df["epochs"].append(epochs)
                        data_df["batch_size"].append(batch)
                        data_df["parameters"].append((maxFeatures + 1) * layer1 + (layer1 + 1) * layer2)
                        data_df["train_loss"].append(loss)
                        data_df["train_accuracy"].append(accuracy)
                        data_df["train_auc"].append(auc)
                        data_df["train_precision"].append(precision)
                        data_df["train_recall"].append(recall)
                        data_df["train_f1"].append(2 * precision * recall / (precision + recall))

                        loss, accuracy, auc, precision, recall = model.evaluate(X_test, y_test, verbose=0)

                        data_df["test_loss"].append(loss)
                        data_df["test_accuracy"].append(accuracy)
                        data_df["test_auc"].append(auc)
                        data_df["test_precision"].append(precision)
                        data_df["test_recall"].append(recall)
                        data_df["test_f1"].append(2 * precision * recall / (precision + recall))

                        print(f'Iteration {iter} complete.\n'
                              f'   {n}, {maxFeatures}, {test_set}, {layer1}, {layer2}, {epochs}, {batch}, '
                              f'{2 * precision * recall / (precision + recall):.3f}, {accuracy:.3f}')
                        iter += 1

loop_data = pd.DataFrame(data_df)
loop_data.to_csv('Keras-Me-cat-LOOP-sent-data.csv')
