# Import required functions from scikit-learn
import copy
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import *

data_df = {
    "file": [],
    "model": [],
    "num_categories": [],
    "max_features": [],
    "test_set": [],
    # "test_loss": [],
    # "test_accuracy": [],
    # "test_precision": [],
    # "test_recall": [],
    "test_f1_macro": [],
    "accuracy": []
}

files = ['silvics_edited_data.csv', 'wikipedia_edited_data.csv', 'merged_edited_data.csv']
field = 'category'
gnb = GaussianNB()
cnb = ComplementNB()
mnb = MultinomialNB()

iter = 1
for file in files:
    df = pd.read_csv(file)

    for n in [10, 19]:
        labels = df[field].value_counts()
        top_labels = labels.index.tolist()[:n]
        top_df_temp = df[df[field].isin(top_labels)]
        top_df = copy.copy(top_df_temp)
        df_text = top_df['text']
        df_field = top_df[field]

        for maxFeatures in [1000, 2000, 5000]:
            for test_set in [0.50, 0.25, 0.10]:

                model_text = 'GaussianNB'
                cv = CountVectorizer(stop_words='english', max_features=maxFeatures)
                bag = cv.fit_transform(df_text)
                bag = np.array(bag.todense())
                X_train, X_test, y_train, y_test = train_test_split(bag, df_field, test_size=test_set, random_state=0)
                gnb.fit(X_train, y_train)
                y_pred = gnb.predict(X_test)
                # prob_mat = gnb.predict_proba(X_test)

                # log_loss = metrics.log_loss(y_test, prob_mat)
                # accuracy = metrics.accuracy_score(y_test, y_pred)
                # precision = metrics.precision_score(y_test, y_pred, average='macro')
                # recall = metrics.recall_score(y_test, y_pred, average='macro')
                f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
                accuracy = metrics.accuracy_score(y_test, y_pred)

                data_df["file"].append(file)
                data_df["model"].append(model_text)
                data_df["num_categories"].append(n)
                data_df["max_features"].append(maxFeatures)
                data_df["test_set"].append(test_set)
                # data_df["test_loss"].append(log_loss)
                # data_df["test_accuracy"].append(accuracy)
                # data_df["test_precision"].append(precision)
                # data_df["test_recall"].append(recall)
                data_df["test_f1_macro"].append(f1_macro)
                data_df["accuracy"].append(accuracy)

                print(f'{file}, {model_text}, {n}, {maxFeatures}, {test_set}, {f1_macro}, {accuracy}')

                model_text = 'ComplementNB'
                cnb.fit(X_train, y_train)
                y_pred = cnb.predict(X_test)
                # prob_mat = cnb.predict_proba(X_test)

                # log_loss = metrics.log_loss(y_test, prob_mat)
                # accuracy = metrics.accuracy_score(y_test, y_pred)
                # precision = metrics.precision_score(y_test, y_pred, average='macro')
                # recall = metrics.recall_score(y_test, y_pred, average='macro')
                f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
                accuracy = metrics.accuracy_score(y_test, y_pred)

                data_df["file"].append(file)
                data_df["model"].append(model_text)
                data_df["num_categories"].append(n)
                data_df["max_features"].append(maxFeatures)
                data_df["test_set"].append(test_set)
                # data_df["test_loss"].append(log_loss)
                # data_df["test_accuracy"].append(accuracy)
                # data_df["test_precision"].append(precision)
                # data_df["test_recall"].append(recall)
                data_df["test_f1_macro"].append(f1_macro)
                data_df["accuracy"].append(accuracy)

                print(f'{file}, {model_text}, {n}, {maxFeatures}, {test_set}, {f1_macro}, {accuracy}')

                model_text = 'MultinomialNB'
                mnb.fit(X_train, y_train)
                y_pred = mnb.predict(X_test)
                # prob_mat = mnb.predict_proba(X_test)

                # log_loss = metrics.log_loss(y_test, prob_mat)
                # accuracy = metrics.accuracy_score(y_test, y_pred)
                # precision = metrics.precision_score(y_test, y_pred, average='macro')
                # recall = metrics.recall_score(y_test, y_pred, average='macro')
                f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
                accuracy = metrics.accuracy_score(y_test, y_pred)

                data_df["file"].append(file)
                data_df["model"].append(model_text)
                data_df["num_categories"].append(n)
                data_df["max_features"].append(maxFeatures)
                data_df["test_set"].append(test_set)
                # data_df["test_loss"].append(log_loss)
                # data_df["test_accuracy"].append(accuracy)
                # data_df["test_precision"].append(precision)
                # data_df["test_recall"].append(recall)
                data_df["test_f1_macro"].append(f1_macro)
                data_df["accuracy"].append(accuracy)

                print(f'{file}, {model_text}, {n}, {maxFeatures}, {test_set}, {f1_macro}, {accuracy}')

                print(f'ITERATION {iter} COMPLETE.\n')
                iter += 1

loop_data = pd.DataFrame(data_df)
loop_data.to_csv('NB-3mod-cv-all-cat-LOOP-data2.csv')