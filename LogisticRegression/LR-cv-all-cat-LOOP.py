# Import required functions from scikit-learn
import copy
import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data_df = {
    "file": [],
    "solver": [],
    "penalty": [],
    "num_categories": [],
    "max_features": [],
    "test_set": [],
    "test_f1_macro": [],
    "accuracy": []
}

files = ['silvics_edited_data.csv', 'wikipedia_edited_data.csv', 'merged_edited_data.csv']
field = 'category'
pen = 'l2'
mc = 'auto'
tl = 0.0001
iter_max = 1000

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

        for solv in ['liblinear', 'lbfgs']:
            model = LogisticRegression(solver=solv, penalty=pen, tol=tl, max_iter=iter_max, multi_class=mc)

            for maxFeatures in [1000, 2000, 5000]:
                for test_set in [0.50, 0.25, 0.10]:

                    model_text = 'GaussianNB'
                    cv = CountVectorizer(stop_words='english', max_features=maxFeatures)
                    bag = cv.fit_transform(df_text)
                    bag = np.array(bag.todense())
                    X_train, X_test, y_train, y_test = train_test_split(bag, df_field, test_size=test_set, random_state=0)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    f1_macro = metrics.f1_score(y_test, y_pred, average='macro')
                    accuracy = metrics.accuracy_score(y_test, y_pred)

                    data_df["file"].append(file)
                    data_df["solver"].append(solv)
                    data_df["penalty"].append(pen)
                    data_df["num_categories"].append(n)
                    data_df["max_features"].append(maxFeatures)
                    data_df["test_set"].append(test_set)
                    data_df["test_f1_macro"].append(f1_macro)
                    data_df["accuracy"].append(accuracy)

                    print(f'ITERATION {iter} COMPLETE.\n'
                          f'   {file}, {solv}, {pen}, {n}, {maxFeatures}, {test_set}, {f1_macro:.3f}, {accuracy:.3f}')
                    iter += 1

loop_data = pd.DataFrame(data_df)
loop_data.to_csv('LR-cv-all-cat-LOOP-data2.csv')