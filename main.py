''' https://www.geeksforgeeks.org/image-classification-using-support-vector-machine-svm-in-python/ '''
''' https://www.kaggle.com/code/gauravduttakiit/visualize-a-decision-tree '''
''' https://stackoverflow.com/questions/69061767/how-to-plot-feature-importance-for-decisiontreeclassifier '''

import pandas as pd
import os


def df_ETL():
    path = 'E:\\IU_courses\\HOMEWORK\\Data for Task 1'
    folder = os.listdir(path)
    for file in folder:
        df = pd.read_csv(f'{path}\\{file}')
        # check info #
        # print(df.head())  # 'Unnamed: 32' can be dropped
        # print(df.info())  # no-missing value to impute  # convert 'diagnosis' column to 0 & 1?
        # print(df.columns)  # put 'diagnosis' at last column for model usage
        # print(df.describe())

        # reorder columns by creating column name list
        ind_var_list = df.columns.drop(['diagnosis', 'Unnamed: 32']).to_list()
        # print(ind_var_list)
        df = df[ind_var_list + ['diagnosis']]
        # print(df.columns)

        return df

df = df_ETL()


# plt.figure(figsize = (10,5))
# sns.heatmap(df.corr(), annot = True, cmap="rainbow")
# plt.show()



from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb


X = df.iloc[:, :-1]
y = df.iloc[:, -1]  # 'diagnosis'

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24, train_size=0.70)

# Method 1: Decision Tree
decision_tree = DecisionTreeClassifier(max_depth=8)
decision_tree.fit(X_train, y_train)
y_pred_dtree = decision_tree.predict(X_test)
accuracy_dtree = accuracy_score(y_test, y_pred_dtree)
print(f'Decision Tree Accuracy: {accuracy_dtree:.4f}')
print(confusion_matrix(y_test, y_pred_dtree))
print (classification_report(y_test, y_pred_dtree))

# # Method 2: K-nearest-neighbor
# knn = KNeighborsClassifier(n_neighbors=5)
# knn.fit(X_train, y_train)
# y_pred_knn = knn.predict(X_test)
# accuracy_knn = accuracy_score(y_test, y_pred_knn)
# print(f'K-Nearest Accuracy: {accuracy_knn:.4f}')
#
# # Method 3: Naive Bayes
# gnb = GaussianNB()
# gnb.fit(X_train, y_train)
# y_pred_gnb = gnb.predict(X_test)
# accuracy_gnb = accuracy_score(y_test, y_pred_gnb)
# print(f'Naive Bayes Accuracy: {accuracy_gnb:.4f}')
#
# # Method 4: Logistic Regression
# reg = LogisticRegression()
# reg.fit(X_train, y_train)
# y_pred_reg = reg.predict(X_test)
# accuracy_reg = accuracy_score(y_test, y_pred_reg)
# print(f'Logistic Regression Accuracy: {accuracy_reg:.4f}')
#
# # # Method 5: SVM (time consuming, give up)
# # svm = SVC(kernel='linear')
# # svm.fit(X_train, y_train)
# # y_pred_svm = svm.predict(X_test)
# # accuracy_svm = accuracy_score(y_test, y_pred_svm)
# # print(f'SVM Accuracy: {accuracy_svm:.4f}')


# import seaborn as sns
# from matplotlib import pyplot as plt
#
# df_xgb = df.copy()
# df_xgb['diagnosis'].replace(['M', 'B'], [0, 1], inplace=True)
#
# plt.figure(figsize=(20, 14))
# sns.set(font_scale=0.7)
# sns.heatmap(df_xgb.corr(), annot=True, cmap="rainbow")
#
# plt.title("Correlation Heatmap")
# plt.savefig('corr_heatmap.png')
# # plt.show()


# decision_tree = DecisionTreeClassifier(max_depth=8)
# decision_tree.fit(X_train, y_train)
# y_pred_dtree = decision_tree.predict(X_test)
# accuracy_dtree = accuracy_score(y_test, y_pred_dtree)


# from sklearn import tree
#
#
# fig = plt.figure(figsize=(25, 20))
# _ = tree.plot_tree(decision_tree,
#                    feature_names=X.columns,
#                    class_names=['M', 'B'],
#                    filled=True)
#
# # fig.show()
# plt.savefig('decision_tree.png')


# # print(type(decision_tree.feature_importances_), decision_tree.feature_importances_)
# feat_importances_df = pd.DataFrame(decision_tree.feature_importances_, index=X_train.columns, columns=["Importance"])
# feat_importances_df.index.name = 'Feature'
# feat_importances_df = feat_importances_df[feat_importances_df.Importance != 0]
# feat_importances_df.sort_values(by='Importance', ascending=True, inplace=True)
#
# values = feat_importances_df.Importance
# idx = feat_importances_df.index
# clrs = ['green' if (x < max(values)) else 'red' for x in values]
#
# sns.set(rc={'figure.figsize':(15, 8)})
# sns.barplot(y=idx, x=values, palette=clrs)
#
# plt.title("Importance Features to Predict Breast Cancer")
# plt.xlabel('Importance')
# plt.ylabel('Feature')
#
# plt.savefig('feature_importance.png')
# # plt.show()


# # Method 6: XGB Boost
# df_xgb = df.copy()
# df_xgb['diagnosis'].replace(['M', 'B'], [0, 1], inplace=True)
#
# X = df_xgb.iloc[:, :-1]
# y = df_xgb.iloc[:, -1]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24, train_size=0.70)
#
# xgb_boost = xgb.XGBRegressor(objective='binary:logistic')
# xgb_boost.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)
#
# accuracy_xgb = xgb_boost.score(X_test, y_test)
# print(f'XGB Boost Accuracy: {accuracy_xgb:.4f}')