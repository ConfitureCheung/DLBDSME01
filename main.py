import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import tree
import seaborn as sns
import plotly
import plotly.express as px
import plotly.graph_objects as go
from sklearn.inspection import PartialDependenceDisplay, permutation_importance
from mlxtend.evaluate import feature_importance_permutation
from lime import lime_tabular
import shap


''' 
Reference for coding and concept:

Interpretable vs Explainable Machine Learning
https://www.youtube.com/watch?v=VY7SCl_DFho
agnostic methods
https://www.youtube.com/watch?v=YuDijSIR9iM
PDP
https://www.youtube.com/watch?v=uQQa3wQgG_s
https://www.youtube.com/watch?v=21QAKe2PDkk
Permutation Feature Importance
https://www.youtube.com/watch?v=VUvShOEFdQo
https://www.youtube.com/watch?v=meTXOuFV-s8
Global Surrogate
https://www.youtube.com/watch?v=uOL-Zb9_DO4
LIME
https://www.youtube.com/watch?v=CYl172IwqKs
https://www.youtube.com/watch?v=ZMB6TwQ6Vuo
Shapley value
https://www.youtube.com/watch?v=UJeu29wq7d0
SHAP
https://www.youtube.com/watch?v=MQ6fFDwjuco
https://www.youtube.com/watch?v=L8_sVRhBDLU
'''


## data consolidation ##
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


# data transformation and train test split
df['diagnosis'].replace(['M', 'B'], [0, 1], inplace=True)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]  # 'diagnosis'

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=24, train_size=0.70)


## build interpretable prediction model ##
# Method 1: Decision Tree
decision_tree = DecisionTreeClassifier(max_depth=8)
decision_tree.fit(X_train, y_train)
y_pred_dtree = decision_tree.predict(X_test)
f1_dtree = f1_score(y_test, y_pred_dtree)
print(f'Decision Tree f1 score: {f1_dtree:.4f}')

# Method 2: K-nearest-neighbor
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
f1_knn = f1_score(y_test, y_pred_knn)
print(f'K-Nearest f1 score: {f1_knn:.4f}')

# Method 3: Naive Bayes
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred_gnb = gnb.predict(X_test)
f1_gnb = f1_score(y_test, y_pred_gnb)
print(f'Naive Bayes f1 score: {f1_gnb:.4f}')

# Method 4: Logistic Regression
reg = LogisticRegression()
reg.fit(X_train, y_train)
y_pred_reg = reg.predict(X_test)
f1_reg = f1_score(y_test, y_pred_reg)
print(f'Logistic Regression f1 score: {f1_reg:.4f}')

## explainable model ##
# Method 5: Random Forest
random_clf = RandomForestClassifier()
random_clf.fit(X_train, y_train)
y_pred_rclf = random_clf.predict(X_test)
f1_rclf = f1_score(y_test, y_pred_rclf)
print(f'Random Forest f1 score: {f1_rclf:.4f}')


## plot graphs ##
# correlation heatmap
plt.figure(figsize=(20, 14))
sns.set(font_scale=0.7)
sns.heatmap(df.corr(), annot=True, cmap="rainbow")

plt.title("Correlation Heatmap")
# plt.savefig('corr_heatmap.png')
plt.show()


# decision tree graph
fig1 = plt.figure(figsize=(25, 20))
_ = tree.plot_tree(decision_tree, feature_names=X.columns, class_names=['M', 'B'], filled=True)
# plt.savefig('decision_tree.png')
fig1.show()


# logistic regression graph
fig2 = plt.figure(figsize=(25, 20))
sns.regplot(x=X['concave points_worst'], y=y, data=df, logistic=True, ci=None, line_kws=dict(color="r"))
# plt.savefig('logistic regression.png')
fig2.show()


# k-means clustering
class KMeansClustering():
    def __init__(self, k):
        self.k = k
        self.centroids = None

    @staticmethod
    def euclidean_distance(data_point, centroids):
        return np.sqrt(np.sum((centroids - data_point)**2, axis=1))

    def fit(self, X, max_iterations=200):
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0), size=(self.k, X.shape[1]))

        for _ in range(max_iterations):
            y = []
            for data_point in X:
                distances = KMeansClustering.euclidean_distance(data_point, self.centroids)
                cluster_num = np.argmin(distances)
                y.append(cluster_num)

            y = np.array(y)

            cluster_indices = []

            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i))

            cluster_centers = []
            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])

            if np.max(self.centroids - np.array(cluster_centers)) < 0.0001:
                break
            else:
                self.centroids = np.array(cluster_centers)

        return y


def plotly_kmeans(df, cluster_num, palette):
    df = df[["concave points_worst", "diagnosis"]].to_numpy()

    kmeans = KMeansClustering(k=cluster_num)
    labels = kmeans.fit(df)

    fig = px.scatter(df, x=df[:, 0], y=df[:, 1], color=labels, title="K-Means Clustering", color_continuous_scale=palette)

    fig.update(layout_coloraxis_showscale=False)
    fig.for_each_xaxis(lambda x: x.update(title="Concave Points_worst"))
    fig.for_each_yaxis(lambda y: y.update(title="Diagnosis"))
    # fig.update_layout(yaxis_range=[0, 15])

    # add centroid
    fig.add_trace(go.Scatter(x=kmeans.centroids[:, 0], y=kmeans.centroids[:, 1], mode='markers', marker_size=15, marker_symbol='star'))
    fig.update_layout(showlegend=False)

    fig.show()
    # path = "E:\\IU_courses\\HOMEWORK"
    # plotly.offline.plot(fig, filename=f"{path}\\K-Means Clustering.html")


# https://plotly.com/python/builtin-colorscales/
# plotly_kmeans(df=df, cluster_num=2, palette='turbid')


# 5.1.1 partial dependence plot
common_params = {
    # "subsample": 50,
    "n_jobs": 2,
    "grid_resolution": 20,
    "random_state": 0,
}
features_info = {
    # features of interest
    "features": X.columns[1:].tolist(),
    # type of partial dependence plot
    "kind": "average",
}

_, ax = plt.subplots(ncols=6, nrows=5, figsize=(14, 12), constrained_layout=True)
display = PartialDependenceDisplay.from_estimator(
    random_clf, X, **features_info, ax=ax, **common_params)

_ = display.figure_.suptitle(
    "Partial dependence Plot of 30 features", fontsize=16,
)

# plt.savefig('partial_dependence_plot.png')
plt.show()


# 5.1.2 permutation feature importance
imp_vals, imp_all = feature_importance_permutation(
    predict_method=random_clf.predict,
    X=X_test.values,
    y=y_test.values,
    metric='accuracy',
    num_rounds=50,
    seed=0
)

std = np.std(imp_all, axis=1)
ind = np.argsort(imp_vals)[::-1]

plt4 = plt.figure(figsize=(25, 15))
plt.title("Importance Features to Predict Breast Cancer")
plt.bar(range(X_train.shape[1]), imp_vals[ind], yerr=std[ind])

plt.xticks(range(X_train.shape[1]), df.columns[1:][ind], rotation=90)
plt.xlim([-1, X_train.shape[1]])
# plt.savefig('permutation_feature_importance.png')
plt4.show()


# 5.1.3 global surrogate model
blackbox_model = RandomForestClassifier()
blackbox_model.fit(X_train, y_train)

new_target = blackbox_model.predict(X_train)
surrogate_model = DecisionTreeClassifier(max_depth=6)
surrogate_model.fit(X_train, new_target)

feat_importances_df = pd.DataFrame(surrogate_model.feature_importances_, index=X_train.columns, columns=["Importance"])
feat_importances_df.index.name = 'Feature'
feat_importances_df = feat_importances_df[feat_importances_df.Importance != 0]
feat_importances_df.sort_values(by='Importance', ascending=False, inplace=True)

plt5 = plt.figure(figsize=(25, 15))
sns. barplot(x=feat_importances_df.index, y=feat_importances_df.Importance)
plt.title("Feature Importance graph by Global Surrogate Method")
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.xticks(rotation=90)

# plt.savefig('surrogate_feature_importance.png')
plt5.show()


# 5.2.1 lime
X_lime = X.to_numpy()
y_lime = y.to_numpy()

X_train_lime, X_test_lime, y_train_lime, y_test_lime = train_test_split(X_lime, y_lime, random_state=24, train_size=0.70)

forest_clf = RandomForestClassifier()
forest_clf.fit(X_train_lime, y_train_lime)

explainer = lime_tabular.LimeTabularExplainer(
    training_data=X_train_lime,
    feature_names=df.columns.values[1:],
    class_names=np.array(['malignant', 'benign']),
    mode='classification'
)

exp = explainer.explain_instance(
    data_row=X_test_lime[4],
    predict_fn=forest_clf.predict_proba,
    num_features=30
)

fig = exp.as_pyplot_figure()
plt.tight_layout()
# plt.savefig('lime_graph.png')
plt.show()


# 5.2.2 shap
shap_explainer = shap.Explainer(random_clf.predict, X_train)
shap_values = shap_explainer(X_test)
# print(np.shape(shap_values.values))
shap.plots.waterfall(shap_values[0])


