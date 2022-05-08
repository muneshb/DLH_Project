import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

data_mat_mod = np.load('data_mat.npy') # load

pred_label_df_mod = pd.read_pickle("pred_label_df.pkl")
pred_label_df_mod['pred_label'] = 1
pred_label_df_mod.loc[pred_label_df_mod['main_label'] == "Diabetes", 'pred_label'] = 1
pred_label_df_mod.loc[pred_label_df_mod['main_label'] == "Rheumatoid Arthritis", 'pred_label'] = 2
pred_label_df_mod.loc[pred_label_df_mod['main_label'] == "Osteonecrosis", 'pred_label'] = 3

pre_label_arr = pred_label_df_mod.pred_label
pred_label_arr = pre_label_arr.to_numpy()

x = data_mat_mod.shape[0]
y = data_mat_mod.shape[1]
z = data_mat_mod.shape[2]

data_mat_arr = data_mat_mod.reshape(x, y*z)
data_mat_arr = np.nan_to_num(data_mat_arr)

def train_test_split_df(data_mat_arr, pred_label_arr, train_split=0.75, test_split = 0.25):
    n = data_mat_arr.shape[0]
    n_train = round(0.75*n)
    X_train = data_mat_arr[0:n_train, :]
    Y_train = pred_label_arr[0:n_train]
    X_test = data_mat_arr[n_train:n, :]
    Y_test = pred_label_arr[n_train:n]
    return X_train, Y_train, X_test, Y_test



def get_error_metrics(Y_test, pred, method = 'Euclidean'):
    precision, recall, fscore, support = precision_recall_fscore_support(Y_test, pred, average='macro')
    accuracy = accuracy_score(Y_test, pred)
    df = pd.DataFrame([[method,accuracy,precision, recall, fscore]],columns=['Method','Accuracy','Precision','Recall','F1 Score'])
    print(confusion_matrix(Y_test, pred))
    cmat = confusion_matrix(Y_test, pred)
    # print(classification_report(y_test, y_pred))
    return df, cmat


### less means its closer
def euclidean_sim(data_mat_arr, pred_label_arr):
    dist = euclidean_distances(data_mat_arr, data_mat_arr)

    X_train, Y_train, X_test, Y_test = train_test_split_df(dist, pred_label_arr, train_split=0.75,
                                                           test_split=0.25)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, Y_train)

    pred = knn.predict(X_test)
    error_df, cmat = get_error_metrics(Y_test, pred, method = 'Euclidean')
    return error_df, cmat



### larger values are similar
def cosine_sim(data_mat_arr, pred_label_arr):
    dist = cosine_similarity(data_mat_arr, data_mat_arr)
    X_train, Y_train, X_test, Y_test = train_test_split_df(dist, pred_label_arr, train_split=0.75,
                                                           test_split=0.25)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, Y_train)

    pred = knn.predict(X_test)
    error_df, cmat = get_error_metrics(Y_test, pred, method='Cosine')
    return error_df, cmat


def itml_sim(data_mat_arr, pred_label_arr):
    from dml import kNN, ITML
    X_train, Y_train, X_test, Y_test = train_test_split_df(data_mat_arr, pred_label_arr, train_split=0.75,
                                                           test_split=0.25)
    itml = ITML()
    knn = kNN(n_neighbors=5, dml_algorithm=itml)

    itml.fit(X_train, Y_train)
    knn.fit(X_train, Y_train)

    pred = knn.predict(X_test)
    error_df, cmat = get_error_metrics(Y_test, pred, method='ITML')
    return error_df, cmat


def lmnn_sim(data_mat_arr, pred_label_arr):
    from metric_learn import LMNN
    X_train, Y_train, X_test, Y_test = train_test_split_df(data_mat_arr, pred_label_arr, train_split=0.75,
                                                           test_split=0.25)
    lmnn = LMNN(k=5, learn_rate=1e-6)
    lmnn.fit(X_train, Y_train)

    pred = lmnn.predict(X_test)
    error_df, cmat = get_error_metrics(Y_test, pred, method='LMNN')

    return error_df, cmat

def pca_knn(data_mat_arr, pred_label_arr):
    pca = PCA(n_components=7)
    pca_transformed = pca.fit_transform(data_mat_arr)

    X_train, Y_train, X_test, Y_test = train_test_split_df(pca_transformed, pred_label_arr, train_split=0.75,
                                                           test_split=0.25)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, Y_train)

    pred = knn.predict(X_test)
    error_df, cmat = get_error_metrics(Y_test, pred, method='PCA')
    return error_df, cmat

def scml_sim(data_mat_arr, pred_label_arr):
    def generate_pairs(df):
        pairs_df = df[np.transpose(np.triu_indices(len(df), 1))]
        return pairs_df

    pairs = generate_pairs(data_mat_arr)
    scml = SCML(random_state=31)
    scml.fit(pairs)

euclidean_metric, euclidean_cmat = euclidean_sim(data_mat_arr, pred_label_arr)
cosine_metric, cosine_cmat = cosine_sim(data_mat_arr, pred_label_arr)
pca_metric, pca_cmat = pca_knn(data_mat_arr, pred_label_arr)
itml_metric, itml_cmat = itml_sim(data_mat_arr, pred_label_arr)
#lmnn_metric = lmnn_sim(data_mat_arr, pred_label_arr)  ### taking too long to run
#scml_metric = scml_sim(data_mat_arr, pred_label_arr)  ### taking too long to run

err_df = pd.concat([euclidean_metric,cosine_metric, itml_metric, pca_metric])
err_df.to_csv("results/Classification_results.csv")

