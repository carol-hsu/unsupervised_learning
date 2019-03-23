import argparse
import pickle
import cv2
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection
from sklearn.cluster import KMeans, FeatureAgglomeration
from sklearn.mixture import GaussianMixture
import mlrose
from sklearn.metrics import accuracy_score
import time 

TRAIN_DATASET="./pickles/pk_large.pickle"
TEST_DATASET="./pickles/pk_tiny.pickle"
THE_ONE="natalie_portman"
DIMEN_REDUCTION=["PCA", "FastICA", "GaussianRandomProjection", "FeatureAgglomeration"]

def load_encodings(encoding_file):
    # load the known faces and embeddings
    print("[INFO] loading encodings...")
    f = open(encoding_file, "rb")
    
    data = {}
    
    while True:
        try:
            if not data:
                data = pickle.load(f, encoding='latin1')
            else: 
                temp_data = pickle.load(f, encoding='latin1')
                data["encodings"] = data["encodings"] + temp_data["encodings"]
                data["names"] = data["names"] + temp_data["names"]
    
        except EOFError as e:
            #bad... but workable
            print("[INFO] Finish loading! Get "+str(len(data["names"]))+" faces ")
            break

    return data

def train_and_validate(X, y, X_t, y_t):
    model = mlrose.NeuralNetwork(hidden_nodes = [50, 20, 8, 2], activation = "relu", \
                                 bias = True, is_classifier = True, early_stopping = True, \
                                 algorithm="gradient_descent", \
                                 max_iters=1000, max_attempts = 100, learning_rate = 0.001)
    start_train = time.time()
    model.fit(X, y)
    print ("training time: "+str(time.time()-start_train))
    print("loss: "+str(model.loss))

    pred = model.predict(X)
    print("training data accuracy: "+str(round(accuracy_score(y, pred)*100,2))+"%")
    
    test_pred = model.predict(X_t)
    print("testing data accuracy: "+str(round(accuracy_score(y_t, test_pred)*100,2))+"%")
    


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--components", default=2, type=int,
            help="number of components for dimensionality reduction (default: 2)")
    ap.add_argument("-n", "--clusters", default=2, type=int,
            help="number of clusters for clustering (default: 2)")
    params = vars(ap.parse_args())


    faces = load_encodings(TRAIN_DATASET)
    test_faces = load_encodings(TEST_DATASET)

    # make target to binary list
    X_train = faces["encodings"]
    X_test = test_faces["encodings"]
    y_train = faces["names"]
    y_test = test_faces["names"]
    y_train = [ 1 if y == THE_ONE else 0 for y in faces["names"] ]
    y_test = [ 1 if y == THE_ONE else 0 for y in test_faces["names"] ]

    np.random.seed(1)
    # build neural network

    print("************** based network ***************")
    train_and_validate(X_train, y_train, X_test, y_test)

    X_list = []
    X_test_list = []

    ## PCA
    model = PCA(n_components=params["components"], svd_solver='randomized', whiten=True)
    X_list.append(model.fit_transform(np.array(X_train)))
    X_test_list.append(model.fit_transform(np.array(X_test)))

    ## FastICA
    model = FastICA(n_components=params["components"], random_state=0, whiten=True)
    X_list.append(model.fit_transform(X_train))
    X_test_list.append(model.fit_transform(X_test))

    ## Randomed Projections
    model = GaussianRandomProjection(n_components=params["components"], random_state=0)
    X_list.append(model.fit_transform(X_train))
    X_test_list.append(model.fit_transform(X_test))

    ## FeatureAgglomeration 
    model = FeatureAgglomeration(n_clusters=params["components"])
    X_list.append(model.fit_transform(X_train))
    X_test_list.append(model.fit_transform(X_test))

    for idx in range(4):
        print("************** network with {} ***************".format(DIMEN_REDUCTION[idx]))
        train_and_validate(X_list[idx], y_train, X_test_list[idx], y_test)

    for idx in range(4):
        print("************** network with K-means after {} ***************".format(DIMEN_REDUCTION[idx]))
        kmeans = KMeans(n_clusters=params["clusters"], random_state=0, copy_x=True)
        X_pred = kmeans.fit_predict(X_list[idx])
        X_test_pred = kmeans.fit_predict(X_test_list[idx])
        multi_attr_X = [ X_list[idx][i]+[X_pred[i]] for i in range(len(X_pred)) ]
        multi_attr_X_test = [ X_test_list[idx][i]+[X_test_pred[i]] for i in range(len(X_test_pred)) ]
        train_and_validate(multi_attr_X, y_train, multi_attr_X_test, y_test)
 
        print("************** network with EM after {} ***************".format(DIMEN_REDUCTION[idx]))
        em = GaussianMixture(n_components=params["clusters"], covariance_type='full', random_state=0)
        X_pred = em.fit_predict(X_list[idx])
        X_test_pred = em.fit_predict(X_test_list[idx])
        multi_attr_X = [ X_list[idx][i]+[X_pred[i]] for i in range(len(X_pred)) ]
        multi_attr_X_test = [ X_test_list[idx][i]+[X_test_pred[i]] for i in range(len(X_test_pred)) ]
        train_and_validate(multi_attr_X, y_train, multi_attr_X_test, y_test)


