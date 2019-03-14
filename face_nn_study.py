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

def train_and_validate(model, X, y):
    start_train = time.time()
    model.fit(X, y)
    print ("training time: "+str(time.time()-start_train))
    print("loss: "+str(model.loss))

    test_pred = model.predict(X)
    print("training data accuracy: "+str(round(accuracy_score(y, test_pred)*100,2))+"%")

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--procedure", default=0, type=int,
            help="Processing data by: [0]dimensionality reduction only \
                                      [1]dimensionality reduction then clustering (default: 0)")
    ap.add_argument("-c", "--components", default=100, type=int,
            help="number of components for dimensionality reduction (default: 100)")
    ap.add_argument("-n", "--clusters", default=100, type=int,
            help="number of clusters for clustering (default: 100)")
    params = vars(ap.parse_args())


    faces = load_encodings(TRAIN_DATASET)
#    test_faces = load_encodings(TEST_DATASET)

    # make target to binary list
    X_train = faces["encodings"]
#    X_test = test_faces["encodings"]
    y_train = faces["names"]
#    y_test = test_faces["names"]
    y_train = [ 1 if y == THE_ONE else 0 for y in faces["names"] ]
 #   y_test = [ 1 if y == THE_ONE else 0 for y in test_faces["names"] ]

    np.random.seed(1)
    # build neural network
    based_model = mlrose.NeuralNetwork(hidden_nodes = [50, 20, 8, 2], activation = "relu", \
                                       bias = True, is_classifier = True, early_stopping = True, \
                                       algorithm="gradient_descent", \
                                       max_iters=1000, max_attempts = 100, \
                                       learning_rate = 0.001)

    print("************** based network ***************")
    train_and_validate(based_model, X_train, y_train)

    X_list = []
    ## PCA
    transformed = PCA(n_components=params["components"], svd_solver='randomized', whiten=True).fit_transform(np.array(X_train))
    X_list.append(transformed)

    ## FastICA
    transformed = FastICA(n_components=params["components"], random_state=0, whiten=True).fit_transform(X_train)
    X_list.append(transformed)

    ## Randomed Projections
    transformed = GaussianRandomProjection(n_components=params["components"], random_state=0).fit_transform(X_train)
    X_list.append(transformed)

    ## FeatureAgglomeration 
    transformed = FeatureAgglomeration(n_clusters=2).fit_transform(X_train)
    X_list.append(transformed)

    for idx in range(4):
        print("************** network with {} ***************".format(DIMEN_REDUCTION[idx]))
        train_and_validate(based_model, X_list[idx], y_train)

    if params["procedure"] >= 1:
        for idx in range(4):
            print("************** network with K-means after {} ***************".format(DIMEN_REDUCTION[idx]))
            kmeans = KMeans(n_clusters=params["clusters"], random_state=0).fit(X_list[idx])
            #single_attr_X = [ [x] for x in kmeans.labels_ ]
            multi_attr_X = [ X_list[idx][i]+[kmeans.labels_[i]] for i in range(len(kmeans.labels_)) ]
            #train_and_validate(based_model, single_attr_X, y_train)
            train_and_validate(based_model, multi_attr_X, y_train)

            print("************** network with EM after {} ***************".format(DIMEN_REDUCTION[idx]))
            em = GaussianMixture(n_components=params["clusters"], covariance_type='full', random_state=0).fit(X_list[idx])
            labels = em.predict(X_list[idx])
            multi_attr_X = [ X_list[idx][i]+[labels[i]] for i in range(len(labels)) ]
            train_and_validate(based_model, multi_attr_X, y_train)


