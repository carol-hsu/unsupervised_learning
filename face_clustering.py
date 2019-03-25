import pickle
import argparse
import cv2
import numpy as np
from sklearn.decomposition import PCA, FastICA
from sklearn.random_projection import GaussianRandomProjection 
from sklearn.cluster import KMeans, FeatureAgglomeration
from sklearn.mixture import GaussianMixture

PK_DATASET="./pickles/pk_large.pickle"
MOM_DATASET="./pickles/mom_large.pickle"
OUTPUT_DIR="."#"./dimen_reduction"
DIMEN_REDUCTION=["None", "PCA", "FastICA", "GaussianRandomProjection", "FeatureAgglomeration"]

def load_encodings(encoding_file):
    # load the known faces and embeddings
    f = open(encoding_file, "rb")
    nums = 0
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
            nums = len(data["names"])
            break

    return data, nums

def print_outputs(labels, ds_size, names):
    for idx in range(len(labels)):
        group_size = len(labels[idx])
        label_out = "label {id} [{num}]:".format(id=idx, num=round(group_size/ds_size, 2))
        for i in range(len(names)-1):
            label_out += " {name}={cover}% /".format(\
                        name=names[i], cover=round(labels[idx].count(names[i])/group_size*100, 2))
        label_out += " {name}={cover}%".format(\
                    name=names[-1], cover=round(labels[idx].count(names[-1])/group_size*100, 2))
        print(label_out)

if __name__ == "__main__":

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--procedure", default=0, type=int,
            help="Processing data by: [0]clustering only \
                                      [1]dimensionality reduction then clustering \
                                      [2]dimensionality reduction only (default: 0)")
    ap.add_argument("-n", "--components", default=2, type=int,
            help="Number of components for dimensionality reduction (default: 2)")
    ap.add_argument("-d", "--dataset", default=0, type=int,
            help="Which dataset to use: [0] PK [1] Mom (default: 0)") 
    ap.add_argument("-c", "--clusters", default=2, type=int,
            help="Number of clusters for clustering (default: 2)")
    ap.add_argument("-o", "--output", action="store_true",
            help="Showing the projected features")
    ap.add_argument("-r", "--random-seed", default=0, type=int,
            help="random seed for random projection")
    params = vars(ap.parse_args())
    show_space = True

    #load encodings and labels
    X = y = names = faces = ds_size = None

    if params["dataset"] <= 0 :
        faces, ds_size = load_encodings(PK_DATASET)
        names = ["natalie_portman", "keira_knightley"] 
    else:
        faces, ds_size = load_encodings(MOM_DATASET)
        names = ["brigitte_lin", "michelle_yeoh", "sandra_bullock"]

    X = faces["encodings"]
    y = faces["names"]

    np.random.seed(0)
    # list for putting dimension reduced data 
    # in ordering: original data, PCA, FastICA, Randomized Projections, FeatureAgglomeration
    X_list = [X] 

    if params["procedure"] > 0:
        ## PCA
        transformed = PCA(n_components=params["components"], svd_solver='randomized', whiten=True).fit_transform(np.array(X))
        X_list.append(transformed)

        ## FastICA
        transformed = FastICA(n_components=params["components"], random_state=0, whiten=True).fit_transform(X)
        X_list.append(transformed)

        ## Randomed Projections
        transformed = GaussianRandomProjection(n_components=params["components"], random_state=params["random_seed"]).fit_transform(X)
        X_list.append(transformed)
    
        ## FeatureAgglomeration
        transformed = FeatureAgglomeration(n_clusters=params["components"]).fit_transform(X)
        X_list.append(transformed)

        if params["output"]:
            for algo in range(1,5):
                files = []
                for i in range(len(names)):
                    files.append(open(OUTPUT_DIR+"/"+names[i]+"_"+DIMEN_REDUCTION[algo]+".csv", "w"))

                for i in range(len(X_list[algo])):
                    if names[0] in y[i]:
                        files[0].write(str(X_list[algo][i][0])+","+str(X_list[algo][i][1])+"\n")
                    elif names[1] in y[i]:
                        files[1].write(str(X_list[algo][i][0])+","+str(X_list[algo][i][1])+"\n")
                    else:
                        files[2].write(str(X_list[algo][i][0])+","+str(X_list[algo][i][1])+"\n")
                for i in range(len(names)):
                    files[i].close()


    if params["procedure"] < 2:
        #K-means    
        for x_id in range(len(X_list)):
            kmeans = KMeans(n_clusters=params["clusters"], random_state=0).fit(X_list[x_id])
            labels = [[] for _ in range(params["clusters"])]
            for idx in range(len(kmeans.labels_)):
                labels[kmeans.labels_[idx]].append(y[idx])
           
            print("====== dataset {} with K-means after {} ======".format(params["dataset"],DIMEN_REDUCTION[x_id]))
            print_outputs(labels, ds_size, names)
            print("inertia: "+str(kmeans.inertia_))
            print("")

        print("")
    
        #EM: Gaussian mixture models

        for x_id in range(len(X_list)):
            em = GaussianMixture(n_components=params["clusters"], covariance_type='full', random_state=0).fit(X_list[x_id])
            predict_y = em.predict(X_list[x_id])
            labels = [[] for _ in range(params["clusters"])]
            for idx in range(len(predict_y)):
                labels[predict_y[idx]].append(y[idx])
        
            print("====== dataset {} with EM after {} ======".format(params["dataset"],DIMEN_REDUCTION[x_id]))
            print_outputs(labels, ds_size, names)
            print("components weights="+str(em.weights_))
            print("iters: "+str(em.n_iter_))
            print("")
        
