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
DIMEN_REDUCTION=["None", "PCA", "FastICA", "GaussianRandomProjection", "FeatureAgglomeration"]

def load_encodings(encoding_file):
    # load the known faces and embeddings
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
            break

    return data

if __name__ == "__main__":

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--procedure", default=0, type=int,
            help="Processing data by: [0]clustering only \
                                      [1]dimensionality reduction then clustering \
                                      [2]dimensionality reduction only (default: 0)")
    ap.add_argument("-n", "--components", default=100, type=int,
            help="number of components for dimensionality reduction (default: 100)")
    params = vars(ap.parse_args())

    #load encodings and labels
    faces = load_encodings(PK_DATASET)
    pk_X = faces["encodings"]
    pk_y = faces["names"]
    pk_names = ["natalie_portman", "keira_knightley"] 

    faces = load_encodings(MOM_DATASET)
    mom_X = faces["encodings"]
    mom_y = faces["names"]
    mom_names = ["brigitte_lin", "michelle_yeoh", "sandra_bullock"] 

    # list for putting dimension reduced data 
    # in ordering: original data, PCA, FastICA, Randomized Projections, FeatureAgglomeration
    pk_X_list = [pk_X] 
    mom_X_list = [mom_X]

    if params["procedure"] > 0:
        ## PCA
        transformed = PCA(n_components=params["components"], svd_solver='randomized', whiten=True).fit_transform(np.array(pk_X))
        pk_X_list.append(transformed)

        transformed = PCA(n_components=params["components"], svd_solver='randomized', whiten=True).fit_transform(np.array(mom_X))
        mom_X_list.append(transformed)

        ## FastICA
        transformed = FastICA(n_components=params["components"], random_state=0, whiten=True).fit_transform(pk_X)
        pk_X_list.append(transformed)

        transformed = FastICA(n_components=params["components"], random_state=0, whiten=True).fit_transform(mom_X)
        mom_X_list.append(transformed)

        ## Randomed Projections
        transformed = GaussianRandomProjection(n_components=params["components"], random_state=0).fit_transform(pk_X)
        pk_X_list.append(transformed)
    
        transformed = GaussianRandomProjection(n_components=params["components"], random_state=0).fit_transform(mom_X)
        mom_X_list.append(transformed)

        ## FeatureAgglomeration
        transformed = FeatureAgglomeration(n_clusters=2).fit_transform(pk_X)
        pk_X_list.append(transformed)

        transformed = FeatureAgglomeration(n_clusters=2).fit_transform(mom_X)
        mom_X_list.append(transformed)

    if params["procedure"] < 2:
        #K-means    
        for x_id in range(len(pk_X_list)):
            kmeans = KMeans(n_clusters=2, random_state=0).fit(pk_X_list[x_id])
            pk_labels = [[],[]]
            for idx in range(len(kmeans.labels_)):
                pk_labels[kmeans.labels_[idx]].append(pk_y[idx])
           
            print("====== PK dataset K-means after {} ======".format(DIMEN_REDUCTION[x_id]))
            for idx in range(len(pk_labels)):
                group_size = len(pk_labels[idx])
                print("label {id}: {name0}={cover0}% / {name1}={cover1}%".format(\
                        id=idx, name0=pk_names[0], cover0=round(pk_labels[idx].count(pk_names[0])/group_size*100, 2),\
                        name1=pk_names[1], cover1=round(pk_labels[idx].count(pk_names[1])/group_size*100, 2)
                        ))
        
        print("")
    
        for x_id in range(len(mom_X_list)):
            kmeans = KMeans(n_clusters=3, random_state=0).fit(mom_X_list[x_id])
            mom_labels = [[],[],[]]
            for idx in range(len(kmeans.labels_)):
                mom_labels[kmeans.labels_[idx]].append(mom_y[idx])
        
            print("====== Mom dataset K-means after {} ======".format(DIMEN_REDUCTION[x_id]))
            for idx in range(len(mom_labels)):
                group_size = len(mom_labels[idx])
                print("label {id}: {name0}={cover0}% / {name1}={cover1}% / {name2}={cover2}%".format(\
                        id=idx, name0=mom_names[0], cover0=round(mom_labels[idx].count(mom_names[0])/group_size*100, 2),\
                        name1=mom_names[1], cover1=round(mom_labels[idx].count(mom_names[1])/group_size*100, 2),\
                        name2=mom_names[2], cover2=round(mom_labels[idx].count(mom_names[2])/group_size*100, 2)
                        ))
    
        print("")
    
        #EM: Gaussian mixture models

        for x_id in range(len(pk_X_list)):
            em = GaussianMixture(n_components=2, covariance_type='full', random_state=0).fit(pk_X_list[x_id])
            pk_predict_y = em.predict(pk_X_list[x_id])
            pk_labels = [[],[]]
            for idx in range(len(pk_predict_y)):
                pk_labels[pk_predict_y[idx]].append(pk_y[idx])
        
            print("====== PK dataset EM after {} ======".format(DIMEN_REDUCTION[x_id]))
            for idx in range(len(pk_labels)):
                group_size = len(pk_labels[idx])
                print("label {id}: {name0}={cover0}% / {name1}={cover1}%".format(\
                        id=idx, name0=pk_names[0], cover0=round(pk_labels[idx].count(pk_names[0])/group_size*100, 2),\
                        name1=pk_names[1], cover1=round(pk_labels[idx].count(pk_names[1])/group_size*100, 2)
                        ))
        
        print("")

        for x_id in range(len(mom_X_list)):
            em = GaussianMixture(n_components=3, covariance_type='full', random_state=0).fit(mom_X_list[x_id])
            mom_predict_y = em.predict(mom_X_list[x_id])
            mom_labels = [[],[],[]]
            for idx in range(len(mom_predict_y)):
                mom_labels[mom_predict_y[idx]].append(mom_y[idx])
        
            print("====== Mom dataset EM after {} ======".format(DIMEN_REDUCTION[x_id]))
            for idx in range(len(mom_labels)):
                group_size = len(mom_labels[idx])
                print("label {id}: {name0}={cover0}% / {name1}={cover1}% / {name2}={cover2}%".format(\
                        id=idx, name0=mom_names[0], cover0=round(mom_labels[idx].count(mom_names[0])/group_size*100, 2),\
                        name1=mom_names[1], cover1=round(mom_labels[idx].count(mom_names[1])/group_size*100, 2),\
                        name2=mom_names[2], cover2=round(mom_labels[idx].count(mom_names[2])/group_size*100, 2)
                        ))
    

