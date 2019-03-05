import argparse
import pickle
import cv2
import numpy as np
import mlrose
from sklearn.metrics import accuracy_score
import time 

DETECTION_METHOD="cnn"
THE_ONE="natalie_portman"

def encode_image(image_file):
    # load the input image and convert it from BGR to RGB
    image = cv2.imread(image_file)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # detect the (x, y)-coordinates of the bounding boxes corresponding
    # to each face in the input image, then compute the facial embeddings
    # for each face
    boxes = face_recognition.face_locations(rgb, model=DETECTION_METHOD)
    encodings = face_recognition.face_encodings(rgb, boxes)

    return encodings


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

def option_parser(input_param):
    #return a map of options
    params={"max_iter": 1500, "max_clip": 10, "lrate": 10, \
            "sched": 2, \
            "pop_size": 100, "mut_prob": 0.4}
    if input_param:
        pairs = input_param.split(",")
        for opt in pairs:
            key_value = opt.split("=")
            if key_value[0] in params.keys():
                params[key_value[0]] = float(key_value[1])

    return params
        

if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--train", required=True,
            help="directory of training set/encodings of training set")
    ap.add_argument("-t", "--test", required=True,
            help="directory of testing set/encodings of testing set")
    ap.add_argument("-r", "--optimization-algo", type=int, default=1,
            help="Use which randomized optimization algorithms: 1)random hill climbing 2)simulated annealing 3)genetic algorithm, default=1")
    ap.add_argument("-o", "--options",
            help="setting options based on the algorithms, configure in KEY=VALUE pair, and separate by comma. E.g. max_iter=1000,lrate=0.1 check\
                    README for complete configurable options")
    params = vars(ap.parse_args())
    
    options = option_parser(params["options"])


    algos = ["gradient_descent", "random_hill_climb", "simulated_annealing", "genetic_alg"]
    faces = load_encodings(params["train"])
    test_faces = load_encodings(params["test"])

    # make target to binary list
    X_train = faces["encodings"]
    X_test = test_faces["encodings"]
    y_train = [ 1 if y == THE_ONE else 0 for y in faces["names"] ]
    y_test = [ 1 if y == THE_ONE else 0 for y in test_faces["names"] ]

    np.random.seed(1)
    # build neural network
    based_model = mlrose.NeuralNetwork(hidden_nodes = [50, 20, 8, 2], activation = "relu", \
                                       bias = True, is_classifier = True, early_stopping = True, \
                                       algorithm=algos[0], \
                                       max_iters=1000, max_attempts = 100, \
                                       learning_rate = 0.001)


    print("************** based network ***************")
    start_train = time.time() 
    based_model.fit(X_train, y_train) 
    print ("training time: "+str(time.time()-start_train))
    print("loss: "+str(based_model.loss))
    
    test_pred = based_model.predict(X_train)
    print("training data accuracy: "+str(accuracy_score(y_train, test_pred)*100)+"%")

    test_pred = based_model.predict(X_test)
    #print(based_model.predicted_probs)
    print("testing data accuracy: "+str(accuracy_score(y_test, test_pred)*100)+"%")

    # RHC
    opti_model = mlrose.NeuralNetwork(hidden_nodes = [50, 20, 8, 2], activation = "relu", \
                                      bias = True, is_classifier = True, early_stopping = True, \
                                      algorithm=algos[1], \
                                      max_iters=options["max_iter"], max_attempts=options["max_iter"]*0.1, \
                                      learning_rate=options["lrate"])

    if params["optimization_algo"] == 2: # SA
        schedule = mlrose.GeomDecay()
        if options["sched"] == 1:
            schedule = mlrose.ArithDecay()
        elif options["sched"] == 2:
            schedule = mlrose.ExpDecay()

        opti_model = mlrose.NeuralNetwork(hidden_nodes = [50, 20, 8, 2], activation = "relu", \
                                          bias = True, is_classifier = True, early_stopping = True, \
                                          algorithm=algos[2], \
                                          max_iters=options["max_iter"], max_attempts=options["max_iter"]*0.1, \
                                          learning_rate=options["lrate"], \
                                          schedule=schedule)
    
    elif params["optimization_algo"] == 3: #GA
        opti_model = mlrose.NeuralNetwork(hidden_nodes = [50, 20, 8, 2], activation = "relu", \
                                          bias = True, is_classifier = True, early_stopping = True, \
                                          algorithm=algos[3], \
                                          max_iters=options["max_iter"], max_attempts=options["max_iter"]*0.1, \
                                          learning_rate=options["lrate"], \
                                          pop_size=options["pop_size"], mutation_prob=options["mut_prob"])

    print("************** after RO by "+algos[params["optimization_algo"]]+" ***************")

    start_train = time.time() 
    opti_model.fit(X_train, y_train, based_model.fitted_weights)
    print ("training time: "+str(time.time()-start_train))
    print("loss: "+str(opti_model.loss))
    
    test_pred = opti_model.predict(X_train)
    print("training data accuracy: "+str(accuracy_score(y_train, test_pred)*100)+"%")

    test_pred = opti_model.predict(X_test)
   # print(opti_model.predicted_probs)
    print("testing data accuracy: "+str(accuracy_score(y_test, test_pred)*100)+"%")
    
