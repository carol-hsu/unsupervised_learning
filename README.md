# [Unsupervised Learning](https://github.com/carol-hsu/unsupervised_learing)

## Environment setup
Make sure you have Python3 on your machine.
After you pull this repo, following commands helps you installing the required packages.

```
// virtual environment is recommended, but optional
$ virtualenv venv -p python3

(venv) $ pip install -r requirements.txt 

```
## Run dimensionality reduction and clustering algorithms

`face_recongnition.py` is the entry point for this topic.

Check help messages `-h` for the details.

```
$ python face_clustering.py -h
usage: face_clustering.py [-h] [-p PROCEDURE] [-n COMPONENTS] [-d DATASET]
                          [-c CLUSTERS] [-o] [-r RANDOM_SEED]

optional arguments:
  -h, --help            show this help message and exit
  -p PROCEDURE, --procedure PROCEDURE
                        Processing data by: [0]clustering only
                        [1]dimensionality reduction then clustering
                        [2]dimensionality reduction only (default: 0)
  -n COMPONENTS, --components COMPONENTS
                        Number of components for dimensionality reduction
                        (default: 2)
  -d DATASET, --dataset DATASET
                        Which dataset to use: [0] PK [1] Mom (default: 0)
  -c CLUSTERS, --clusters CLUSTERS
                        Number of clusters for clustering (default: 2)
  -o, --output          Showing the projected features
  -r RANDOM_SEED, --random-seed RANDOM_SEED
                        random seed for random projection
```
apply the flags to test the algorithms you want


## Run dimensionality reduction and clustering algorithms with neural network

`face_nn_study.py` is the entry point for this topic.

Check help messages `-h` for the details.

```
$ python face_nn_study.py -h
usage: face_nn_study.py [-h] [-c COMPONENTS] [-n CLUSTERS]

optional arguments:
  -h, --help            show this help message and exit
  -c COMPONENTS, --components COMPONENTS
                        number of components for dimensionality reduction
                        (default: 2)
  -n CLUSTERS, --clusters CLUSTERS
                        number of clusters for clustering (default: 2)
```
