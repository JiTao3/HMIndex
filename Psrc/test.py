import imp
from train import *

if __name__ == "__main__":
    trainRandomInitialModle(
        "/data/jitao/dataset/Tiger/split_no_db/1386.csv",
        "/data/jitao/dataset/Tiger/trained_param_for_split_no_db/abc.csv",
        1386,
    )
