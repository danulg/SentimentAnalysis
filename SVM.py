import numpy as np
import scipy
from DataLoader import IMDBDataSet
import sklearn


if __name__=="__main__":
    data_load = IMDBDataSet()
    train_data, train_labels = data_load.load_original_train(is_numpy=True)



