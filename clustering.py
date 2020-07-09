import numpy as np
import scipy
from dataloader import IMDBDataSet
import sklearn


if __name__=="__main__":
    data = IMDBDataSet()
    tr_dt, tr_lbl, word_index = data.load_data(name='train', is_numpy=True)
    train_data = tr_dt[:20000]
    train_label = tr_lbl[:20000]
    val_data = tr_dt[20000:]
    val_label = tr_dt[20000:]


embedding_dim = 100
