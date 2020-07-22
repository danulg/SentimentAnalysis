from tensorflow import keras
from data_loader import IMDBDataSet
import tensorflow as tf
from network_architectures import SentimentAnalysisBasic

if __name__ == '__main__':
    #Create new model: Parameters should match with model being loaded
    model_arch = SentimentAnalysisBasic(rate=0.6)
    model_arch.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model_arch.load_weights('0.9_basic_model_4_3_128_128_128.h5')

    model_glove = SentimentAnalysisBasic(rate=0.6)
    model_glove.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model_glove.load_weights('0.9_glove_basic_model_4_3_128_128_128.h5')

    model_iter = SentimentAnalysisBasic(rate=0.6)
    model_iter.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model_iter.load_weights('0.9_basic_itermodel_4_3_128_128_128.h5')

    # model_iter = keras.models.load_model('0.95_basic_itermodel_4_3_128_128_128')
    temp = IMDBDataSet()
    test_dt, test_lbl, _ = temp.load_data(name='test')

    model_arch.evaluate(test_dt, test_lbl)
    model_glove.evaluate(test_dt, test_lbl)
    model_iter.evaluate(test_dt, test_lbl)
