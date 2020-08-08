import tensorflow as tf
from dataloader import IMDBDataSet
from architectures import AnalysisBasic as Basic
from architectures import AnalysisBidirectional as Bidirectional
from architectures import ConvolutionalLSTM as ConvLSTM

if __name__ == '__main__':
    max_words = 20000
    embedding_dim = 100
    # Prep GPU:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print(gpus)
    tf.config.experimental.set_virtual_device_configuration(gpus[0],
                                                            [tf.config.experimental.VirtualDeviceConfiguration(
                                                                memory_limit=6000)])
    logical = tf.config.experimental.list_logical_devices('GPU')
    print(logical[0])

    # Load data
    # Load data
    imdb = IMDBDataSet()
    seq_text, labels, _ = imdb.load_data_default(name='test')

    model = ConvLSTM()
    model.build((max_words, embedding_dim))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    model.load_weights('conv_conv_lstm_model_100_32128_0.5.h5')
    model.evaluate(seq_text, labels)

# Code for reloading saved tokenizer and different approaches to saving tokenizer
# =========================================================================================================================
# Primary
# with open('tokenizer.json') as f:
#     data = json.load(f)
#     tokenizer = tokenizer_from_json(data)
# =========================================================================================================================
# tf.keras.preprocessing.text.tokenizer_from_json(json_string)

# Alternates
# import pickle
#
# # saving
# with open('tokenizer.pickle', 'wb') as handle:
#     pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# # loading
# with open('tokenizer.pickle', 'rb') as handle:
#     tokenizer = pickle.load(handle)