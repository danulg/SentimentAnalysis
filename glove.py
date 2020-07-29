import os
import numpy as np
from dataloader import IMDBDataSet

class LoadGloVe():
    def __init__(self, words):
        super().__init__()
        self.glove_dir = './IMDB/glove.6B'
        self.embeddings_index = {}
        self.word_index = words

    def create_corpus(self):
        imdb = IMDBDataSet()
        text, _ = imdb.reviews(name='unsup', ret_val=True)
        with open('unsup.txt', 'w') as filehandle:
            for mod_review in text:
                filehandle.write(mod_review + ' ')

    def load_glove(self, max_words=30000, embedding_dim=100, name='glove.6B.'):
        file_name = name+str(embedding_dim)+'d.txt'
        with open(os.path.join(self.glove_dir, file_name)) as f:
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                self.embeddings_index[word] = coefs

        print('Found %s word vectors.' % len(self.embeddings_index))

        embedding_matrix = np.zeros((max_words, embedding_dim))
        for word, i in self.word_index.items():
            if i < max_words:
                embedding_vector = self.embeddings_index.get(word)
                if embedding_vector is not None:
                    embedding_matrix[i] = embedding_vector

        return embedding_matrix


if __name__ == '__main__':
    # Check load functionality
    # temp = IMDBDataSet()
    # x, y, words = temp.load_data()
    # glove = LoadGloVe(words)
    #
    # matrix = glove.load_glove()
    # print(matrix.shape)

    # Check corpus
    glove = LoadGloVe([])
    glove.create_corpus()
