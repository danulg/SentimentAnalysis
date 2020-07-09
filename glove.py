import os

    glove_dir = '/Users/fchollet/Downloads/glove.6B'
    embeddings_index = {}
    f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'))
    for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
    f.close()
    print('Found %s word vectors.' % len(embeddings_index))

    embedding_dim = 100
    embedding_matrix = np.zeros((max_words, embedding_dim))
    for word, i in word_index.items():
    if i < max_words:
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
    embedding_matrix[i] = embedding_vector