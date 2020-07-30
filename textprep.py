import re
import os
import dill


class TextPrep:
    def __init__(self):
        super().__init__()
        self.amazing_sub = {'amazing', 'awesome', 'stunning', 'astounding'}
        self.amazingly_sub = {'amazingly', 'stunningly', 'astoundingly'}
        self.love_sub = {'love', 'adore'}
        self.great_sub = {'great', 'really nice'}
        self.imdb_dir = './IMDB'
        self.stopwords = {'a', 'and', 'for', 'of', 'that', 'are', 'i', 'am', 'on', 'this', 'the', 'try', 'it', 'its', 'it\'s',
                          'to', 'in', 'an', 'these', 'his', 'her', 'in', 'if', 'as', 'he', 'she', 'me', 'i.e.', 'i\'ll',
                          'e.g.', 'at', 'e', 'g', 'my', 'i\'m', 'was', 'with', 'we', 'i\'ve', 'wa', 'you', 'ha', 'doe'}

    # Data augmentation methods
    def data_augmentation(self):
        # To be added
        pass

    # Load data from source files.
    def load_from_source(self, name='train'):
        labels = []
        text = []
        dir = os.path.join(self.imdb_dir, name)
        if name == 'train' or name == 'test':
            for label_type in ['pos', 'neg']:
                dir_name = os.path.join(dir, label_type)
                for fname in os.listdir(dir_name):
                    with open(os.path.join(dir_name, fname)) as f:
                        x = f.read()
                        text.append(x)

                    if label_type == 'neg':
                        labels.append(0)
                    else:
                        labels.append(1)

            return text, labels

        elif name == 'unsup':
            for fname in os.listdir(dir):
                with open(os.path.join(dir, fname)) as f:
                    x = f.read()
                    text.append(x)

            return text, labels

    # Method for stripping punctuation etc
    def strip_punc(self, x, strip_tags=True, rem_punc=True):
        if strip_tags:
            x = re.sub(r'\<br', ' ', x)
            x = re.sub(r'\/br\>', ' ', x)
            x = re.sub(r'\/\>', ' ', x)

        if rem_punc:
            x = re.sub(r'[!;:,.()?-]', ' ', x)
            x = re.sub(r'[\']', '', x)

        return x

    # Method for removing unwanted lists of words
    def remove_stopwords(self, text, word_list, non_invert=True):
        stripped_text = []
        for review in text:
            tokens = review.split(" ")
            if non_invert:
                tokens_filtered = [word for word in tokens if not (word in word_list)]

            else:
                tokens_filtered = [word for word in tokens if word in word_list]

            tokens_filtered = " ".join(tokens_filtered)
            stripped_text.append(tokens_filtered)

        return stripped_text

    # Format text
    def text_formatting(self, text, save=False):
        text = [x.lower() for x in text]
        text = self.remove_stopwords(text, self.stopwords)
        text = [self.strip_punc(x) for x in text]
        text = self.remove_stopwords(text, self.stopwords)

        if save:
            dill.dump(text, open('saved_text.pkd', 'wb'))

        return text


