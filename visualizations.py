# Handles various visualizations
from matplotlib import pyplot as plt
from bokeh.plotting import figure, show, output_file
import dill
from wordcloud import WordCloud
from dataloader import IMDBDataSet
import spacy
from textprep import TextPrep
from sklearn.feature_extraction.text import CountVectorizer
from tensorflow.keras.utils import plot_model
from architectures import AnalysisBasic, AnalysisBidirectional, ConvolutionalLSTM


class PlotCurves:
    def __init__(self):
        super().__init__()
        self.nlp = spacy.load("en_core_web_sm")

    # The following code duplicates the functionality of the draw but using bokeh for interactive plots
    def draw_comparison(self, model_histories, epochs=20, name='basic'):
        # Plot non-iterative training graphs
        dedicated_dict = model_histories[0]
        transfer_dict = model_histories[1]
        semi_sup_dict = model_histories[2]
        epochs_ar = range(1, epochs + 1)
        fig = figure(title='Epochs vs. Accuracy', x_axis_label='Epochs', y_axis_label='Accuracy', plot_width=1200,
                     plot_height=800)

        # Draw training accuracy
        fig.circle(epochs_ar, dedicated_dict['acc'], size=20, color="blue", alpha=0.5, legend_label='Dedicated')
        fig.circle(epochs_ar, transfer_dict['acc'], size=20, color="green", alpha=0.5, legend_label='Transfer Learning')
        fig.circle(epochs_ar, semi_sup_dict['acc'], size=20, color="red", alpha=0.5,
                   legend_label='Semi-supervised Learning')
        fig.line(epochs_ar, dedicated_dict['val_acc'], color='blue', legend_label='Dedicated: Validation')
        fig.line(epochs_ar, transfer_dict['val_acc'], color='green', legend_label='Transfer: Validation')
        fig.line(epochs_ar, semi_sup_dict['val_acc'], color='red', legend_label='Semi-supervised: Validation')

        # show and save the results
        name = name + '_' + str(epochs) + '.html'
        output_file(name)
        show(fig)

    def draw_two(self, model_histories, epochs=60, name='basic'):
        # Plot non-iterative training graphs
        transfer_dict = model_histories[0]
        semi_sup_dict = model_histories[1]
        epochs_ar = range(1, epochs + 1)
        fig = figure(title='Epochs vs. Accuracy', x_axis_label='Epochs', y_axis_label='Accuracy', plot_width=1200,
                     plot_height=800)

        # Draw training accuracy
        fig.circle(epochs_ar, transfer_dict['acc'], size=20, color="green", alpha=0.5, legend_label='Transfer Learning')
        fig.circle(epochs_ar, semi_sup_dict['acc'], size=20, color="red", alpha=0.5,
                   legend_label='Semi-supervised Learning')
        fig.line(epochs_ar, transfer_dict['val_acc'], color='green', legend_label='Transfer: Validation')
        fig.line(epochs_ar, semi_sup_dict['val_acc'], color='red', legend_label='Semi-supervised: Validation')

        # show and save the results
        name = name + '_' + str(epochs) + '.html'
        output_file(name)
        show(fig)

    def draw_word_cloud(self, text):
        # stopwords = {'is', 'film is', 'movie is', 'film', 'movie', 'or', 'but', 'all', 'from', 'there', 'then', 'also',
        #              'when', 'really', 'before', 'after', 'guy', 'girl', 'who is', 'part', 'such', 'by', 'kid', 'here',
        #              'because', 'their', 'has', 'actually', 'man', 'still', 'any', 'show', 'series', 'however', 'while',
        #              'yet', 'get', 'work', 'which', 'had', 'make', 'where', 'take', 'is not', 'isnt', 'who is',
        #              'just', 'only', 'looking', 'having', 'very', 'would', 'have', 'not', 'so', 'is one', 'could',
        #              'maybe', 'is no', 'again', 'although', 'is one', 'who is', 'two', 'episode', 'character', 'even',
        #              'story is', 'although', 'into', 'like', 'love', 'one', 'think', 'another', 'have', 'been',
        #              'have been', 'him', 'her', 'some', 'way', 'story', 'well', 'book', 'those', 'who', 'who is',
        #              'now look', 'time', 'scene', 'now', 'look', 'life', 'them', 'play', 'actor', 'director', 'being',
        #              'role', 'plot', 'out', 'would', 'would be', 'be', 'acting', 'actress', 'both', 'fact', 'update',
        #              'little', 'end', 'lot', 'most', 'made', 'world', 'should have', 'i thought', 'that', 'the', 'idea',
        #              'say', 'almost', 'so much', 'over', 'day', 'use', 'though', 'people', 'great', 'since', 'give',
        #              'new', 'more', 'than', 'did', 'set', 'course', 'star', 'family', 'script', 'which', 'real'
        #              }
        # stopwords.update(
        #     ["movie", "hi", 'film', 'wa', 'this', 'this movie', 'whole', "the whole", 'thi', 'story', 'ha', 'doe'])
        text = " ".join(review for review in text)
        # text = text.lower()
        # Create and generate a word cloud image: How to change scale?
        # wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

        wordcloud = WordCloud(background_color="white").generate(text)

        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()

    def extract_list(self, name='train', wtype='VERB', strip=False):
        imdb = IMDBDataSet()
        text, _ = imdb.reviews(name=name, ret_val=True)
        name = wtype + '_list.pkd'
        text_ex = " ".join(review for review in text)

        if strip:
            text_split = []
            i = 0
            max_len = 1000000
            entries = len(text_ex) // max_len
            print(entries)
            while i <= entries - 1:
                text_split.append(text_ex[i * max_len:(i + 1) * max_len])
                i += 1

            text_split.append(text_ex[i * max_len:len(text)])
            word_list = set()

            for sub_string in text_split:
                doc = self.nlp(sub_string)
                temp_list = {token.lemma_ for token in doc if token.pos_ == wtype}
                word_list = word_list.union(temp_list)
                print(word_list)

            dill.dump(word_list, open(name, 'wb'))

        else:
            word_list = dill.load(open(name, 'rb'))

        prep = TextPrep()
        text = prep.remove_stopwords(text, word_list, non_invert=False)
        return text

    # Hashing vectorizer would give performance boost but does not have a vocabulary_ attribute.
    def count_based_removal(self, name='train', wtype='VERB', mindf=0, maxdf=100):
        imdb = IMDBDataSet()
        text, _ = imdb.reviews(name=name, ret_val=True)
        counts = CountVectorizer(min_df=mindf, max_df=maxdf)
        counts.fit_transform(text)
        word_list = list(counts.vocabulary_)
        print(word_list)
        name = 'words_min_'+str(mindf)+'_max_'+str(maxdf)+'.pkd'
        dill.dump(word_list, open(name, 'wb'))
        prep = TextPrep()
        text = prep.remove_stopwords(text, word_list, non_invert=False)
        self.draw_word_cloud(text)


    def draw_model(self, name='basic'):
        if name == 'basic':
            model = AnalysisBasic()
            plot_model(model, to_file='basic.png')

        elif name == 'bidirectional':
            model = AnalysisBidirectional()
            plot_model(model, to_file='bidirectional.png')

        elif name=='conv_lstm':
            model = ConvolutionalLSTM()
            model.build((20000, 100))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
            model.load_weights('w2vec_conv_lstm_model_100_32128_0.5.h5')

            plot_model(model, to_file='conv.png')



if __name__ == "__main__":
    curves = PlotCurves()
    curves.draw_model(name='conv_lstm')


    # curves.count_based_removal()
    # text = curves.extract_list(wtype='VERB')
    # curves.draw_word_cloud(text)

    # history = dill.load(open('history_conv_lstm_res.pkd', 'rb'))
    # curves.draw_two(history, epochs=60, name='LSTM_Conv_res')
    # curves.draw_two(dill.load(open('history_conv_lstm_res_100.pkd', 'rb')), epochs=100, name='conv_lstm_100')
    # curves.draw_word_cloud()
