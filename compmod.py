#Draw images using bokeh on train / test accuracies
from matplotlib import pyplot as plt
from bokeh.plotting import figure, show, output_file
import dill
from wordcloud import WordCloud, STOPWORDS
from dataloader import IMDBDataSet

class PlotCurves:
    def __init__(self):
        super().__init__()

    #The following code duplicates the functionality of the draw but using bokeh for interactive plots
    def bokeh_draw(self, model_histories, sub_epochs=5, iterates=2):
        #Plot non-iterative training graphs
        dedicated_dict = model_histories[0]
        transfer_dict = model_histories[1]
        epochs_ar = range(1, sub_epochs*iterates+1)
        fig = figure(title='Epochs vs. Accuracy', x_axis_label='Epochs', y_axis_label='Accuracy', plot_width=1200, plot_height=800)

        # Draw training accuracy
        fig.circle(epochs_ar, dedicated_dict['acc'], size=20, color="blue", alpha=0.5, legend_label='Dedicated')
        fig.circle(epochs_ar, transfer_dict['acc'], size=20, color="green", alpha=0.5, legend_label='Transfer Learning')
        fig.line(epochs_ar, dedicated_dict['val_acc'], color='blue', legend_label='Dedicated: Validation')
        fig.line(epochs_ar, transfer_dict['val_acc'], color='green', legend_label='Transfer: Validation')


        iter_history = model_histories[2]
        length = len(iter_history)
        i = 0
        while(i<length):
            iter_dict = iter_history[i]
            if i==0:
                fig.circle(range(1, sub_epochs+1), iter_dict['acc'], size=20, color="red", alpha=0.5, legend_label='Iterative')
                fig.line(range(1, sub_epochs+1), iter_dict['val_acc'], color='red', legend_label='Dedicated: Validation')
                i+=1
            else:
                fig.circle(range(1+sub_epochs*i, 1+sub_epochs*(i+1)), iter_dict['acc'], size=20, color="red", alpha=0.5)
                fig.line(range(1+sub_epochs*i, 1+sub_epochs*(i+1)), iter_dict['val_acc'], color='red')
                x = [iter_history[i-1]['val_acc'][sub_epochs-1], iter_dict['val_acc'][0]]
                fig.line(range(sub_epochs*i, sub_epochs*i+2), x, color='red')
                i+=1

        # show and save the results
        name = str(sub_epochs)+'_'+str(iterates)+'.html'
        output_file(name)
        show(fig)

    def draw_word_cloud(self, name='train'):
        imdb = IMDBDataSet()
        text, _ = imdb.reviews(name=name, ret_val=True)

        stopwords = {'is', 'film is', 'movie is', 'film', 'movie', 'or', 'but', 'all', 'from', 'there', 'then', 'also',
                     'when', 'really', 'before', 'after', 'guy', 'girl', 'who is', 'part', 'such', 'by', 'kid', 'here',
                     'because', 'their', 'has', 'actually', 'man', 'still', 'any', 'show', 'series', 'however', 'while',
                     'yet', 'get', 'work', 'which', 'had', 'make', 'where', 'take', 'is not', 'isnt', 'who is',
                     'just', 'only', 'looking', 'having', 'very', 'would', 'have', 'not', 'so', 'is one', 'could',
                     'maybe', 'is no', 'again', 'although', 'is one', 'who is', 'two', 'episode', 'character', 'even',
                     'story is', 'although', 'into', 'like', 'love', 'one', 'think', 'another', 'have', 'been',
                     'have been', 'him', 'her', 'some', 'way', 'story', 'well', 'book', 'those', 'who', 'who is',
                     'now look', 'time', 'scene', 'now', 'look', 'life', 'them', 'play', 'actor', 'director', 'being',
                     'role', 'plot', 'out', 'would', 'would be', 'be', 'acting', 'actress', 'both', 'fact', 'update',
                     'little', 'end', 'lot', 'most', 'made', 'world', 'should have', 'i thought', 'that', 'the', 'idea',
                     'say', 'almost', 'so much', 'over', 'day', 'use', 'though', 'people', 'great', 'since', 'give',
                     'new', 'more', 'than', 'did', 'set', 'course', 'star', 'family', 'script', 'which', 'real'
                     }
        # stopwords.update(
        #     ["movie", "hi", 'film', 'wa', 'this', 'this movie', 'whole', "the whole", 'thi', 'story', 'ha', 'doe'])
        text = " ".join(review for review in text)
        # text = text.lower()
        # Create and generate a word cloud image: How to change scale?
        wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()


    def draw_bar_chart(self, word):
        pass


if __name__=="__main__":
    curves = PlotCurves()
    # curves.bokeh_draw(dill.load(open('history_1.pkd', 'rb')), sub_epochs=4, iterates=2)
    curves.draw_word_cloud()

    #model_histories = dill.load(open('history_bidirectional_4.pkd', 'rb'))
    #curves.bokeh_draw(model_histories, sub_epochs=8, iterates=2)