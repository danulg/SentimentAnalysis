#Draw images using bokeh on train / test accuracies
from matplotlib import pyplot as plt
from bokeh.plotting import figure, show, output_file
import dill
from collections import Counter
from wordcloud import WordCloud, STOPWORDS

class PlotCurves():
    def __init__(self):
        super().__init__()

    def draw(self, model_histories, sub_epochs=5, iterates=2):
        # Plot non-iterative training graphs
        dedicated_dict = model_histories[0]
        transfer_dict = model_histories[1]
        epochs_ar = range(1, sub_epochs*iterates+1)
        plt.plot(epochs_ar, dedicated_dict['acc'], 'bo', label='Training basline')
        plt.plot(epochs_ar, dedicated_dict['val_acc'], 'b', label='Validation baseline')
        plt.plot(epochs_ar, transfer_dict['acc'], 'yo', label='Training accuracy transfer-learning')
        plt.plot(epochs_ar, transfer_dict['val_acc'], 'y', label='Validation transfer-learning')

        iter_history = model_histories[2]
        length = len(iter_history)
        i = 0
        while(i<length):
            iter_dict = iter_history[i]
            if i==0:
                plt.plot(range(1, sub_epochs+1), iter_dict['acc'], 'ro', label='Training accuracy semi-supervised')
                plt.plot(range(1, sub_epochs+1), iter_dict['val_acc'], 'r', label='Validation accuracy semi-supervised')
                i+=1
            else:
                plt.plot(range(1+sub_epochs*i, 1+sub_epochs*(i+1)), iter_dict['acc'], 'ro')
                plt.plot(range(1+sub_epochs*i, 1+sub_epochs*(i+1)), iter_dict['val_acc'], 'r')
                _ = [iter_history[i-1]['val_acc'][sub_epochs-1], iter_dict['val_acc'][0]]
                plt.plot(range(sub_epochs*i, sub_epochs*i+2), _, 'r')
                i+=1

        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

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

    def draw_word_cloud(self):
        with open('train_text.pkd', 'rb') as f:
            text = dill.load(f)

        stopwords = set(STOPWORDS)
        stopwords.update(
            ["movie", "hi", 'film', 'wa', 'this', 'this movie', 'whole', "the whole", 'thi', 'story', 'ha'])
        text = " ".join(review for review in text)
        text = text.lower()
        # Create and generate a word cloud image:
        wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

        # Display the generated image:
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        plt.show()
        pass

    def draw_bar_chart(self, word):
        pass


if __name__=="__main__":
    curves = PlotCurves()
    curves.bokeh_draw(dill.load(open('history_1.pkd', 'rb')), sub_epochs=4, iterates=2)

    model_histories = dill.load(open('history_bidirectional_4.pkd', 'rb'))
    curves.bokeh_draw(model_histories, sub_epochs=8, iterates=2)
    pass