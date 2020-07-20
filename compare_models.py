#Draw images using bokeh on train / test accuracies
from matplotlib import pyplot as plt
import dill
import bokeh

class PlotCurves():
    def __init__(self):
        super().__init__()

    def draw(self, model_histories, sub_epochs=5, iterates=2):
        #Plot non-iterative training graphs
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

if __name__ == "__main__":
    # Draw plots
    curves = PlotCurves()
    curves.draw(dill.load(open('history_1.pkd', 'rb')))