#Draw images using bokeh on train / test accuracies
from matplotlib import pyplot as plt
import dill
import bokeh

class PlotCurves():
    def __init__(self):
        super().__init__()

    def draw(self, model_histories, epochs=10):
        bench_dict = model_histories[0]
        small_dict = model_histories[1]
        iter1_dict = model_histories[2][0]
        iter2_dict = model_histories[2][1]
        print(bench_dict)
        epochs_ar = range(1, epochs+1)
        plt.plot(epochs_ar, bench_dict['acc'], 'bo', label='Training basline')
        plt.plot(epochs_ar, bench_dict['val_acc'], 'b', label='Validation baseline')
        plt.plot(epochs_ar, small_dict['acc'], 'yo', label='Training accuracy transfer-learning')
        plt.plot(epochs_ar, small_dict['val_acc'], 'y', label='Validation transfer-learning')
        plt.plot(range(1, 6), iter1_dict['acc'], 'ro', label='Training accuracy semi-supervised')
        plt.plot(range(1, 6), iter1_dict['val_acc'], 'r', label='Validation accuracy semi-supervised')
        plt.plot(range(6, 11), iter2_dict['acc'], 'ro')
        plt.plot(range(6, 11), iter2_dict['val_acc'], 'r')
        _ = [iter2_dict['val_acc'][3], iter2_dict['val_acc'][0]]
        plt.plot(range(5, 7), _, 'r')
        plt.title('Training and validation loss')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

if __name__ == "__main__":
    # Draw plots
    curves = PlotCurves()
    curves.draw(dill.load(open('history_1.pkd', 'rb')))