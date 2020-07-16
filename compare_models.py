#Draw images using bokeh on train / test accuracies

bench_dict = history_bench.history
small_dict = history_small.history
iter1_dict = history_iter1.history
iter2_dict = history_iter2.history
epochs_ar = range(1, epochs+1)
plt.plot(epochs_ar, bench_dict['accuracy'], 'bo', label='Training basline')
plt.plot(epochs_ar, bench_dict['val_accuracy'], 'b', label='Validation baseline')
plt.plot(epochs_ar, small_dict['accuracy'], 'yo', label='Training small')
plt.plot(epochs_ar, small_dict['val_accuracy'], 'y', label='Validation small')
plt.plot(range(1, 4), iter1_dict['accuracy'], 'ro', label='Training iterative')
plt.plot(range(1, 4), iter1_dict['val_accuracy'], 'r', label='Validation iterative')
plt.plot(range(4, 9), iter2_dict['accuracy'], 'ro')
plt.plot(range(4, 9), iter2_dict['val_accuracy'], 'r')
_ = [iter2_dict['val_accuracy'][3], iter2_dict['val_accuracy'][0]]
plt.plot(range(3, 5), _, 'r')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()