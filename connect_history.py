import pandas as pd
import glob
import os
import matplotlib as mpl
import matplotlib.pyplot as plt

directory = r'result\230614-023628_72x72_50epoch_vspo1-100-images'

histories = glob.glob(os.path.join(directory, '*.csv'))

print(histories)

data = pd.concat(map(lambda history: pd.read_csv(history), histories)).reset_index(drop=True)

plt.plot(data['loss'])
plt.plot(data['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(f'{directory}/total-loss.png')
plt.show()
plt.clf()

plt.plot(data['accuracy'])
plt.plot(data['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.savefig(f'{directory}/total-acc.png')
plt.show()
plt.clf()
