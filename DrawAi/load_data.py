import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split

folder = "processed_data"
files = os.listdir(folder)


def load_data():
    x_load = []
    y_load = []
    x_load_test = []
    y_load_test = []

    count = 0
    for file in files:
        file = f'{folder}//{file}'

        x = np.load(file)

        # test = np.reshape(x[0], (28, 28))
        y = [count for _ in range(len(x))]
        y = np.array(y).astype('float32')
        y = y.reshape(y.shape[0], 1)

        x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0,
                                                            test_size=0.1,
                                                            shuffle=False)
        x_load.extend(x_train)
        y_load.extend(y_train)

        x_load_test.extend(x_test)
        y_load_test.extend(y_test)

        count += 1

    return x_load, y_load, x_load_test, y_load_test


features, labels, test_features, test_labels = load_data()

with open("divided_data/features", "wb") as f:
    pickle.dump(features, f, protocol=4)
with open("divided_data/labels", "wb") as f:
    pickle.dump(labels, f, protocol=4)

with open("divided_data/test_features", "wb") as f:
    pickle.dump(test_features, f, protocol=4)
with open("divided_data/test_labels", "wb") as f:
    pickle.dump(test_labels, f, protocol=4)
