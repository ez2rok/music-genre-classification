# import statements
import json
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow.keras as keras
import matplotlib.pyplot as plt

DATASET_PATH = "model-inputs.json"


# load data
def load_data(dataset_path):

    with open(dataset_path, "r") as fp:
        data = json.load(fp)

    # convert lists into numpy arrays
    inputs = np.array(data["mfcc"])
    targets = np.array(data["labels"])

    # get number of genres
    n_outputs = len(data["mapping"])

    print("Data succesfully loaded")

    return inputs, targets, n_outputs


def plot_history(history):

    fig, axs = plt.subplots(2)

    # accuracy subplot
    axs[0].plot(history.history['accuracy'], label="train accuracy")
    axs[0].plot(history.history['val_accuracy'], label="test accuracy")
    axs[0].set_ylabel('Accuracy')
    axs[0].legend(loc='lower right')
    axs[0].set_title('Multilayer Perceptron (MLP) Accuracy')

    # error/loss subplot
    axs[1].plot(history.history['loss'], label="train error")
    axs[1].plot(history.history['val_loss'], label="test error")
    axs[1].set_ylabel('Error')
    axs[1].set_xlabel('Epoch')
    axs[1].legend(loc='upper right')
    axs[1].set_title('Multilayer Perceptron (MLP) Error')

    plt.show()


if __name__ == "__main__":

    # load data
    X, y, n_outputs = load_data(DATASET_PATH)

    # split the data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # build network architecture
    model = keras.Sequential([
        # input layer
        keras.layers.Flatten(input_shape=(X.shape[1], X.shape[2])),

        # 1st hidden layer
        keras.layers.Dense(512, activation="relu",
                           kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 2nd hidden layer
        keras.layers.Dense(256, activation="relu",
                           kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # 3rd hidden layer
        keras.layers.Dense(64, activation="relu",
                           kernel_regularizer=keras.regularizers.l2(0.001)),
        keras.layers.Dropout(0.3),

        # output layer
        keras.layers.Dense(n_outputs, activation="softmax")
    ])

    # compile network
    optimizer = keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer,
                  loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    # train model
    history = model.fit(X_train, y_train,
                        validation_data=(X_test, y_test),
                        batch_size=32,
                        epochs=50
                        )

    # plot accuracy and error over the epochs
    plot_history(history)
