import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from keras import Sequential
from tensorflow.keras.layers import Input, Dense
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


def select_file(prompt):
    root = tk.Tk()
    root.withdraw()  # Ukrycie głównego okna
    file_selected = filedialog.askopenfilename(title=prompt)  # Wybór pliku
    root.destroy()
    return file_selected


def load_data(filepath):
    data = pd.read_csv(filepath)
    X = data.drop('label', axis=1)
    y = data['label']
    print(f"Liczba unikalnych klas: {data['label'].nunique()}")
    return X, y


def preprocess_labels(y):
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    onehot_encoder = OneHotEncoder()
    y_onehot = onehot_encoder.fit_transform(y_encoded.reshape(-1, 1)).toarray()
    return y_onehot


def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Input(shape=(input_shape,)))
    model.add(Dense(10, activation='sigmoid'))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    return model


def train_and_evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, epochs=100, batch_size=10, shuffle=True)
    y_pred = model.predict(X_test)
    y_pred_int = np.argmax(y_pred, axis=1)
    y_test_int = np.argmax(y_test, axis=1)
    cmatrix = confusion_matrix(y_test_int, y_pred_int)
    return cmatrix


def main():
    file_path = select_file("Wybierz plik")
    #file_path = "texture_features_lab04.csv" #ominiecie tkintera w jupyter i google colab

    X, y = load_data(file_path)
    y_onehot = preprocess_labels(y)
    num_classes = y_onehot.shape[1]
    X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.3)

    model = build_model(X_train.shape[1], num_classes)
    cmatrix = train_and_evaluate_model(model, X_train, y_train, X_test, y_test)
    print("Macierz pomyłek:\n", cmatrix)


if __name__ == "__main__":
    main()
