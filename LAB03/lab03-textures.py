import tkinter as tk
from tkinter import filedialog, messagebox
import os
from skimage import io, color, exposure
from skimage.util import img_as_ubyte
from skimage.feature import graycomatrix, graycoprops
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

def choose_directory_gui():
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory()
    return directory

def crop_images_in_directory(input_dir, crop_size=(128, 128)):
    cropped_images = []
    for filename in os.listdir(input_dir):
        if filename.lower().endswith('.jpg'):
            file_path = os.path.join(input_dir, filename)
            img = img_as_ubyte(io.imread(file_path))
            height, width = img.shape[:2]
            for i in range(0, height - crop_size[0] + 1, crop_size[0]):
                for j in range(0, width - crop_size[1] + 1, crop_size[1]):
                    if i + crop_size[0] <= height and j + crop_size[1] <= width:
                        crop_img = img[i:i + crop_size[0], j:j + crop_size[1]]
                        cropped_images.append(crop_img)
    return cropped_images

def convert_to_gray_and_quantize(images, levels=32):
    quantized_images = []
    for img in images:
        gray_img = color.rgb2gray(img)
        quantized_img = np.floor(gray_img * (levels - 1)).astype(np.uint8)
        quantized_images.append(quantized_img)
    return quantized_images

def calculate_texture_features(images, distances, angles, properties, levels=32):
    features_list = []
    for img in images:
        glcm = graycomatrix(img, distances, angles, levels=levels, symmetric=True, normed=True)
        features = {}
        for prop in properties:
            prop_values = graycoprops(glcm, prop)
            for i, dist in enumerate(distances):
                for j, angle in enumerate(angles):
                    features[f"{prop}_d{dist}_a{angle}"] = prop_values[i, j]
        features_list.append(features)
    return features_list

def main():
    total_features = []
    total_labels = []
    for _ in range(4):
        chosen_dir = choose_directory_gui()
        if chosen_dir:
            category = os.path.basename(chosen_dir)
            cropped_images = crop_images_in_directory(chosen_dir)
            processed_images = convert_to_gray_and_quantize(cropped_images)
            properties = ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']
            distances = [1, 3, 5]
            angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
            texture_features = calculate_texture_features(processed_images, distances, angles, properties)
            for feature in texture_features:
                total_features.append(list(feature.values()))
                total_labels.append(category)

    # Zapis danych do pliku CSV
    df = pd.DataFrame(total_features, columns=[f"feature_{i}" for i in range(len(total_features[0]))])
    df['label'] = total_labels
    df.to_csv('texture_features.csv', index=False)

    # Wczytanie danych
    data = pd.read_csv('texture_features.csv')
    X = data.iloc[:, :-1].values
    y = data['label'].values

    # Podział na zbiory treningowe i testowe
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Trening klasyfikatora SVM
    svm = SVC(kernel='linear')
    svm.fit(X_train, y_train)

    # Ocena klasyfikatora
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    # Pozwól użytkownikowi wielokrotnie wybierać obraz i przewidywać jego teksturę
    root = tk.Tk()
    root.withdraw()
    continue_classification = True
    while continue_classification:
        image_path = filedialog.askopenfilename(title="Select an Image", filetypes=[("JPEG files", "*.jpg")])
        if image_path:
            img = img_as_ubyte(io.imread(image_path))
            img_cropped = crop_images_in_directory(os.path.dirname(image_path))
            img_processed = convert_to_gray_and_quantize(img_cropped)
            img_features = calculate_texture_features(img_processed, distances, angles, properties)
            img_features_array = np.array([list(f.values()) for f in img_features])
            prediction = svm.predict(img_features_array)
            messagebox.showinfo("Predicted Texture", f"Predicted Texture: {prediction[0]} \nAccuracy: {accuracy:.2f}")
            continue_classification = messagebox.askyesno("Continue", "Do you want to classify another image?")
        else:
            continue_classification = False

if __name__ == "__main__":
    main()
