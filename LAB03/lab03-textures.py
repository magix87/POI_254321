import tkinter as tk
from tkinter import filedialog, messagebox
import os
from skimage import io, color
from skimage.util import img_as_ubyte
from skimage.feature import graycomatrix, graycoprops
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


def choose_texture_folder(prompt):
    root = tk.Tk()
    root.withdraw()  # Ukrycie głównego okna
    folder_chosen = filedialog.askdirectory(title=prompt)
    root.destroy()
    return folder_chosen

def extract_and_resize_images(directory, resize=(128, 128)):
    processed_images = []
    for filename in os.listdir(directory):
        if filename.lower().endswith('.jpg'):
            img_path = os.path.join(directory, filename)
            img = io.imread(img_path)
            for i in range(0, img.shape[0], resize[0]):
                for j in range(0, img.shape[1], resize[1]):
                    if i + resize[0] <= img.shape[0] and j + resize[1] <= img.shape[1]:
                        processed_images.append(img[i:i + resize[0], j:j + resize[1]])
    return processed_images

def compute_texture_features(images, dist_list, angle_list, feature_list, level_count=64):
    feature_data = []
    for img in images:
        gray_img = color.rgb2gray(img)
        gray_img = img_as_ubyte(gray_img) // 4
        glcm_matrix = graycomatrix(gray_img, distances=dist_list, angles=angle_list, levels=level_count, symmetric=True, normed=True)
        feature_vector = []
        for prop in feature_list:
            prop_values = graycoprops(glcm_matrix, prop).ravel()
            feature_vector.append(prop_values)
        feature_data.append(np.concatenate(feature_vector))
    return np.array(feature_data)

def setup_classifier():
    # Ustawienie modelu z opcją probability
    return SVC(kernel='linear', probability=True)
def execute_image_classification(model, scaler):
    root = tk.Tk()
    root.withdraw()
    continue_classifying = True
    while continue_classifying:
        image_file = filedialog.askopenfilename(title="Select an Image to Classify", filetypes=[("JPEG files", "*.jpg")])
        if image_file:
            image = io.imread(image_file)
            image = img_as_ubyte(image)
            image_features = compute_texture_features([image], [1, 3, 5], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM'])
            scaled_features = scaler.transform(image_features)
            probabilities = model.predict_proba(scaled_features)[0]  # Pobranie prawdopodobieństw dla pierwszego (i jedynego) obrazu
            prediction = model.predict(scaled_features)
            # Formatowanie wiadomości z maksymalnym prawdopodobieństwem
            max_prob = np.max(probabilities)
            message = f"Predicted Texture: {prediction[0]}\nConfidence: {max_prob:.2f}"
            messagebox.showinfo("Prediction Results", message)
            continue_classifying = messagebox.askyesno("Continue", "Do you want to classify another image?")
        else:
            continue_classifying = False
    root.destroy()

def main():
    texture_folders = [choose_texture_folder(f'Select Texture Folder {i + 1}') for i in range(4)]
    all_features, all_labels = [], []
    feature_properties = ['dissimilarity', 'correlation', 'contrast', 'energy', 'homogeneity', 'ASM']
    distances = [1, 3, 5]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    feature_scaler = StandardScaler()

    for folder in texture_folders:
        images = extract_and_resize_images(folder)
        features = compute_texture_features(images, distances, angles, feature_properties)
        all_features.extend(features)
        all_labels.extend([os.path.basename(folder)] * len(features))

    feature_df = pd.DataFrame(all_features)
    feature_df['label'] = all_labels
    feature_df.to_csv('texture_features.csv', index=False)

    data = pd.read_csv('texture_features.csv')
    X = data.iloc[:, :-1].values
    y = data['label'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    X_train_scaled = feature_scaler.fit_transform(X_train)
    X_test_scaled = feature_scaler.transform(X_test)

    texture_classifier = setup_classifier()
    texture_classifier.fit(X_train_scaled, y_train)
    predictions = texture_classifier.predict(X_test_scaled)

    execute_image_classification(texture_classifier, feature_scaler)


if __name__ == "__main__":
    main()
