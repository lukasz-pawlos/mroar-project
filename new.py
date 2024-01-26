import cv2
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt


def load_model_and_codebook(model_filename, codebook_filename):
    loaded_model = joblib.load(model_filename)
    loaded_codebook = np.load(codebook_filename)
    return loaded_model, loaded_codebook

def predict_object(image_path, model, codebook, feature_detector):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    _, des = feature_detector.detectAndCompute(image, None)

    if des is not None and len(des) > 0:
        histogram = np.zeros(len(codebook))
        for feature in des:
            idx = np.argmin(np.linalg.norm(codebook - feature, axis=1))
            histogram[idx] += 1

        histogram = histogram.reshape(1, -1)
        prediction = model.predict(histogram)[0]

        return prediction
    else:
        return None
    
def get_file_names(directory_path):
    file_names = []

    try:
        # Sprawdź, czy katalog istnieje
        if os.path.exists(directory_path) and os.path.isdir(directory_path):
            # Uzyskaj listę plików w katalogu
            file_names = [file for file in os.listdir(directory_path) if os.path.isfile(os.path.join(directory_path, file))]
        else:
            print(f"Directory not found: {directory_path}")

    except Exception as e:
        print(f"Error: {e}")

    return file_names

def display_images(image1, image2, title1="Image 1", title2="Image 2"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    axes[0].set_title(title1)
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    axes[1].set_title(title2)
    axes[1].axis('off')

    plt.show()

def main():
    data_test_path = "DATATEST/"
    model_filename = "_10000_words_dict/test_model.joblib"  # Zmień na właściwą ścieżkę do pliku modelu
    codebook_filename = "_10000_words_dict/codebook.npy"  # Zmień na właściwą ścieżkę do pliku codebook

    image_paths = get_file_names(data_test_path)

    feature_detector = cv2.ORB_create()
    
    for image_path in image_paths:
      image_with_path = data_test_path + image_path
      loaded_model, loaded_codebook = load_model_and_codebook(model_filename, codebook_filename)
      print(type (loaded_model))
      predicted_label = predict_object(image_with_path, loaded_model, loaded_codebook, feature_detector)

      if predicted_label is not None:
        predicted_label_str = str(predicted_label).zfill(2)

        print(f"Tested Label: {image_path.split('_', 1)[0]}")
        print(f"Predicted Label: {predicted_label_str}")

        image1 = cv2.imread(image_with_path)
        image2 = cv2.imread(f"DATASET/{predicted_label_str}_00_00.jpg")

        display_images(image1, image2, title1="Test Image", title2="Result Image")
      else:
        print("No features detected in the image.")

if __name__ == "__main__":
    main()
