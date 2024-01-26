import os
import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import csv
import joblib
import random


def load_and_split_dataset(folder_path, test_ratio=0.2):
    image_paths = [os.path.join(folder_path, file) for file in os.listdir(folder_path) if file.endswith(".jpg")]
    random.shuffle(image_paths)

    num_test = int(len(image_paths) * test_ratio)
    test_image_paths = image_paths[:num_test]
    train_image_paths = image_paths[num_test:]

    return train_image_paths, test_image_paths

# Example function to extract label from image path (modify as needed)
def get_label_from_path(image_path):
    # Assuming the label is the first part of the filename before the first underscore
    filename = os.path.basename(image_path)
    label = filename.split('_')[0]
    
    # Convert the label to an integer
    label = int(label)

    return label

def extract_features(image_paths, feature_detector):
    features = []
    for image_path in tqdm(image_paths, desc="Extracting features", unit="image"):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, des = feature_detector.detectAndCompute(image, None)
        if des is not None:
            features.extend(des)
    return np.array(features)

def create_codebook(features, num_dict_features):
    if len(features) == 0:
        # Handle the case when no features are available
        print("Warning: No features found. Returning default codebook.")
        return np.zeros((1, num_dict_features))

    # Ensure features is a 2D array
    features = features.reshape(-1, 1) if len(features.shape) == 1 else features

    kmeans = KMeans(n_clusters=num_dict_features, n_init=10)
    with tqdm(desc="Clustering", total=10, unit="iteration") as pbar:
        kmeans.fit(features)
        pbar.update(1)

    return kmeans.cluster_centers_

def image_representation(image_paths, codebook, feature_detector):
    representations = []
    labels = []
    for image_path in tqdm(image_paths, desc="Generating representations", unit="image"):
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        _, des = feature_detector.detectAndCompute(image, None)
        if des is not None and len(des) > 0:  # Check if features are detected
            histogram = np.zeros(len(codebook))
            for feature in des:
                idx = np.argmin(np.linalg.norm(codebook - feature, axis=1))
                histogram[idx] += 1
            representations.append(histogram)
            labels.append(get_label_from_path(image_path))

    return np.array(representations), np.array(labels)

def train_model(train_image_paths, test_image_paths, feature_detector, num_dict_features):
    # Extract features from training images
    train_features = extract_features(train_image_paths, feature_detector)

    # Create codebook using KMeans clustering
    codebook = create_codebook(train_features, num_dict_features)

    # Represent training and testing images using BoVW
    train_data, train_labels = image_representation(train_image_paths, codebook, feature_detector)
    test_data, test_labels = image_representation(test_image_paths, codebook, feature_detector)

    print(f"Number of training labels: {len(train_labels)}")
    print(f"Number of test labels: {len(test_labels)}")

    print(f"Number of training images: {len(train_data)}")
    print(f"Number of testing images: {len(test_data)}")

    # Train an SVM model
    svm_model = SVC()
    svm_model.fit(train_data, train_labels)


    # Make predictions on the test set
    predictions = svm_model.predict(test_data)

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)

    return svm_model, accuracy, codebook

dataset_path = "DATASET/"
train_image_paths, test_image_paths = load_and_split_dataset(dataset_path, test_ratio=0.2)

print(f"Number of training images: {len(train_image_paths)}")
print(f"Number of testing images: {len(test_image_paths)}")

feature_detector = cv2.ORB_create()
num_dict_features = 10000  # Parameter num_dict_features

train_labels = [get_label_from_path(path) for path in train_image_paths]
print(f"Unique labels in training set: {set(train_labels)}")

trained_model, accuracy, codebook = train_model(train_image_paths, test_image_paths, feature_detector, num_dict_features)

print(f"Model Accuracy: {accuracy}")

def evaluate_model_on_subset(svm_model, image_paths, codebook, feature_detector):
    test_data, test_labels = image_representation(image_paths, codebook, feature_detector)

    print(f"Number of training labels: {len(train_labels)}")
    print(f"Number of test labels: {len(test_labels)}")

    # Make predictions on the test set
    predictions = svm_model.predict(test_data)

    # Calculate accuracy
    accuracy = accuracy_score(test_labels, predictions)

    return svm_model, accuracy


autumn_image_paths, winter_image_paths, spring_image_paths = [], [], []
autumn_day_image_paths, autumn_night_image_paths = [], []
winter_day_image_paths, winter_night_image_paths = [], []
spring_day_image_paths, spring_night_image_paths = [], []

image_paths, _ = load_and_split_dataset(dataset_path, 0)

for image_path in image_paths:
    # Assuming your images are named with the specified convention: <label>_<season_and_time_of_day>_<numerical_index>.jpg
    label, season_and_time_of_day, _ = os.path.splitext(os.path.basename(image_path))[0].split('_')

    if season_and_time_of_day == "00":
        autumn_day_image_paths.append(image_path)
        autumn_image_paths.append(image_path)
    elif season_and_time_of_day == "01":
        autumn_night_image_paths.append(image_path)
        autumn_image_paths.append(image_path)
    elif season_and_time_of_day == "02":
        winter_day_image_paths.append(image_path)
        winter_image_paths.append(image_path)
    elif season_and_time_of_day == "03":
        winter_night_image_paths.append(image_path)
        winter_image_paths.append(image_path)
    elif season_and_time_of_day == "04":
        spring_day_image_paths.append(image_path)
        spring_image_paths.append(image_path)
    elif season_and_time_of_day == "05":
        spring_night_image_paths.append(image_path)
        spring_image_paths.append(image_path)

# Evaluate on subsets
autumn_accuracy = evaluate_model_on_subset(trained_model, autumn_image_paths, codebook, feature_detector)
winter_accuracy = evaluate_model_on_subset(trained_model, winter_image_paths, codebook, feature_detector)
spring_accuracy = evaluate_model_on_subset(trained_model, spring_image_paths, codebook, feature_detector)

autumn_day_accuracy = evaluate_model_on_subset(trained_model, autumn_day_image_paths, codebook, feature_detector)
autumn_night_accuracy = evaluate_model_on_subset(trained_model, autumn_night_image_paths, codebook, feature_detector)
winter_day_accuracy = evaluate_model_on_subset(trained_model, winter_day_image_paths, codebook, feature_detector)
winter_night_accuracy = evaluate_model_on_subset(trained_model, winter_night_image_paths, codebook, feature_detector)
spring_day_accuracy = evaluate_model_on_subset(trained_model, spring_day_image_paths, codebook, feature_detector)
spring_night_accuracy = evaluate_model_on_subset(trained_model, spring_night_image_paths, codebook, feature_detector)

# Print results
print(f"Autumn Accuracy: {autumn_accuracy}")
print(f"Winter Accuracy: {winter_accuracy}")
print(f"Spring Accuracy: {spring_accuracy}")

print(f"Autumn Day Accuracy: {autumn_day_accuracy}")
print(f"Autumn Night Accuracy: {autumn_night_accuracy}")
print(f"Winter Day Accuracy: {winter_day_accuracy}")
print(f"Winter Night Accuracy: {winter_night_accuracy}")
print(f"Spring Day Accuracy: {spring_day_accuracy}")
print(f"Spring Night Accuracy: {spring_night_accuracy}")

data = [
    ("Overall", accuracy),
    ("Autumn", autumn_accuracy),
    ("Winter", winter_accuracy),
    ("Spring", spring_accuracy),
    ("Autumn Day", autumn_day_accuracy),
    ("Autumn Night", autumn_night_accuracy),
    ("Winter Day", winter_day_accuracy),
    ("Winter Night", winter_night_accuracy),
    ("Spring Day", spring_day_accuracy),
    ("Spring Night", spring_night_accuracy),
]

def create_directory(directory_path):
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)

def save_codebook(codebook, file_path):
    np.save(file_path, codebook)

folder_path = '_10000_words_dict'

create_directory(folder_path)

csv_file_path = f"{folder_path}/accuracy_data.csv"

# Write data to the CSV file
with open(csv_file_path, mode='w', newline='') as file:
    writer = csv.writer(file)

    # Write headers
    writer.writerow(["test set", "accuracy"])

    # Write data
    writer.writerows(data)

print(f"Data has been saved to {csv_file_path}")

model_filename = f"{folder_path}/test_model.joblib"

# Save the trained model to a file
joblib.dump(trained_model, model_filename)

print(f"Model saved to {model_filename}")

codebook_fname = f'{folder_path}/codebook.npy'
save_codebook(codebook, codebook_fname)

print(f'Codebook saved to {codebook_fname}')