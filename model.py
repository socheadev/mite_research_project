import pandas as pd
import numpy as np
import requests
from PIL import Image
from io import BytesIO
from sklearn.model_selection import train_test_split
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load CSV file containing image URLs and labels
csv_file_path = 'image_urls.csv'
df = pd.read_csv(csv_file_path)


# Helper function to download and preprocess images, handling Google Drive links
def download_and_preprocess_image(url, target_size=(224, 224)):
    try:
        # Check if the URL is a Google Drive link
        if "drive.google.com" in url:
            file_id = url.split('id=')[-1]
            url = f"https://drive.google.com/uc?export=download&id={file_id}"

        response = requests.get(url, timeout=10)
        response.raise_for_status()  # Raise error for bad HTTP responses
        image = Image.open(BytesIO(response.content))
        image = image.resize(target_size)
        image = np.array(image)  # Convert to numpy array

        if len(image.shape) == 2:  # If grayscale, convert to RGB
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[-1] == 4:  # If RGBA, convert to RGB
            image = image[..., :3]

        return image / 255.0  # Rescale pixel values to [0, 1]
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return None


# Preprocess images and labels, filtering out failed downloads
images = []
labels = []

for index, row in df.iterrows():
    img = download_and_preprocess_image(row['image_url'])
    if img is not None:
        images.append(img)
        labels.append(row['label'])

# Convert lists to numpy arrays and ensure they have valid data
if len(images) > 0 and len(labels) > 0:
    images = np.array(images)
    labels = np.array(labels, dtype=int)
else:
    raise ValueError("No valid images or labels were found")

# Ensure images are consistent
if len(images.shape) != 4 or images.shape[1:] != (224, 224, 3):
    raise ValueError(f"Unexpected image shape: {images.shape}. Expected (224, 224, 3)")

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # Binary classification (harmful/non_harmful)
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val))

# Save the model
model.save('cnn_model_detector.h5')

# Show the model summary
model.summary()
