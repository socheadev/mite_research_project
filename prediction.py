import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

# Load the saved model
model = load_model('data/model/cnn_model_1.keras')


def preprocess_image(img_path, target_size=(224, 224)):
    # Load the image with the specified target size
    img = image.load_img(img_path, target_size=target_size)

    # Convert the image to an array
    img_array = image.img_to_array(img)

    # Expand dimensions to match the model input (batch_size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)

    # Normalize pixel values to [0, 1]
    img_array /= 255.0

    return img_array


def make_prediction(model, img_path):
    # Preprocess the image
    img_array = preprocess_image(img_path)

    # Make the prediction
    prediction = model.predict(img_array)

    # Get the index of the class with the highest predicted probability
    predicted_class = np.argmax(prediction, axis=1)

    return predicted_class


def display_prediction_with_labels(img_path, model, class_labels):
    # Display the image
    img = image.load_img(img_path)
    plt.imshow(img)
    plt.axis('off')  # Hide axes
    plt.show()

    # Make a prediction
    predicted_class = make_prediction(model, img_path)

    # Map the predicted index to the class label
    predicted_label = class_labels[predicted_class[0]]

    # Print the predicted label
    print(f"Predicted Label: {predicted_label}")


# Define class labels
class_labels = ['safe', 'unsafe']

# Path to the image you want to predict
img_path = 'data/testing/test_image_1.jpg'

# Display the image and its prediction
display_prediction_with_labels(img_path, model, class_labels)
