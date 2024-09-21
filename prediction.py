import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from dotenv import load_dotenv
import os

load_dotenv()

MODEL_DIR = os.getenv('MODEL_PATH')
TEST_DIR = os.getenv('TEST_DIR')

img_size = 224

class_labels = ['harmful', 'non_harmful']

model = load_model(MODEL_DIR)

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(img_size, img_size),
    class_mode=None,
    batch_size=64,
    color_mode='grayscale',
    shuffle=False
)

predictions = model.predict(test_generator)

print(f"predictions: {predictions}")

filenames = test_generator.filenames
for i, filename in enumerate(filenames):
    predicted_class = np.argmax(predictions[i])
    predicted_label = class_labels[predicted_class]
    print(f"Image: {filename}, Predicted class index: {predicted_class}, Predicted label: {predicted_label}")
