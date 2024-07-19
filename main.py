import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

# Load the saved model
model = tf.keras.models.load_model('my_model.h5')

# Define the class names
class_names = ['Potato_Early_Blight', 'Potato_Late_Blight', 'Potato_Healthy']  # replace with your actual class names

# Define the prediction function
def predict(image_path):
    # Load and preprocess the image
    img = image.load_img(image_path, target_size=(224, 224))  # use the target size your model expects
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create a batch

    # Make a prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return predicted_class, confidence

# Create the Gradio interface
iface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type='filepath'),
    outputs=[
        gr.Label(num_top_classes=3),  # Shows the top 3 predictions with confidence
        gr.Textbox()  # Shows the confidence of the top prediction
    ],
    title="Image Classification",
    description="Upload an image and the model will classify it into one of three classes."
)

# Launch the interface
iface.launch()
