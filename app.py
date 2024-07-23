from flask import Flask, request, jsonify, render_template
from PIL import Image
import torch
from torchvision import models, transforms
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Define image transformations
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load the trained model
model_path = './resnet50_full_model.pth'  # Update the path

# Use GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load the model with the appropriate device
model = torch.load(model_path, map_location=device)
model.eval()

# Class names
class_names = ['metal', 'glass', 'trash', 'paper', 'cardboard', 'plastic']

# Function to predict the class of an image
def predict_image(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        outputs = model(image)
        _, predicted = torch.max(outputs, 1)
        class_name = class_names[predicted[0]]
    
    return class_name

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_image():
    if 'file' not in request.files:
        return jsonify({"error": "No image file found in the request"}), 400
    
    image_file = request.files['file']
    if image_file.filename == '':
        return jsonify({"error": "No image file selected"}), 400

    filename = secure_filename(image_file.filename)
    image_path = os.path.join('/tmp', filename)
    image_file.save(image_path)
    
    predicted_class = predict_image(image_path)
    
    return jsonify({"predicted_class": predicted_class})

if __name__ == '__main__':
    app.run(debug=True)
