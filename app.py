from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from PIL import Image
from io import BytesIO
import torch
import torch.nn as nn
from torchvision import transforms
from model import CustomResNet  

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

classes = ['papel', 'plastico', 'vidrio']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def load_model():
    model = CustomResNet()
    model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

def predict_image(img, model):
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        outputs = model(img_tensor)
    _, predicted = torch.max(outputs, 1)
    class_idx = predicted.item()
    return classes[class_idx]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file and allowed_file(file.filename):
            img = Image.open(BytesIO(file.read())).convert('RGB')
            img = img.resize((224, 224))  
            model = load_model()
            prediction = predict_image(img, model)
            filename = secure_filename(file.filename)
            img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return render_template('result.html', prediction=prediction, filename=filename)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
