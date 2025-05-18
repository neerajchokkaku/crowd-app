from flask import Flask, render_template, request, jsonify
import os
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
from scipy.ndimage import gaussian_filter
from ultralytics import YOLO
import torch
from torchvision import transforms as T
from csrnet_model import CSRNet
from flask import redirect


app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
RESULT_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULT_FOLDER'] = RESULT_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load models
yolo_model = YOLO("yolov8x.pt")
csrnet_model = CSRNet(load_weights=False)
checkpoint = torch.load('csrnet.pth', map_location='cpu')
if 'model_state_dict' in checkpoint:
    checkpoint = checkpoint['model_state_dict']
csrnet_model.load_state_dict(checkpoint)
csrnet_model.eval()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def visualize_crowd_count(image_path):
    # Load and resize image
    image = cv2.imread(image_path)
    resized = cv2.resize(image, (1280, 720))

    # Detect with lower confidence threshold
    results = yolo_model.predict(resized, conf=0.0005)[0]
    
    # Filter people
    people_boxes = []
    for i, cls_id in enumerate(results.boxes.cls.cpu().numpy()):
        if int(cls_id) == 0:  # class 0 = person
            box = results.boxes.xyxy[i].cpu().numpy()
            people_boxes.append(box)
    
    # Grid analysis
    grid_size = 5
    grid_width = resized.shape[1] // grid_size
    grid_height = resized.shape[0] // grid_size
    grid_counts = np.zeros((grid_size, grid_size), dtype=int)

    # Count people in each grid
    for box in people_boxes:
        x1, y1, x2, y2 = map(int, box)
        grid_x = x1 // grid_width
        grid_y = y1 // grid_height
        if grid_x < grid_size and grid_y < grid_size:
            grid_counts[grid_y, grid_x] += 1

    # Draw visualization
    for y in range(grid_size):
        for x in range(grid_size):
            density = grid_counts[y, x]
            color = (0, 255, 0)  # Green
            if density > 15: color = (0, 0, 255)  # Red
            elif density > 5: color = (0, 255, 255)  # Yellow
            
            cv2.rectangle(resized, (x * grid_width, y * grid_height),
                        ((x + 1) * grid_width, (y + 1) * grid_height), color, 2)

    # Draw individual detections
    for box in people_boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(resized, "Person", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Save visualization
    filename = 'crowd_vis_' + os.path.basename(image_path)
    result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
    cv2.imwrite(result_path, resized)
    
    return len(people_boxes), result_path

def generate_heatmap(image_path):
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    img = Image.open(image_path).convert('RGB')
    img = transform(img).unsqueeze(0)

    with torch.no_grad():
        density_map = csrnet_model(img)

    density_map = density_map.squeeze().cpu().numpy()
    density_map = gaussian_filter(density_map, sigma=1)
    density_map -= density_map.min()
    if density_map.max() > 0:
        density_map /= density_map.max()

    density_map_resized = cv2.resize(density_map, 
                                   (original_img.shape[1], original_img.shape[0]), 
                                   interpolation=cv2.INTER_CUBIC)
    heatmap = (density_map_resized * 255).astype(np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    output = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

    return output

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(upload_path)
            return render_template('index.html', 
                                original_image=upload_path,
                                filename=filename)
    
    return render_template('index.html')

@app.route('/process_image', methods=['POST'])
def process_image():
    data = request.json
    image_path = data.get('image_path')
    action = data.get('action')
    
    if not os.path.exists(image_path):
        return jsonify({'error': 'Image not found'}), 404
    
    if action == 'heatmap':
        heatmap = generate_heatmap(image_path)
        filename = 'heatmap_' + os.path.basename(image_path)
        result_path = os.path.join(app.config['RESULT_FOLDER'], filename)
        cv2.imwrite(result_path, cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR))
        return jsonify({'result_path': result_path})
    
    elif action == 'count':
        count, vis_path = visualize_crowd_count(image_path)
        return jsonify({
            'count': count,
            'vis_path': vis_path
        })
    
    return jsonify({'error': 'Invalid action'}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)