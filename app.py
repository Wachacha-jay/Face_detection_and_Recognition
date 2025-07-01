from flask import Flask, render_template, request, redirect, url_for, flash, send_file
import os
from pathlib import Path
import subprocess
import cv2
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import yt_dlp
import json
import csv

def load_known_faces(known_faces_dir, mtcnn, resnet):
    name_to_embedding = {}
    for person_dir in known_faces_dir.iterdir():
        if person_dir.is_dir():
            embeddings = []
            for img_path in person_dir.glob('*.jpg'):
                img = Image.open(img_path)
                face = mtcnn(img)
                if face is not None:
                    if len(face.shape) == 4:
                        face = face[0]
                    emb = resnet(face.unsqueeze(0))
                    embeddings.append(emb)
            if embeddings:
                avg_emb = torch.mean(torch.stack(embeddings), dim=0)
                name_to_embedding[person_dir.name] = avg_emb
    return name_to_embedding

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Needed for flash messages

# Directories
BASE_DIR = Path(__file__).parent
STATIC_DIR = BASE_DIR / 'static'
RESULTS_DIR = STATIC_DIR / 'results'
KNOWN_FACES_DIR = BASE_DIR / 'known_faces'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Models
mtcnn = MTCNN(device=device, keep_all=True, min_face_size=60, post_process=False)
resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Load known faces (initially)
def get_name_to_embedding():
    return load_known_faces(KNOWN_FACES_DIR, mtcnn, resnet)

name_to_embedding = get_name_to_embedding()

@app.route('/', methods=['GET', 'POST'])
def index():
    global name_to_embedding
    result_imgs = []
    video_details = []
    if request.method == 'POST':
        youtube_urls = request.form['youtube_url'].strip().splitlines()
        if not youtube_urls:
            flash('Please provide at least one YouTube URL.')
            return render_template('index.html', result_imgs=None, video_details=None)
        for idx, youtube_url in enumerate(youtube_urls):
            try:
                video_id = f'{idx}'
                # Download video
                video_path = RESULTS_DIR / f'video_{video_id}.mp4'
                ydl_opts = {
                    'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
                    'outtmpl': str(video_path),
                    'quiet': True,
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([youtube_url])
                # Trim video (first 1 min)
                trimmed_path = RESULTS_DIR / f'trimmed_{video_id}.mp4'
                cmd = [
                    'ffmpeg', '-y', '-ss', '00:00:00', '-i', str(video_path),
                    '-t', '00:01:00', '-c', 'copy', str(trimmed_path)
                ]
                subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                # Extract frames
                cap = cv2.VideoCapture(str(trimmed_path))
                frame_rate = cap.get(cv2.CAP_PROP_FPS)
                interval = int(frame_rate * 0.2)
                count = 0
                saved_frames = []
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if count % interval == 0:
                        frame_path = RESULTS_DIR / f'frame_{video_id}_{count}.jpg'
                        cv2.imwrite(str(frame_path), frame)
                        saved_frames.append(frame_path)
                    count += 1
                cap.release()
                # Pick a sample frame for detection
                if saved_frames:
                    sample_frame = saved_frames[len(saved_frames)//2]
                    img = Image.open(sample_frame)
                    boxes, probs = mtcnn.detect(img)
                    faces = mtcnn(img)
                    fig, ax = plt.subplots()
                    ax.imshow(img)
                    face_details = []
                    if boxes is not None and faces is not None:
                        for i, box in enumerate(boxes):
                            rect = plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], fill=False, color='blue')
                            ax.add_patch(rect)
                            emb = resnet(faces[i].unsqueeze(0))
                            min_dist = float('inf')
                            name = 'Unrecognized'
                            for known_name, known_emb in name_to_embedding.items():
                                dist = torch.dist(emb, known_emb).item()
                                if dist < min_dist:
                                    min_dist = dist
                                    name = known_name if dist < 0.8 else 'Unrecognized'
                            ax.text(box[0], box[1], f'{name} ({min_dist:.2f})', fontsize=8, color='red' if name=='Unrecognized' else 'blue')
                            # Save cropped face
                            cropped_face_path = RESULTS_DIR / f'face_{video_id}_{i}.jpg'
                            face_img = faces[i].permute(1,2,0).int().numpy()
                            face_img = Image.fromarray(face_img.astype('uint8'))
                            face_img.save(cropped_face_path)
                            face_details.append({
                                'name': name,
                                'distance': float(min_dist),
                                'face_img': f'results/face_{video_id}_{i}.jpg',
                                'box': [float(x) for x in box]
                            })
                    plt.axis('off')
                    result_img_path = RESULTS_DIR / f'result_{video_id}.jpg'
                    plt.savefig(result_img_path, bbox_inches='tight')
                    plt.close(fig)
                    # Save details as JSON
                    details_path = RESULTS_DIR / f'result_{video_id}.json'
                    with open(details_path, 'w') as f:
                        json.dump({'faces': face_details, 'result_img': f'results/result_{video_id}.jpg'}, f)
                    result_imgs.append({'img': url_for('static', filename=f'results/result_{video_id}.jpg'), 'video_id': video_id})
                    video_details.append({'video_id': video_id, 'faces': face_details})
            except Exception as e:
                flash(f'Error processing video {youtube_url}: {e}')
    return render_template('index.html', result_imgs=result_imgs, video_details=video_details)

@app.route('/result/<video_id>')
def result_detail(video_id):
    details_path = RESULTS_DIR / f'result_{video_id}.json'
    if not details_path.exists():
        flash('Result not found.')
        return redirect(url_for('index'))
    with open(details_path) as f:
        data = json.load(f)
    # Prepare CSV for download
    csv_path = RESULTS_DIR / f'result_{video_id}.csv'
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Name', 'Distance', 'Face Image'])
        for face in data['faces']:
            writer.writerow([face['name'], face['distance'], face['face_img']])
    return render_template('result_detail.html', result_img=data['result_img'], faces=data['faces'], video_id=video_id)

@app.route('/download/<video_id>/<filename>')
def download_file(video_id, filename):
    file_path = RESULTS_DIR / filename
    if file_path.exists():
        return send_file(file_path, as_attachment=True)
    flash('File not found.')
    return redirect(url_for('result_detail', video_id=video_id))

@app.route('/upload_known_face', methods=['GET', 'POST'])
def upload_known_face():
    global name_to_embedding
    if request.method == 'POST':
        person_name = request.form['person_name'].strip()
        files = request.files.getlist('face_images')
        if not person_name or not files:
            flash('Please provide a name and at least one image.')
            return redirect(request.url)
        person_dir = KNOWN_FACES_DIR / person_name
        person_dir.mkdir(exist_ok=True)
        for file in files:
            if file and file.filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                file.save(str(person_dir / file.filename))
        # Reload known faces so new uploads are recognized immediately
        name_to_embedding = get_name_to_embedding()
        flash(f'Successfully uploaded images for {person_name}.')
        return redirect(url_for('upload_known_face'))
    return render_template('upload_known_face.html')

if __name__ == '__main__':
    app.run(debug=True) 