import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import os
import pandas as pd
import random
import requests
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
from transformers import ElectraModel, ElectraTokenizer, DistilBertModel, DistilBertTokenizer
import torchvision.models as tv_models
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')


def download_model(url, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    if not os.path.exists(output_path):
        print(f"Downloading model from {url}...")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"Model downloaded to {output_path}")
    else:
        print(f"Model already exists at {output_path}")


MODEL_PTH_URL = "https://huggingface.co/Sadrib-111/mpeqs-model/resolve/main/mpeqs_model.pth"
MODEL_PKL_URL = "https://huggingface.co/Sadrib-111/mpeqs-model/resolve/main/mpeqs_model.pkl"


download_model(MODEL_PTH_URL, 'notebook/saved_models/mpeqs_model.pth')
download_model(MODEL_PKL_URL, 'notebook/saved_models/mpeqs_model.pkl')


print("="*60)
print("Loading MPEQS Cell 1 Model...")
print("="*60)

try:
    model_state = torch.load('notebook/saved_models/mpeqs_model.pt', map_location='cpu')
    print("Loaded from mpeqs_model.pt")
except:
    with open('notebook/saved_models/mpeqs_model.pkl', 'rb') as f:
        artifacts = pickle.load(f)
    model_state = artifacts['model_state_dict']
    class_names = artifacts['model_config']['class_names']
    print("Loaded from mpeqs_model.pkl")

class_names = ['Benign', 'Phishing-only', 'Quishing-only', 'Smishing-only', 'Mixed Attack']
print(f"Class names: {class_names}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class CrossAttention(nn.Module):
    def __init__(self, dim=256, heads=4):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.W_q = nn.Linear(dim, dim, bias=False)
        self.W_k = nn.Linear(dim, dim, bias=False)
        self.W_v = nn.Linear(dim, dim, bias=False)
        self.W_o = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, e, q, s):
        B = e.shape[0]
        V = torch.stack([e, q, s], dim=1)
        Q = self.W_q(e.unsqueeze(1)).view(B, 1, self.heads, -1).transpose(1, 2)
        K = self.W_k(V).view(B, 3, self.heads, -1).transpose(1, 2)
        V2 = self.W_v(V).view(B, 3, self.heads, -1).transpose(1, 2)
        attn = F.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1)
        attn = self.dropout(attn)
        out = (attn @ V2).transpose(1, 2).reshape(B, 1, -1)
        out = self.W_o(out)
        return self.norm(e.unsqueeze(1) + out).squeeze(1)

class MPEQSModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.electra = ElectraModel.from_pretrained('google/electra-small-discriminator')
        self.mobilenet = tv_models.mobilenet_v2(pretrained=True)
        self.mobilenet.classifier = nn.Identity()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.e_proj = nn.Sequential(nn.Linear(256, 256), nn.BatchNorm1d(256))
        self.q_proj = nn.Sequential(nn.Linear(1280, 256), nn.BatchNorm1d(256))
        self.s_proj = nn.Sequential(nn.Linear(768, 256), nn.BatchNorm1d(256))
        self.fusion = CrossAttention()
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 5)
        )
        for p in self.electra.parameters(): p.requires_grad = False
        for p in self.mobilenet.parameters(): p.requires_grad = False
        for p in self.distilbert.parameters(): p.requires_grad = False

    def forward(self, e_ids, e_mask, q_img, s_ids, s_mask):
        with torch.no_grad():
            e = self.electra(e_ids, e_mask).last_hidden_state[:, 0, :]
            q = self.mobilenet(q_img)
            s = self.distilbert(s_ids, s_mask).last_hidden_state[:, 0, :]
        e = self.e_proj(e)
        q = self.q_proj(q)
        s = self.s_proj(s)
        fused = self.fusion(e, q, s)
        return self.classifier(fused)

model = MPEQSModel()
model.load_state_dict(model_state)
model.to(device)
model.eval()
print("Cell 1 model loaded successfully")

email_tokenizer = ElectraTokenizer.from_pretrained('google/electra-small-discriminator')
sms_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
print("Tokenizers loaded")

class EmailClassifier:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        self.is_trained = False
        self._extract_and_train()
    
    def extract_email_features(self, email_text):
        if not email_text or len(email_text) < 20:
            return np.zeros(256)
        
        inputs = email_tokenizer(email_text, max_length=256, padding='max_length', 
                                  truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model.electra(input_ids, attention_mask)
            features = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        return features
    
    def _extract_and_train(self):
        print("\n" + "="*60)
        print("Training EMAIL Classifier (ELECTRA + Random Forest)")
        print("="*60)
        
        human_legit = pd.read_csv('notebook/datasets/email_dataset/legit.csv').head(2000)
        llm_legit = pd.read_csv('notebook/datasets/email_dataset/llm_legit.csv').head(1000)
        
        benign_features = []
        
        print("Extracting features from BENIGN emails...")
        for idx, row in human_legit.iterrows():
            if idx % 500 == 0:
                print(f"Processed {idx}/{len(human_legit)}")
            text = str(row['body'])[:512]
            if len(text) > 20:
                features = self.extract_email_features(text)
                benign_features.append(features)
        
        for idx, row in llm_legit.iterrows():
            text = str(row['text'])[:512] if 'text' in row else str(row.iloc[0])[:512]
            if len(text) > 20:
                features = self.extract_email_features(text)
                benign_features.append(features)
        
        print(f"Extracted {len(benign_features)} BENIGN email features")
        
        print("Creating negative samples...")
        negative_features = []
        for feat in benign_features[:len(benign_features)//2]:
            noise = np.random.normal(0, 0.1, feat.shape)
            negative_features.append(feat + noise)
        
        benign_features = np.array(benign_features)
        negative_features = np.array(negative_features)
        
        X = np.vstack([benign_features, negative_features])
        y = np.array([1] * len(benign_features) + [0] * len(negative_features))
        
        print(f"Total: {len(X)} samples (Benign: {sum(y==1)}, Not Benign: {sum(y==0)})")
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.classifier.fit(X_train, y_train)
        accuracy = self.classifier.score(X_val, y_val)
        
        self.is_trained = True
        print(f"Email classifier trained! Validation accuracy: {accuracy:.2%}")
    
    def predict(self, text):
        if not text or len(text) < 20 or not self.is_trained:
            return 0.5, False
        
        features = self.extract_email_features(text).reshape(1, -1)
        prob = self.classifier.predict_proba(features)[0]
        is_benign = prob[1] > 0.6
        return prob[1], is_benign


class SMSClassifier:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        self.is_trained = False
        self._extract_and_train()
    
    def extract_sms_features(self, sms_text):
        if not sms_text or len(sms_text) < 10:
            return np.zeros(256)
        
        inputs = sms_tokenizer(sms_text[:128], max_length=128, padding='max_length',
                                truncation=True, return_tensors='pt')
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        with torch.no_grad():
            outputs = model.distilbert(input_ids, attention_mask)
            features = outputs.last_hidden_state[:, 0, :].cpu().numpy()[0]
        
        return features
    
    def _extract_and_train(self):
        print("\n" + "="*60)
        print("Training SMS CLASSIFIER (DistilBERT + Random Forest)")
        print("="*60)
        
        smishing_df = pd.read_csv('notebook/datasets/smishing_dataset/Dataset_10191.csv')
        
        benign_sms = smishing_df[smishing_df['LABEL'] == 'ham']['TEXT'].astype(str).tolist()[:1000]
        
        print(f"Extracting features from {len(benign_sms)} BENIGN SMS...")
        benign_features = []
        for i, text in enumerate(benign_sms):
            if i % 200 == 0:
                print(f"Processed {i}/{len(benign_sms)}")
            if len(text) > 10:
                features = self.extract_sms_features(text)
                benign_features.append(features)
        
        print(f"Extracted {len(benign_features)} BENIGN SMS features")
        
        print("Creating negative samples...")
        negative_features = []
        for feat in benign_features[:len(benign_features)//2]:
            noise = np.random.normal(0, 0.1, feat.shape)
            negative_features.append(feat + noise)
        
        benign_features = np.array(benign_features)
        negative_features = np.array(negative_features)
        
        X = np.vstack([benign_features, negative_features])
        y = np.array([1] * len(benign_features) + [0] * len(negative_features))
        
        print(f"Total: {len(X)} samples (Benign: {sum(y==1)}, Not Benign: {sum(y==0)})")
        
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        self.classifier.fit(X_train, y_train)
        accuracy = self.classifier.score(X_val, y_val)
        
        self.is_trained = True
        print(f"SMS classifier trained! Validation accuracy: {accuracy:.2%}")
    
    def predict(self, text):
        if not text or len(text) < 10 or not self.is_trained:
            return 0.5, False
        
        features = self.extract_sms_features(text).reshape(1, -1)
        prob = self.classifier.predict_proba(features)[0]
        is_benign = prob[1] > 0.6
        return prob[1], is_benign


class QRImageClassifier:
    def __init__(self):
        self.classifier = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
        self.is_trained = False
        self._extract_and_train()
    
    def _extract_and_train(self):
        print("\n" + "="*60)
        print("Training QR Classifier (MobileNet + Random Forest)")
        print("="*60)
        
        qr_features = []
        qr_labels = []
        
        benign_folder = 'notebook/datasets/quishing_dataset/benign_qr'
        if os.path.exists(benign_folder):
            for file in os.listdir(benign_folder):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(benign_folder, file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (224, 224)) / 255.0
                        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            features = model.mobilenet(img_tensor).cpu().numpy()[0]
                        
                        qr_features.append(features)
                        qr_labels.append(0)
            print(f"Loaded {qr_labels.count(0)} Benign QR images")
        
        malicious_folder = 'notebook/datasets/quishing_dataset/malicious_qr'
        if os.path.exists(malicious_folder):
            for file in os.listdir(malicious_folder):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(malicious_folder, file)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (224, 224)) / 255.0
                        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            features = model.mobilenet(img_tensor).cpu().numpy()[0]
                        
                        qr_features.append(features)
                        qr_labels.append(1)
            print(f"Loaded {qr_labels.count(1)} Malicious QR images")
        
        if len(qr_features) < 5:
            print("Not enough QR images for training!")
            self.is_trained = False
            return
        
        X = np.array(qr_features)
        y = np.array(qr_labels)
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        self.classifier.fit(X_train, y_train)
        accuracy = self.classifier.score(X_test, y_test)
        
        print(f"\nQR Classifier trained!")
        print(f"Total samples: {len(qr_features)}")
        print(f"Validation Accuracy: {accuracy:.2%}")
        self.is_trained = True
    
    def predict(self, qr_tensor):
        if not self.is_trained or qr_tensor is None:
            return 0.5, None
        
        with torch.no_grad():
            features = model.mobilenet(qr_tensor).cpu().numpy()[0]
        
        prob = self.classifier.predict_proba([features])[0]
        is_malicious = prob[1] > 0.5
        confidence = prob[1] if is_malicious else prob[0]
        
        return confidence, is_malicious


print("\n" + "="*60)
print("Initializing Classifiers...")
print("="*60)

email_classifier = EmailClassifier()
sms_classifier = SMSClassifier()
qr_classifier = QRImageClassifier()


app = Flask(__name__)
CORS(app)

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MPEQS | Advanced Phishing Detection</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { font-family: 'Inter', sans-serif; background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%); min-height: 100vh; color: #fff; }
        .container { max-width: 1400px; margin: 0 auto; padding: 40px 20px; }
        .header { text-align: center; margin-bottom: 50px; }
        .logo { font-size: 56px; margin-bottom: 16px; animation: pulse 2s infinite; }
        @keyframes pulse { 0%, 100% { transform: scale(1); } 50% { transform: scale(1.05); } }
        h1 { font-size: 48px; font-weight: 800; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 12px; }
        .tagline { font-size: 18px; color: rgba(255,255,255,0.7); margin-bottom: 24px; }
        .features { display: flex; justify-content: center; gap: 16px; flex-wrap: wrap; margin-top: 20px; }
        .feature-badge { background: rgba(255,255,255,0.1); backdrop-filter: blur(10px); padding: 8px 20px; border-radius: 40px; font-size: 14px; font-weight: 500; border: 1px solid rgba(255,255,255,0.2); }
        .row { display: flex; gap: 25px; flex-wrap: wrap; justify-content: center; margin-bottom: 35px; }
        .col { flex: 1; min-width: 320px; }
        .card { background: rgba(255,255,255,0.08); backdrop-filter: blur(12px); border-radius: 28px; padding: 28px; border: 1px solid rgba(255,255,255,0.12); transition: all 0.3s; }
        .card:hover { transform: translateY(-8px); background: rgba(255,255,255,0.12); }
        .card-header { display: flex; align-items: center; gap: 12px; margin-bottom: 20px; padding-bottom: 15px; border-bottom: 1px solid rgba(255,255,255,0.1); }
        .card-icon { font-size: 32px; }
        .card-title { font-size: 22px; font-weight: 600; }
        textarea { width: 100%; padding: 16px; border-radius: 20px; border: 1px solid rgba(255,255,255,0.2); background: rgba(0,0,0,0.3); color: white; font-family: monospace; font-size: 13px; resize: vertical; }
        textarea:focus { outline: none; border-color: #667eea; }
        input[type="file"] { width: 100%; padding: 14px; border: 2px dashed rgba(255,255,255,0.3); border-radius: 20px; background: rgba(0,0,0,0.2); color: white; cursor: pointer; }
        .qr-preview { margin-top: 20px; text-align: center; }
        .qr-preview img { max-width: 140px; border-radius: 20px; }
        .analyze-btn { display: block; width: 300px; margin: 30px auto; padding: 18px 40px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border: none; border-radius: 60px; color: white; font-size: 18px; font-weight: 700; cursor: pointer; transition: all 0.3s; }
        .analyze-btn:hover { transform: scale(1.03); }
        .loading { text-align: center; padding: 40px; display: none; }
        .spinner { width: 60px; height: 60px; border: 3px solid rgba(255,255,255,0.2); border-top-color: #667eea; border-radius: 50%; animation: spin 0.8s linear infinite; margin: 0 auto 20px; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .result { background: rgba(255,255,255,0.1); backdrop-filter: blur(12px); border-radius: 28px; padding: 35px; margin-top: 35px; text-align: center; display: none; animation: fadeIn 0.5s ease; }
        @keyframes fadeIn { from { opacity: 0; transform: translateY(20px); } to { opacity: 1; transform: translateY(0); } }
        .result-icon { font-size: 64px; margin-bottom: 20px; }
        .result-title { font-size: 36px; font-weight: 800; margin-bottom: 15px; }
        .result-title.benign { color: #4ade80; }
        .result-title.phishing { color: #f87171; }
        .result-confidence { font-size: 16px; color: rgba(255,255,255,0.7); margin-bottom: 20px; }
        .result-badges { display: flex; justify-content: center; gap: 12px; flex-wrap: wrap; margin-bottom: 25px; }
        .result-badge { background: rgba(102,126,234,0.3); padding: 6px 16px; border-radius: 30px; font-size: 12px; }
        .result-message { background: rgba(0,0,0,0.3); border-radius: 20px; padding: 20px; font-size: 16px; line-height: 1.6; }
        .footer { text-align: center; margin-top: 50px; padding-top: 25px; border-top: 1px solid rgba(255,255,255,0.1); font-size: 12px; color: rgba(255,255,255,0.5); }
        @media (max-width: 768px) { h1 { font-size: 32px; } .result-title { font-size: 28px; } }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="logo">🛡️</div>
            <h1>MPEQS</h1>
            <div class="tagline">Advanced Multimodal Phishing Detection System</div>
            <div class="features">
                <span class="feature-badge">📧 Email Scanner</span>
                <span class="feature-badge">📱 QR Scanner</span>
                <span class="feature-badge">📲 SMS Scanner</span>
                <span class="feature-badge">🤖 AI-Powered</span>
                <span class="feature-badge">⚡ Real-time</span>
            </div>
        </div>
        
        <div class="row">
            <div class="col">
                <div class="card">
                    <div class="card-header">
                        <span class="card-icon">📧</span>
                        <span class="card-title">Email Analysis</span>
                    </div>
                    <textarea id="email_text" rows="6" placeholder="Paste email content here..."></textarea>
                </div>
            </div>
            
            <div class="col">
                <div class="card">
                    <div class="card-header">
                        <span class="card-icon">📱</span>
                        <span class="card-title">QR Analysis</span>
                    </div>
                    <input type="file" id="qr_image" accept="image/*">
                    <div id="qr_preview" class="qr-preview"></div>
                </div>
            </div>
            
            <div class="col">
                <div class="card">
                    <div class="card-header">
                        <span class="card-icon">📲</span>
                        <span class="card-title">SMS Analysis</span>
                    </div>
                    <textarea id="sms_text" rows="6" placeholder="Paste SMS content here..."></textarea>
                </div>
            </div>
        </div>
        
        <button class="analyze-btn" onclick="predict()">🔍 Scan for Threats</button>
        
        <div class="loading" id="loading">
            <div class="spinner"></div>
            <div>🤖 AI is analyzing your content...</div>
        </div>
        
        <div id="result" class="result"></div>
        
        <div class="footer">
            <p>MPEQS v2.0 | ELECTRA + MobileNet + DistilBERT | Cross-Attention Fusion</p>
        </div>
    </div>
    
    <script>
        document.getElementById('qr_image').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                const reader = new FileReader();
                reader.onload = function(event) {
                    document.getElementById('qr_preview').innerHTML = `<img src="${event.target.result}" alt="QR Preview">`;
                };
                reader.readAsDataURL(file);
            }
        });
        
        async function predict() {
            const email_text = document.getElementById('email_text').value;
            const sms_text = document.getElementById('sms_text').value;
            const qr_file = document.getElementById('qr_image').files[0];
            
            if (!email_text && !sms_text && !qr_file) {
                alert('Please provide at least one input');
                return;
            }
            
            document.getElementById('loading').style.display = 'block';
            document.getElementById('result').style.display = 'none';
            
            const formData = new FormData();
            formData.append('email_text', email_text);
            formData.append('sms_text', sms_text);
            if (qr_file) formData.append('qr_image', qr_file);
            
            try {
                const response = await fetch('/predict', { method: 'POST', body: formData });
                const data = await response.json();
                displayResult(data);
            } catch (error) {
                alert('Error: ' + error);
            }
            document.getElementById('loading').style.display = 'none';
        }
        
        function displayResult(data) {
            const resultDiv = document.getElementById('result');
            const class_name = data.class;
            const corrected_types = data.corrected_types || [];
            const confidence = (Math.max(...data.probabilities) * 100).toFixed(1);
            
            const isBenign = class_name === 'Benign';
            let icon = '', message = '';
            
            if (class_name === 'Benign') {
                icon = '✅';
                message = 'This content appears to be LEGITIMATE. No threat detected.';
            } else if (class_name === 'Phishing-only') {
                icon = '⚠️';
                message = 'PHISHING detected! Do not click any links or reply. Report immediately.';
            } else if (class_name === 'Quishing-only') {
                icon = '⚠️';
                message = 'MALICIOUS QR CODE detected! Do NOT scan this QR code.';
            } else if (class_name === 'Smishing-only') {
                icon = '⚠️';
                message = 'SMISHING detected! Do not click links. Block sender.';
            } else {
                icon = '🚨';
                message = 'MULTI-CHANNEL ATTACK detected! Report immediately.';
            }
            
            let badges = '';
            if (corrected_types.includes('email')) badges += '<span class="result-badge">📧 Email Corrected</span>';
            if (corrected_types.includes('sms')) badges += '<span class="result-badge">📲 SMS Corrected</span>';
            if (corrected_types.includes('qr_malicious')) badges += '<span class="result-badge">📱 Malicious QR</span>';
            if (corrected_types.includes('qr_benign')) badges += '<span class="result-badge">📱 Benign QR</span>';
            
            resultDiv.innerHTML = `
                <div class="result-icon">${icon}</div>
                <div class="result-title ${isBenign ? 'benign' : 'phishing'}">${class_name}</div>
                <div class="result-confidence">Confidence: ${confidence}%</div>
                <div class="result-badges">${badges}</div>
                <div class="result-message"><strong>Analysis Result</strong><br><br>${message}</div>
            `;
            resultDiv.style.display = 'block';
            resultDiv.scrollIntoView({ behavior: 'smooth' });
        }
    </script>
</body>
</html>
'''

def preprocess_qr_image(image_bytes, target_size=(224, 224)):
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        img = np.zeros((224, 224, 3), dtype=np.uint8)
    img = cv2.resize(img, target_size)
    img = img / 255.0
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img.unsqueeze(0)

def create_dummy_qr():
    return torch.zeros(1, 3, 224, 224)

@app.route('/')
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        email_text = request.form.get('email_text', '')
        sms_text = request.form.get('sms_text', '')
        qr_file = request.files.get('qr_image')
        
        if not email_text or email_text.strip() == '':
            email_text = "No email provided"
        if not sms_text or sms_text.strip() == '':
            sms_text = "No SMS provided"
        
        email_enc = email_tokenizer(email_text, max_length=256, padding='max_length', truncation=True, return_tensors='pt')
        sms_enc = sms_tokenizer(sms_text, max_length=128, padding='max_length', truncation=True, return_tensors='pt')
        
        qr_tensor = None
        if qr_file and qr_file.filename:
            qr_bytes = qr_file.read()
            qr_tensor = preprocess_qr_image(qr_bytes)
            qr_images = qr_tensor
        else:
            qr_images = create_dummy_qr()
        
        email_input_ids = email_enc['input_ids'].to(device)
        email_attention_mask = email_enc['attention_mask'].to(device)
        sms_input_ids = sms_enc['input_ids'].to(device)
        sms_attention_mask = sms_enc['attention_mask'].to(device)
        
        with torch.no_grad():
            logits = model(email_input_ids, email_attention_mask, qr_images, sms_input_ids, sms_attention_mask)
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()[0]
            predicted_class = int(np.argmax(probabilities))
        
        final_class = class_names[predicted_class]
        corrected_types = []
        
        if qr_tensor is not None and qr_file and qr_file.filename:
            confidence, is_malicious = qr_classifier.predict(qr_tensor)
            
            if is_malicious is not None:
                if is_malicious and confidence > 0.6:
                    final_class = 'Quishing-only'
                    corrected_types.append('qr_malicious')
                    print(f"QR: Malicious (conf: {confidence:.2%}) → Quishing-only")
                elif not is_malicious and confidence > 0.6:
                    final_class = 'Benign'
                    corrected_types.append('qr_benign')
                    print(f"QR: Benign (conf: {confidence:.2%}) → Benign")
        
        if email_text and email_text != "No email provided" and final_class != 'Benign':
            legit_score, is_legitimate = email_classifier.predict(email_text)
            if is_legitimate and legit_score > 0.65:
                final_class = 'Benign'
                corrected_types.append('email')
                print(f"Email corrected (score: {legit_score:.2f})")
        
        if sms_text and sms_text != "No SMS provided" and final_class != 'Benign':
            legit_score, is_legitimate = sms_classifier.predict(sms_text)
            if is_legitimate and legit_score > 0.65:
                final_class = 'Benign'
                corrected_types.append('sms')
                print(f"SMS corrected (score: {legit_score:.2f})")
        
        return jsonify({
            'class': final_class,
            'probabilities': probabilities.tolist(),
            'corrected_types': corrected_types,
            'success': True
        })
        
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("\n" + "="*60)
    print("MPEQS: Advanced Phishing Detection System")
    print("="*60)
    print(f"Cell 1 Model: {len(class_names)} classes")
    print("Email classifier: ACTIVE")
    print("SMS classifier: ACTIVE")
    print("QR classifier: ACTIVE")
    
    port = int(os.environ.get('PORT', 10000))
    print(f"\nServer running on port: {port}")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=port, debug=False)
