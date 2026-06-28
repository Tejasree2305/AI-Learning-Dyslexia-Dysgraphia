# avatar_app.py
import io
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import joblib
import speech_recognition as sr
import pyttsx3
import os
import re
from datetime import datetime
import queue
import threading

app = Flask(__name__)

# Initialize Text-to-Speech engine
try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)
    tts_engine.setProperty('volume', 0.8)
    tts_available = True
except:
    tts_available = False
    print("⚠️ TTS engine not available")

# Speech queue for thread-safe TTS
speech_queue = queue.Queue()
tts_thread_running = True

def tts_worker():
    """Background worker for TTS to avoid threading issues"""
    while tts_thread_running:
        try:
            text = speech_queue.get(timeout=1)
            if text is None:
                break
            if tts_available:
                tts_engine.say(text)
                tts_engine.runAndWait()
            speech_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"TTS Error: {e}")

# Start TTS worker thread
tts_thread = threading.Thread(target=tts_worker, daemon=True)
tts_thread.start()

# BERT + Neural Network Model Class
class EnhancedBERTClassifier(nn.Module):
    def __init__(self, model_name='prajjwal1/bert-mini', n_classes=5, dropout_rate=0.4):
        super(EnhancedBERTClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),
            nn.Linear(64, n_classes)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(pooled_output)
        return logits

# Load BERT + Neural Network model
try:
    bert_training_info = joblib.load('enhanced_bert_info.pkl')
    MODEL_NAME = bert_training_info.get('model_name', 'prajjwal1/bert-mini')
    
    bert_tokenizer = joblib.load('enhanced_bert_tokenizer.pkl')
    bert_model = EnhancedBERTClassifier(model_name=MODEL_NAME, n_classes=5)
    bert_model.load_state_dict(torch.load('enhanced_bert_nn_model.pth', map_location=torch.device('cpu')))
    bert_model.eval()
    
    print("✅ BERT + Neural Network model loaded successfully!")
    
except Exception as e:
    print(f"❌ Error loading BERT model: {str(e)}")
    bert_tokenizer = None
    bert_model = None

# Store uploaded resumes and results
uploaded_resumes = {}
results_data = {}

def extract_name_from_resume(text):
    """Extract candidate name from resume text"""
    lines = text.split('\n')
    for line in lines[:10]:
        line = line.strip()
        if len(line) > 2 and len(line) < 50 and not any(char.isdigit() for char in line):
            words = line.split()
            if 1 <= len(words) <= 3:
                return line
    # Fallback: use filename or generic name
    return "Candidate"

def predict_bert_match(resume_text, job_description):
    """Predict match score using BERT + Neural Network"""
    if not bert_tokenizer or not bert_model:
        return None, None, None
    
    try:
        job_desc_short = str(job_description)[:800]
        resume_short = str(resume_text)[:800]
        text = f"Job Requirements: {job_desc_short} Candidate Qualifications: {resume_short}"
        
        encoding = bert_tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = bert_model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            )
            probabilities = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, dim=1)
            
            prediction = int(prediction.item() + 1)
            confidence = float(confidence.item())
            
            is_shortlisted = bool(prediction >= 4)
            return is_shortlisted, confidence, prediction
            
    except Exception as e:
        print(f"BERT prediction error: {e}")
        return None, None, None

def speak_text(text):
    """Safely speak text using TTS via queue"""
    if tts_available:
        speech_queue.put(text)

def record_audio():
    """Record audio and convert to text with better error handling"""
    try:
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            print("🎤 Listening... Speak now!")
            recognizer.adjust_for_ambient_noise(source, duration=1)
            audio = recognizer.listen(source, timeout=10, phrase_time_limit=15)
            print("✅ Processing speech...")
            text = recognizer.recognize_google(audio)
            print(f"📝 You said: {text}")
            return text
    except sr.WaitTimeoutError:
        return "timeout"
    except sr.UnknownValueError:
        return "unclear"
    except sr.RequestError as e:
        print(f"Speech recognition service error: {e}")
        return "service_error"
    except Exception as e:
        print(f"Speech recognition error: {e}")
        return "error"

@app.route('/')
def index():
    return render_template('avatar_index.html')

@app.route('/upload_resumes', methods=['POST'])
def upload_resumes():
    """Handle multiple resume uploads"""
    try:
        if 'resumes' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('resumes')
        uploaded_files = []
        
        for file in files:
            if file and file.filename != '':
                if file.filename.lower().endswith(('.txt', '.pdf')):
                    try:
                        if file.filename.lower().endswith('.txt'):
                            resume_text = file.read().decode('utf-8')
                        else:
                            import PyPDF2
                            file_stream = io.BytesIO(file.read())
                            pdf_reader = PyPDF2.PdfReader(file_stream)
                            resume_text = ""
                            for page in pdf_reader.pages:
                                resume_text += page.extract_text() + " "
                        
                        candidate_name = extract_name_from_resume(resume_text)
                        
                        resume_id = f"resume_{len(uploaded_resumes) + 1}"
                        uploaded_resumes[resume_id] = {
                            'name': candidate_name,
                            'content': resume_text,
                            'filename': file.filename
                        }
                        
                        uploaded_files.append({
                            'id': resume_id,
                            'name': candidate_name,
                            'filename': file.filename
                        })
                    except Exception as e:
                        print(f"Error processing file {file.filename}: {e}")
                        continue
        
        return jsonify({
            'message': f'Successfully uploaded {len(uploaded_files)} resumes',
            'uploaded_files': uploaded_files
        })
        
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500

@app.route('/process_job', methods=['POST'])
def process_job():
    """Process job description and evaluate resumes"""
    try:
        data = request.get_json()
        job_description = data.get('job_description', '')
        use_voice = data.get('use_voice', False)
        
        if not job_description and not use_voice:
            return jsonify({'error': 'Please provide job description or use voice input'}), 400
        
        # If voice input requested
        voice_result = None
        if use_voice:
            voice_result = record_audio()
            if voice_result in ['timeout', 'unclear', 'error', 'service_error']:
                if voice_result == 'timeout':
                    return jsonify({'error': 'No speech detected. Please try again.'}), 400
                elif voice_result == 'unclear':
                    return jsonify({'error': 'Speech was unclear. Please try again.'}), 400
                else:
                    return jsonify({'error': 'Voice input failed. Please try typing instead.'}), 400
            job_description = voice_result
        
        if not job_description.strip():
            return jsonify({'error': 'No job description provided'}), 400
        
        # Check if we have resumes
        if not uploaded_resumes:
            return jsonify({'error': 'No resumes uploaded. Please upload resumes first.'}), 400
        
        # Process all uploaded resumes
        results = []
        shortlisted_count = 0
        
        for resume_id, resume_data in uploaded_resumes.items():
            is_shortlisted, confidence, score = predict_bert_match(
                resume_data['content'], job_description
            )
            
            # Handle case where prediction fails
            if is_shortlisted is None:
                is_shortlisted = False
                confidence = 0
                score = 0
            
            result = {
                'id': resume_id,
                'name': resume_data['name'],
                'shortlisted': is_shortlisted,
                'confidence': round(float(confidence) * 100, 2),
                'score': score,
                'filename': resume_data['filename']
            }
            results.append(result)
            
            if is_shortlisted:
                shortlisted_count += 1
        
        # Sort by score (highest first)
        results.sort(key=lambda x: x['score'], reverse=True)
        
        # Store results
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_data[session_id] = {
            'job_description': job_description,
            'results': results,
            'timestamp': datetime.now().isoformat(),
            'voice_input': use_voice
        }
        
        # Generate speech output
        if shortlisted_count > 0:
            shortlisted_names = [r['name'] for r in results if r['shortlisted']]
            if len(shortlisted_names) == 1:
                speech_text = f"I found {shortlisted_count} suitable candidate. The shortlisted candidate is {shortlisted_names[0]}."
            else:
                names_text = ', '.join(shortlisted_names[:-1]) + ' and ' + shortlisted_names[-1]
                speech_text = f"I found {shortlisted_count} suitable candidates. The shortlisted candidates are: {names_text}."
        else:
            speech_text = "No candidates were shortlisted for this job description."
        
        # Speak the results
        speak_text(speech_text)
        
        return jsonify({
            'session_id': session_id,
            'job_description': job_description,
            'results': results,
            'shortlisted_count': shortlisted_count,
            'total_candidates': len(results),
            'speech_text': speech_text,
            'voice_used': use_voice
        })
        
    except Exception as e:
        print(f"Process job error: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500

@app.route('/speak_results', methods=['POST'])
def speak_results():
    """Speak the results again"""
    try:
        data = request.get_json()
        session_id = data.get('session_id')
        
        if session_id not in results_data:
            return jsonify({'error': 'Session not found'}), 404
        
        results = results_data[session_id]['results']
        shortlisted = [r for r in results if r['shortlisted']]
        
        if shortlisted:
            names = [r['name'] for r in shortlisted]
            if len(names) == 1:
                speech_text = f"The shortlisted candidate is {names[0]}."
            else:
                names_text = ', '.join(names[:-1]) + ' and ' + names[-1]
                speech_text = f"The shortlisted candidates are: {names_text}."
        else:
            speech_text = "No candidates were shortlisted."
        
        speak_text(speech_text)
        
        return jsonify({'message': 'Speaking results', 'speech_text': speech_text})
        
    except Exception as e:
        return jsonify({'error': f'Speech failed: {str(e)}'}), 500

@app.route('/get_sessions')
def get_sessions():
    """Get all processing sessions"""
    sessions = []
    for session_id, data in results_data.items():
        sessions.append({
            'session_id': session_id,
            'job_description_preview': data['job_description'][:100] + '...' if len(data['job_description']) > 100 else data['job_description'],
            'timestamp': data['timestamp'],
            'total_candidates': len(data['results']),
            'shortlisted_count': len([r for r in data['results'] if r['shortlisted']]),
            'voice_input': data.get('voice_input', False)
        })
    
    sessions.sort(key=lambda x: x['timestamp'], reverse=True)
    return jsonify({'sessions': sessions})

@app.route('/get_session/<session_id>')
def get_session(session_id):
    """Get specific session details"""
    if session_id not in results_data:
        return jsonify({'error': 'Session not found'}), 404
    
    return jsonify(results_data[session_id])

@app.route('/clear_resumes', methods=['POST'])
def clear_resumes():
    """Clear all uploaded resumes"""
    uploaded_resumes.clear()
    return jsonify({'message': 'All resumes cleared'})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'bert_loaded': bert_model is not None,
        'tts_available': tts_available,
        'resumes_uploaded': len(uploaded_resumes),
        'sessions_stored': len(results_data)
    })

def cleanup():
    """Cleanup function for graceful shutdown"""
    global tts_thread_running
    tts_thread_running = False
    if tts_available:
        tts_engine.stop()

import atexit
atexit.register(cleanup)

if __name__ == '__main__':
    print("🚀 Starting AI Avatar Resume Screener...")
    print("🔊 TTS Available:", tts_available)
    print("🤖 BERT Model Loaded:", bert_model is not None)
    
    if tts_available:
        speak_text("AI Avatar system ready. Please upload resumes and provide a job description.")
    
    # Use threaded=False to avoid Flask threading issues
    app.run(debug=True, host='0.0.0.0', port=5001, threaded=False)