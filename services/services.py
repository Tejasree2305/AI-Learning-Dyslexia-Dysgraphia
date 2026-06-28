# services.py

import os
import cv2
import numpy as np
import random
from nltk.corpus import words as nltk_words
from nltk.tokenize import word_tokenize
from Levenshtein import distance as levenshtein
from typing import List, Tuple
from datetime import datetime


# --- Configuration & Initialization ---
english_vocab = set(w.lower() for w in nltk_words.words())
DYSLEXIA_SCORE_THRESHOLD = 0.4 


# --- Assessment Sentences Pool ---
SENTENCE_POOL = [
    "The delicious aroma of freshly baked bread filled the kitchen.",
    "Geometry involves studying shapes and their properties.",
    "A colorful parrot spoke amusing phrases with clarity.",
    "Cybersecurity professionals protect systems and confidential data.",
    "Biodiversity refers to the variety of life across species.",
    "A soaring eagle circled high above the mountain peaks.",
    "The ancient pyramid stood silently under the desert sun.",
    "Complex algorithms drive the artificial intelligence systems.",
    "The locomotive chugged along the winding track into the distance.",
    "Photosynthesis is the process plants use to create food."
]

# Global Model Variable
XCEPTION_MODEL = None 

# ==============================================================================
# 1. DYSLEXIA DETECTION (NLP/Levenshtein) - UNCHANGED
# ==============================================================================

def get_sentences(count: int = 3) -> List[str]:
    return random.sample(SENTENCE_POOL, count)

def calculate_dyslexia_score(spoken_text: str, reference_text: str) -> float:
    """Calculates a normalized Levenshtein-based score (0 to 1)."""
    s_tokens = word_tokenize(spoken_text.lower())
    t_tokens = word_tokenize(reference_text.lower())
    
    s = " ".join(s_tokens)
    t = " ".join(t_tokens)
    
    lev_dist = levenshtein(s, t)
    max_len = max(len(s), len(t))
    
    if max_len == 0:
        return 0.0 
    
    normalized_score = lev_dist / max_len
    
    errors = 0
    for word in s_tokens:
        if word.isalpha() and len(word) > 2 and word not in english_vocab:
            errors += 0.05 
    
    final_score = min(normalized_score + errors, 1.0)
    
    return final_score


# ==============================================================================
# 2. DYSGRAPHIA DETECTION (Deep Learning/CV) - UNCHANGED
# ==============================================================================

def preprocess_image(image_path: str, size: Tuple[int, int] = (128, 128)) -> np.ndarray:
    """Prepares a handwriting image for the CNN model."""
    image = cv2.imread(image_path)
    if image is None: 
        raise ValueError(f"Image not found or corrupt at path: {image_path}")
        
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    resized = cv2.resize(binary, size, interpolation=cv2.INTER_AREA)
    
    return resized

def predict_handwriting(image_path: str) -> Tuple[str, float]:
    """Dysgraphia Prediction using the globally loaded Keras model or simulation."""
    global XCEPTION_MODEL
    
    # --- SIMULATION/FALLBACK LOGIC ---
    if XCEPTION_MODEL is None:
        if random.random() < 0.3:
            label = "Non-Dysgraphia"
            conf = random.uniform(0.85, 0.99)
        else:
            label = "Likelihood of Dysgraphia Detected"
            conf = random.uniform(0.6, 0.9)
            
        try:
            preprocess_image(image_path)
        except ValueError:
            return "Processing Error: Image File Missing", 0.0
            
        return label, float(conf)
    
    # --- REAL PREDICTION FLOW ---
    try:
        img = preprocess_image(image_path)
        img = img / 255.0
        img = np.stack((img,) * 3, axis=-1) 
        img = np.expand_dims(img, axis=0)
        from tensorflow.keras.models import load_model 
        DYSGRAPHIA_MODEL_PATH = "model.h5"
        XCEPTION_MODEL = load_model(DYSGRAPHIA_MODEL_PATH)

        pred = XCEPTION_MODEL.predict(img)[0][0] 
        
        if pred > 0.5:
            label = "Likelihood of Dysgraphia Detected"
            conf = pred
        else:
            label = "Non-Dysgraphia"
            conf = 1 - pred
            
        return label, float(conf)
        
    except Exception as e:
        print(f"Prediction failed during runtime: {e}")
        return "Internal Model Error", 0.0

# ==============================================================================
# 3. GEMINI AI CURRICULUM GENERATION (Refined with <br> breaks)
# ==============================================================================

def generate_personalized_curriculum(child_name: str, dyslexia_score: float, dysgraphia_score: float, week: int) -> str:
    """
    Simulates generating a personalized weekly curriculum with clear, step-by-step procedures.
    Uses HTML <ul>/<li> and explicit <br> tags for maximum readability in the web template.
    """
    
    dyslexia_risk = "High" if dyslexia_score >= 0.4 else "Moderate" if dyslexia_score >= 0.2 else "Low"
    dysgraphia_risk = "High" if dysgraphia_score >= 0.6 else "Moderate" if dysgraphia_score >= 0.4 else "Low"
    
    # --- Weekly Reading Focus (Structured Steps) ---
    if week == 1:
        reading_title = "Phonological Awareness<br>: Short Vowels & CVC Words"
        reading_steps = [
            "**Step 1: Introduction (5 min):** Focus on **short vowel sounds** (A, E, I, O, U).<br>Use visual aids/flashcards.",
            "**Step 2: Blending Practice (10 min):** Practice blending **CVC words** (e.g., c-a-t, b-e-d).<br>Emphasize segmenting and blending sounds.",
            "**Step 3: Guided Reading (15 min):** Use CVC-heavy decodable texts or short stories focusing on the short 'a' sound.",
            "**Daily Goal:** Achieve **less than 3 miscue errors** in the 50-word reading passage by the end of the week."
        ]
        
        handwriting_title = "Early Handwriting: Basic Grip and Letter Size"
        handwriting_steps = [
            "**Step 1: Grip Practice (5 min):** Focus on maintaining a proper **tripod grip** on the pencil.",
            "**Step 2: Letter Focus (10 min):** Practice forming **o, a, c, e** (small, continuous letters).<br>Emphasize starting and stopping points.",
            "**Step 3: Fine Motor Drill:** Complete a simple **tracing exercise** (e.g., circles and straight lines) for 5 minutes daily.",
            "**Daily Goal:** Maintain **consistent letter size** on at least 80% of lines (use lined paper for reference)."
        ]
        
    elif week == 2:
        reading_title = "Phonics: Long Vowels & Silent 'E' Pattern"
        reading_steps = [
            "**Step 1: Teach Rule:** Introduce the **Silent 'E'** rule (e.g., cap -> cape).<br>Explain how the 'e' changes the vowel sound.",
            "**Step 2: Contrast Drill:** Practice reading pairs of contrasting words (e.g., pin/pine, cut/cute) to reinforce the rule.",
            "**Step 3: Dictation:** Dictate silent 'e' words and have the child write them, emphasizing the need for the final 'e' to make the vowel 'long'.",
            "**Daily Goal:** Correctly **pronounce 10 out of 12** target words containing the silent 'e' pattern."
        ]
        
        handwriting_title = "Letter Formation: Ascenders (b, d, h, k)"
        handwriting_steps = [
            "**Step 1: Warm-up:** Practice straight vertical strokes and loops to prepare for ascenders.",
            "**Step 2: Letter Focus (15 min):** Practice **h, k, b, d** (letters that go above the midline).<br>Emphasize proper height and loop formation.",
            "**Step 3: Visual Memory:** Use a model letter and ask the child to copy it *after* the model is covered to improve visual recall of the stroke pattern.",
            "**Daily Goal:** Form 10 target ascender letters correctly **without exceeding the top line** (use three-line paper)."
        ]
        
    elif week == 3:
        reading_title = "Advanced Phonics: Consonant Blends & Digraphs"
        reading_steps = [
            "**Step 1: Identify Sounds:** Introduce common initial **consonant blends** (st, bl, tr) and digraphs (sh, ch, th).<br>Stress that blends keep both sounds.",
            "**Step 2: Articulation:** Use a mirror to help the child practice smooth articulation of the blend sounds (e.g., /s/-/t/ in 'stop').",
            "**Step 3: Phrase Reading:** Practice reading short phrases that incorporate 3 or more consonant blends to build fluidity.",
            "**Daily Goal:** **Accurately read phrases** containing 3 or more consonant blends smoothly."
        ]
        
        handwriting_title = "Spacing & Descenders (p, q, y, g)"
        handwriting_steps = [
            "**Step 1: Descenders:** Focus on **p, q, y, g** (letters that drop below the baseline).<br>Use colored lines (e.g., green baseline) to highlight the drop.",
            "**Step 2: Spacing Tool:** Introduce using a **finger or popsicle stick** as a physical guide to enforce consistent word spacing.",
            "**Step 3: Copying Task:** Copy a short sentence focusing *only* on maintaining proper **word spacing**.",
            "**Daily Goal:** Maintain **proper vertical alignment** for 8 out of 10 descenders AND limit word spacing errors to 2 or fewer."
        ]
        
    else: # week 4
        reading_title = "Fluency Building & Error Correction"
        reading_steps = [
            "**Step 1: Syllabification:** Practice breaking **multi-syllable words** into parts (e.g., con-tain-er) before reading them.",
            "**Step 2: Timed Reading:** Implement **3-minute timed reading** sessions using previously reviewed text to track Words Per Minute (WPM).",
            "**Step 3: Quick Correction Cycle:** Teach the child to quickly spot and correct their own errors using the sound-out strategy.",
            "**Daily Goal:** Increase reading speed to **50 WPM** (Words Per Minute) on a 75-word passage."
        ]
        
        handwriting_title = "Speed, Legibility, and Review"
        handwriting_steps = [
            "**Step 1: Timed Copying:** Complete a short copying task with a **time limit** (e.g., 2 minutes) to practice writing speed.",
            "**Step 2: Legibility Check:** Immediately after timing, circle any letter that is hard to read or is floating off the line for **self-correction**.",
            "**Step 3: Multi-Task:** Complete **3 short dictation exercises** using simple sentences, focusing on listening, writing, and proofreading simultaneously.",
            "**Daily Goal:** Maintain legibility and complete the timed copying task while minimizing inconsistent spacing."
        ]


    # 4. Construct the HTML Curriculum Plan with clear list formatting
    plan_html = f"""
    <h4 style="color: #007bff; margin-bottom: 15px;">🎯 **WEEK {week} PROCEDURE: Personalized Intervention**</h4>
    
    <div style="padding: 10px; border: 1px solid #dc3545; border-radius: 4px; margin-bottom: 15px;">
        <h5 style="color: #dc3545;">📖 Reading & Spelling ({dyslexia_risk} Risk)</h5>
        <p style="font-weight: bold; margin-top: 5px; margin-bottom: 5px; text-decoration: underline;">Focus: {reading_title}</p>
        <ul style="margin-left: 20px; padding-left: 10px; line-height: 1.6;">
            {''.join([f'<li style="margin-bottom: 5px;">{step}</li>' for step in reading_steps])}
        </ul>
    </div>
    
    <div style="padding: 10px; border: 1px solid #28a745; border-radius: 4px; margin-bottom: 15px;">
        <h5 style="color: #28a745;">✍️ Handwriting & Motor ({dysgraphia_risk} Risk)</h5>
        <p style="font-weight: bold; margin-top: 5px; margin-bottom: 5px; text-decoration: underline;">Focus: {handwriting_title}</p>
        <ul style="margin-left: 20px; padding-left: 10px; line-height: 1.6;">
            {''.join([f'<li style="margin-bottom: 5px;">{step}</li>' for step in handwriting_steps])}
        </ul>
    </div>
    
    <p style='margin-top: 15px; font-style: italic; color: #555; padding: 10px; border-left: 3px solid #ffc107; background-color: #fffbe6;'>
    **🧠 Gemini AI Recommendation for Parent:** {child_name} responds well to **{'auditory learning and verbal repetition' if dyslexia_score >= 0.5 else 'kinesthetic tasks and movement breaks'}.** Use a **multisensory** approach for all activities.
    </p>
    """
    
    return plan_html


import os
import json
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

def init_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables.")
    genai.configure(api_key=api_key)
    # Using the model string you provided
    return genai.GenerativeModel("gemini-1.5-flash") 

model = init_gemini()

def generate_full_4_week_curriculum(child_name, dyslexia_score, dysgraphia_score):
    """Generates a complete 4-week intervention plan in one AI call."""
    
    prompt = f"""
    Act as a specialized Educational Therapist. Create a 4-week intervention plan for a child named {child_name}.
    Assessment Scores (0 to 1 scale, where 1 is high risk):
    - Dyslexia Score: {dyslexia_score}
    - Dysgraphia Score: {dysgraphia_score}

    The plan must be structured as a JSON object with keys "Week 1", "Week 2", "Week 3", and "Week 4".
    Each week should include:
    1. "Focus": The main goal for the week.
    2. "Activities": A list of 3 specific exercises.
    3. "Parent_Tip": A short piece of advice for the parent.

    Return ONLY the JSON object.
    """

    response = model.generate_content(prompt)
    
    # Clean the response text (remove markdown backticks if present)
    clean_json = response.text.replace('```json', '').replace('```', '').strip()
    return json.loads(clean_json)