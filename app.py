# app.py

import os
import sqlite3
import csv
import json
from datetime import datetime
from functools import wraps
from flask import Flask, render_template, request, redirect, url_for, session, send_file, jsonify, flash
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import nltk
from dotenv import load_dotenv # <-- NEW: Import for .env loading

# --------------------------------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------------------------------
# This will look for a .env file in the current directory and load its contents
load_dotenv()

# NOTE: Ensure services/services.py exists and contains these functions.
from services.services import (
    get_sentences, calculate_dyslexia_score, predict_handwriting, 
    generate_personalized_curriculum, DYSLEXIA_SCORE_THRESHOLD
)

# Initialize NLTK
try:
    nltk.download('punkt', quiet=True)
    nltk.download('words', quiet=True)
except:
    pass

# --- Config ---
UPLOAD_FOLDER = 'static/uploads'
DATABASE = 'dyslexia_app.db' 
TODO_DATABASE = 'todo_curriculum.db'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.secret_key = 'dyslexia_app_secret_2025_advanced'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


# ==============================================================================
# 1. DATABASE HELPERS AND INITIALIZATION
# ==============================================================================

def get_db_connection():
    conn = sqlite3.connect(DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def get_latest_scores_from_db(child_id):
    """Retrieves the latest dyslexia + dysgraphia scores."""
    conn = get_db_connection()
    latest_assessment = conn.execute("""
        SELECT dyslexia_score, dysgraphia_score
        FROM assessments
        WHERE child_id = ?
        ORDER BY created_at DESC
        LIMIT 1
    """, (child_id,)).fetchone()
    conn.close()

    if latest_assessment:
        return {
            'dyslexia_score': latest_assessment['dyslexia_score'],
            'dysgraphia_score': latest_assessment['dysgraphia_score']
        }
    return None

def init_db():
    conn = get_db_connection()
    c = conn.cursor()

    c.execute('''
        CREATE TABLE IF NOT EXISTS parents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            contact TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS children (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            parent_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            age INTEGER NOT NULL,
            school_class TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (parent_id) REFERENCES parents(id) ON DELETE CASCADE
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            child_id INTEGER,
            parent_id INTEGER,
            screening_type TEXT NOT NULL,
            dyslexia_score REAL DEFAULT 0.0,
            dysgraphia_score REAL DEFAULT 0.0,
            verdict TEXT,
            raw_data_path TEXT,
            reference_text TEXT, 
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (child_id) REFERENCES children(id) ON DELETE SET NULL,
            FOREIGN KEY (parent_id) REFERENCES parents(id) ON DELETE CASCADE
        )
    ''')

    c.execute('''
        CREATE TABLE IF NOT EXISTS interventions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            child_id INTEGER NOT NULL,
            start_date TIMESTAMP NOT NULL,
            current_week INTEGER DEFAULT 1,
            curriculum_json TEXT,
            last_completed_at TIMESTAMP,
            FOREIGN KEY (child_id) REFERENCES children(id) ON DELETE CASCADE
        )
    ''')

    conn.commit()
    conn.close()

# ==============================================================================
def get_todo_db_connection():
    """Get connection to the separate to-do database"""
    conn = sqlite3.connect(TODO_DATABASE)
    conn.row_factory = sqlite3.Row
    return conn

def init_todo_db():
    """Initialize the separate to-do database"""
    conn = get_todo_db_connection()
    c = conn.cursor()
    
    # Table for static curriculum activities
    c.execute('''
        CREATE TABLE IF NOT EXISTS static_curriculum (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            difficulty_level TEXT NOT NULL,  -- 'low', 'medium', 'high'
            week_number INTEGER NOT NULL,
            day_number INTEGER NOT NULL,
            activity_title TEXT NOT NULL,
            activity_description TEXT NOT NULL,
            dyslexia_focus BOOLEAN DEFAULT 0,
            dysgraphia_focus BOOLEAN DEFAULT 0,
            estimated_minutes INTEGER DEFAULT 15
        )
    ''')
    
    # Table for child progress tracking
    c.execute('''
        CREATE TABLE IF NOT EXISTS child_progress_tracking (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            child_id INTEGER NOT NULL,
            curriculum_type TEXT DEFAULT 'static',
            week_number INTEGER NOT NULL,
            day_number INTEGER NOT NULL,
            activity_id INTEGER NOT NULL,
            completed BOOLEAN DEFAULT 0,
            completed_at TIMESTAMP,
            notes TEXT
        )
    ''')
    
    # Table for child curriculum assignments
    c.execute('''
        CREATE TABLE IF NOT EXISTS child_curriculum (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            child_id INTEGER NOT NULL,
            difficulty_level TEXT NOT NULL,
            start_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        )
    ''')
    
    conn.commit()
    conn.close()
    populate_todo_curriculum()  # Populate with default activities

def populate_todo_curriculum():
    """Populate the to-do database with default activities (only for medium and high difficulty)"""
    conn = get_todo_db_connection()
    c = conn.cursor()
    
    # Check if data already exists
    count = c.execute("SELECT COUNT(*) FROM static_curriculum").fetchone()[0]
    
    if count == 0:
        # NO LOW DIFFICULTY ACTIVITIES - Children with scores < 0.3 get congratulations message
        
        # MEDIUM DIFFICULTY (scores 0.3-0.7): MODERATE SUPPORT NEEDED
        medium_difficulty = [
            # Week 1: Foundational Skills
            (1, 1, 'Sound-by-Sound Word Breaking', 
             'Take 10 simple 3-letter words (cat, dog, sun, etc.). Say each word slowly, breaking it into individual sounds. For "cat", say "c-a-t" clearly. Write each sound on a separate card and have the child point to each sound as they say it.', 
             1, 0, 15),
            
            (1, 2, 'Letter Formation with Guided Strokes', 
             'Practice writing each letter of the alphabet using dotted-line worksheets. Focus on correct starting points and stroke direction. For example, start "a" at the top of the circle, not the bottom. Use arrows to show proper movement.', 
             0, 1, 15),
            
            (1, 3, 'Interactive Sight Word Recognition', 
             'Create 10 sight word cards (the, and, is, you, etc.). Place them face down. Turn one over and have the child read it within 5 seconds. If correct, they keep it. Create simple sentences using collected words.', 
             1, 0, 20),
            
            (1, 4, 'Pattern Tracing for Motor Control', 
             'Use worksheets with increasingly complex patterns (straight lines, curves, zigzags, circles). Trace each pattern 5 times, focusing on smooth, continuous movement without lifting the pencil.', 
             0, 1, 15),
            
            (1, 5, 'Rhyming Word Family Exploration', 
             'Choose a word family (-at). Brainstorm and write all words in that family (cat, bat, hat, sat, mat). Create silly sentences using at least 3 words from the family. Draw pictures for each word.', 
             1, 0, 15),
            
            # Week 2: Building Confidence
            (2, 1, 'Picture Context Reading', 
             'Show a picture with several elements (park scene with kids playing, trees, animals). Ask: "What do you see?" Write down observations. Then read a short paragraph about the scene, connecting words to picture elements.', 
             1, 0, 20),
            
            (2, 2, 'Sentence Building with Word Cards', 
             'Create cards with different parts of speech: subjects (The cat, A dog), verbs (jumped, slept), objects (on the mat, in the sun). Mix and match to create 5 grammatically correct sentences. Write them neatly.', 
             1, 1, 20),
            
            (2, 3, 'Emotional Expression Reading', 
             'Write the same sentence 5 times with different emotions: "I want ice cream." Read it as happy, sad, excited, angry, and scared. Discuss how punctuation changes meaning (period vs exclamation).', 
             1, 0, 15),
            
            (2, 4, 'Consistent Spacing Practice', 
             'Write a simple sentence. Place a small sticker or draw a dot between each word as a spacing guide. Rewrite the sentence 3 times, using the sticker as a reminder for consistent finger-width spacing.', 
             0, 1, 15),
            
            (2, 5, 'Spelling Pattern Recognition', 
             'Focus on "sh" pattern. Find 10 words with "sh" (ship, fish, wish, etc.). Sort them by where "sh" appears (beginning, middle, end). Create a story using at least 5 "sh" words.', 
             1, 0, 20),
            
            # Week 3: Skill Development
            (3, 1, 'Comprehension through Question Stems', 
             'Read a short paragraph (5-7 sentences). Answer: 1) Who are the characters? 2) Where does this take place? 3) What happened first? 4) What happened next? 5) How did it end? Draw a simple timeline of events.', 
             1, 0, 20),
            
            (3, 2, 'Paragraph Structure with Topic Sentences', 
             'Choose a topic (My Pet, My Favorite Food, A Rainy Day). Write a topic sentence. Add 3 supporting sentences with details. End with a concluding sentence. Focus on staying on topic.', 
             0, 1, 25),
            
            (3, 3, 'Descriptive Vocabulary Expansion', 
             'Take a simple word (big). Brainstorm synonyms (large, huge, enormous, gigantic, massive). Use each in a sentence. Create a "word wall" with the new vocabulary words and their meanings.', 
             1, 0, 15),
            
            (3, 4, 'Neatness Challenge with Self-Assessment', 
             'Write the alphabet in order. Then rate each letter: ✓ for perfect, ~ for okay, ✗ for needs work. Circle the 3 letters needing most improvement and practice those specifically 5 times each.', 
             0, 1, 20),
            
            (3, 5, 'Error Detection and Correction', 
             'Read a paragraph with 5 intentional errors (spelling, punctuation, capitalization). Identify and correct each error. Explain why each correction is needed (capitalize names, periods end sentences, etc.).', 
             1, 1, 15),
            
            # Week 4: Application
            (4, 1, 'Story Retelling with Sequencing', 
             'Read a short story (3 paragraphs). Retell it in your own words to someone else. Use transition words: First, Next, Then, Finally. Draw 4 pictures showing the beginning, middle, and end.', 
             1, 0, 25),
            
            (4, 2, 'Creative Story Writing Framework', 
             'Use story structure: 1) Characters and setting (Once upon a time...), 2) Problem (But one day...), 3) Solution (So they decided to...), 4) Ending (And they lived...). Write 5-7 sentences following this structure.', 
             0, 1, 30),
            
            (4, 3, 'Timed Reading for Fluency', 
             'Choose a familiar passage. Time how long it takes to read it aloud smoothly. Record the time. Practice difficult words separately. Read again, trying to beat your time while maintaining accuracy.', 
             1, 0, 20),
            
            (4, 4, 'Letter Size Consistency Grid Practice', 
             'Use grid paper with 1-inch squares. Write each letter to fill exactly one square (small letters) or two squares (tall letters like b, d, h). Practice maintaining consistent size across the entire alphabet.', 
             0, 1, 20),
            
            (4, 5, 'Integrated Reading-Writing Application', 
             'Read a short informational text (about animals, plants, etc.). Write: 1) Two facts you learned, 2) One question you still have, 3) A connection to something you already knew. Use complete sentences.', 
             1, 1, 25),
        ]
        
        # HIGH DIFFICULTY (scores > 0.7): INTENSIVE SUPPORT NEEDED
        high_difficulty = [
            # Week 1: Intensive Foundations
            (1, 1, 'Multi-sensory Letter-Sound Association', 
             'Use textured materials (sand, rice, shaving cream). Say a letter sound clearly while tracing the letter in the material. Associate each letter with a keyword (a-apple, b-ball). Trace with eyes closed to reinforce muscle memory.', 
             1, 1, 25),
            
            (1, 2, 'Fine Motor Skill Development Sequence', 
             'Start with large motor movements (drawing big circles in air), progress to medium (tracing large shapes on paper), then small (using tweezers to pick up small beads). End with precision tasks (threading beads on a string).', 
             0, 1, 20),
            
            (1, 3, 'Systematic Sound Blending with Visual Aids', 
             'Use colored blocks to represent sounds. Push one block for each sound while saying it slowly. Gradually push blocks closer together while blending sounds. Progress from 2-sound to 3-sound words with continuous support.', 
             1, 0, 20),
            
            (1, 4, 'Pencil Grip Correction with Adaptive Tools', 
             'Use pencil grips or triangular pencils. Practice proper tripod grip with "pinch and flip" method. Do hand-strengthening exercises (squeezing stress balls) before writing. Trace thick lines with adaptive grips.', 
             0, 1, 15),
            
            (1, 5, 'Phonemic Awareness with Manipulatives', 
             'Use counting chips or tokens. Say a word (cat). Have child push one chip for each sound they hear (3 chips). Identify beginning, middle, ending sounds using colored chips for each position.', 
             1, 0, 20),
            
            # Week 2: Targeted Interventions
            (2, 1, 'Structured Phonics Pattern Recognition', 
             'Focus on one vowel pattern at a time (short a words: cat, bat, hat). Create word families. Use color coding (vowels in red, consonants in blue). Practice until automatic, then move to next pattern with cumulative review.', 
             1, 0, 30),
            
            (2, 2, 'Assistive Writing Tool Exploration', 
             'Try different tools: weighted pens, slanted boards, raised line paper. Determine which reduces fatigue. Practice writing with preferred tool for 5 minutes, rest, repeat. Document which tool works best.', 
             0, 1, 25),
            
            (2, 3, 'Decoding Strategy Toolkit Development', 
             'Learn and apply strategies: 1) Look for known parts, 2) Break into syllables, 3) Look at pictures for clues, 4) Skip and return, 5) Guess and check. Apply each strategy to 5 unfamiliar words with guidance.', 
             1, 0, 25),
            
            (2, 4, 'Spatial Organization with Visual Guides', 
             'Use graph paper or specially lined paper with colored zones (green for top line, yellow for middle, red for bottom). Practice keeping letters within appropriate zones. Use highlighted baseline for alignment.', 
             0, 1, 20),
            
            (2, 5, 'Mnemonic Device Creation for Spelling', 
             'Create memorable associations for tricky words: "because" = Big Elephants Can Always Understand Small Elephants. Draw pictures to match mnemonics. Practice using the mnemonic to recall spelling.', 
             1, 0, 25),
            
            # Week 3: Skill Building
            (3, 1, 'Guided Reading with Scaffolded Support', 
             'Read together: adult reads a sentence, child repeats. Progress to alternating sentences. Use sticky notes to mark difficult words. Stop frequently to check comprehension with simple questions.', 
             1, 0, 30),
            
            (3, 2, 'Writing Stamina Building with Scheduled Breaks', 
             'Set timer for 5 minutes of focused writing, then 3-minute break with stretching. Gradually increase writing time by 1 minute each session while decreasing break time. Track progress in a stamina chart.', 
             0, 1, 35),
            
            (3, 3, 'Subject-Specific Vocabulary with Visual Supports', 
             'Choose a topic (weather). Learn 5 key terms (precipitation, temperature, forecast). Create flashcards with word on front, definition and picture on back. Use in sentences related to the topic.', 
             1, 0, 25),
            
            (3, 4, 'Precision Writing with Gradual Complexity', 
             'Start with tracing thick dotted lines, progress to thinner lines, then copying without tracing. Use different writing tools (marker, pencil, pen) to build control. Focus on accuracy over speed.', 
             0, 1, 25),
            
            (3, 5, 'Memory-Based Writing Reinforcement', 
             'Read a sentence aloud twice. Cover it. Write it from memory. Compare to original. Identify missing elements. Repeat with gradually longer sentences. Focus on retaining meaning, not perfect replication.', 
             1, 1, 30),
            
            # Week 4: Practical Application
            (4, 1, 'Independent Strategy Application', 
             'Read a new passage without help. Use previously learned strategies independently. After reading, explain which strategies were used and why. Self-evaluate effectiveness of each strategy choice.', 
             1, 0, 35),
            
            (4, 2, 'Real-World Functional Writing Tasks', 
             'Write: 1) A grocery list for 5 items, 2) A note to a family member, 3) Instructions for a simple task (making a sandwich), 4) An email greeting. Focus on clarity and purpose rather than perfection.', 
             0, 1, 30),
            
            (4, 3, 'Test Strategy Rehearsal', 
             'Practice: 1) Skimming questions first, 2) Underlining key words, 3) Eliminating obviously wrong answers, 4) Managing time. Complete a practice test with these strategies, then review effectiveness.', 
             1, 0, 25),
            
            (4, 4, 'Speed-Legibility Balance Practice', 
             'Write the same sentence 3 ways: 1) As neatly as possible (no time limit), 2) At normal speed, 3) As fast as possible while still readable. Compare results and identify optimal balance for different situations.', 
             0, 1, 25),
            
            (4, 5, 'Progress Review and Personalized Planning', 
             'Review completed activities. Identify: 1) 3 biggest improvements, 2) 2 ongoing challenges, 3) 1 new goal. Create a simple maintenance plan with 2-3 weekly practice activities based on this review.', 
             1, 1, 30),
        ]
        
        # Insert only medium and high difficulty activities
        for activities, difficulty in [(medium_difficulty, 'medium'), (high_difficulty, 'high')]:
            for week, day, title, desc, dyslex_focus, dysgraph_focus, minutes in activities:
                c.execute('''
                    INSERT INTO static_curriculum 
                    (difficulty_level, week_number, day_number, activity_title, 
                     activity_description, dyslexia_focus, dysgraphia_focus, estimated_minutes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (difficulty, week, day, title, desc, dyslex_focus, dysgraph_focus, minutes))
        
        conn.commit()
        print("To-do curriculum data populated successfully (only medium and high difficulty)!")
    
    conn.close()

def get_difficulty_from_scores(dyslexia_score, dysgraphia_score):
    """Determine if curriculum is needed based on scores"""
    avg_score = (dyslexia_score + dysgraphia_score) / 2
    
    if avg_score < 0.3:
        return None  # No curriculum needed - child is doing well
    elif avg_score < 0.7:
        return 'medium'  # Moderate support needed
    else:
        return 'high'    # Intensive support needed

def get_child_todo_data(child_id):
    """Get to-do list data for a child"""
    conn = get_todo_db_connection()
    
    # Get child's assigned curriculum
    curriculum = conn.execute('''
        SELECT difficulty_level, start_date 
        FROM child_curriculum 
        WHERE child_id = ? AND is_active = 1
        ORDER BY start_date DESC LIMIT 1
    ''', (child_id,)).fetchone()
    
    if not curriculum:
        conn.close()
        return None
    
    difficulty = curriculum['difficulty_level']
    
    if difficulty == 'low':
        conn.close()
        return None    


    # Get activities for this difficulty
    activities = conn.execute('''
        SELECT * FROM static_curriculum 
        WHERE difficulty_level = ?
        ORDER BY week_number, day_number
    ''', (difficulty,)).fetchall()
    
    # Get progress for each activity
    activities_with_progress = []
    for activity in activities:
        progress = conn.execute('''
            SELECT completed, completed_at, notes 
            FROM child_progress_tracking 
            WHERE child_id = ? AND activity_id = ?
            ORDER BY id DESC LIMIT 1
        ''', (child_id, activity['id'])).fetchone()
        
        activity_dict = dict(activity)
        if progress:
            activity_dict.update({
                'completed': progress['completed'],
                'completed_at': progress['completed_at'],
                'notes': progress['notes'],
                'progress_id': progress['id'] if 'id' in progress else None
            })
        else:
            activity_dict.update({
                'completed': False,
                'completed_at': None,
                'notes': None,
                'progress_id': None
            })
        
        activities_with_progress.append(activity_dict)
    
    # Calculate statistics
    total_activities = len(activities_with_progress)
    completed_activities = sum(1 for a in activities_with_progress if a['completed'])
    progress_percent = int((completed_activities / total_activities * 100)) if total_activities > 0 else 0
    
    # Organize by week
    weeks_data = {}
    for activity in activities_with_progress:
        week = activity['week_number']
        if week not in weeks_data:
            weeks_data[week] = []
        weeks_data[week].append(activity)
    
    conn.close()
    
    return {
        'difficulty': difficulty,
        'start_date': curriculum['start_date'],
        'total_activities': total_activities,
        'completed_activities': completed_activities,
        'progress_percent': progress_percent,
        'weeks_data': weeks_data,
        'all_activities': activities_with_progress
    }
# ==============================================================================
# 2. AUTHENTICATION AND DECORATORS
# ==============================================================================

def login_required_role(role):
    def wrapper(f):
        @wraps(f)
        def decorated(*args, **kwargs):
            if role == 'admin' and session.get('role') != 'admin':
                flash("Please log in as Admin.", "warning")
                return redirect(url_for('login'))
            if role == 'parent' and session.get('role') != 'parent':
                flash("Please log in as Parent.", "warning")
                return redirect(url_for('login'))
            return f(*args, **kwargs)
        return decorated
    return wrapper

# Standardized Admin Decorator
admin_required = login_required_role('admin')
parent_required = login_required_role('parent')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Admin login
        if username == 'admin' and password == 'admin2025!':
            session['role'] = 'admin'
            flash("Admin login successful!", "success")
            return redirect(url_for('admin_dashboard'))

        # Parent login
        conn = get_db_connection()
        user = conn.execute(
            "SELECT id, password_hash FROM parents WHERE username = ?", 
            (username,)
        ).fetchone()
        conn.close()

        if user and check_password_hash(user['password_hash'], password):
            session['parent_id'] = user['id']
            session['role'] = 'parent'
            flash("Parent login successful!", "success")
            return redirect(url_for('parent_dashboard'))

        flash("Invalid credentials", "danger")

    return render_template('login.html')


@app.route('/parent/register', methods=['GET','POST'])
def parent_register():
    if request.method == 'POST':
        username = request.form['username']
        email    = request.form['email']
        password = request.form['password']
        contact  = request.form.get('contact','')

        pw_hash = generate_password_hash(password)

        try:
            conn = get_db_connection()
            conn.execute('''
                INSERT INTO parents (username, password_hash, email, contact)
                VALUES (?, ?, ?, ?)
            ''', (username, pw_hash, email, contact))
            conn.commit()
            flash("Registration successful! Please log in.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Username or email already exists.", "danger")
        finally:
            conn.close()

    return render_template('parent_register.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out.", "info")
    return redirect(url_for('index'))


# ==============================================================================
# 3. DASHBOARDS AND ADMIN TOOLS
# ==============================================================================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/admin/dashboard')
@admin_required
def admin_dashboard():
    conn = get_db_connection()

    records = conn.execute("""
        SELECT
            c.id, c.name, c.age,
            p.username AS parent_name,
            latest_a.created_at AS last_assessment_date,
            latest_a.dyslexia_score,
            latest_a.dysgraphia_score
        FROM children c
        JOIN parents p ON c.parent_id = p.id
        LEFT JOIN assessments latest_a
            ON latest_a.id = (
                SELECT MAX(id) FROM assessments WHERE child_id = c.id
            )
        GROUP BY c.id
        ORDER BY c.id DESC;
    """).fetchall()

    conn.close()

    return render_template('admin_dashboard.html',
                           records=[dict(r) for r in records])


# --- Parent Dashboard ---
@app.route('/parent/dashboard')
@parent_required
def parent_dashboard():
    parent_id = session['parent_id']
    conn = get_db_connection()

    parent = conn.execute(
        "SELECT username,email,contact FROM parents WHERE id=?",
        (parent_id,)
    ).fetchone()

    children = conn.execute(
        "SELECT id,name,age,school_class FROM children WHERE parent_id=?",
        (parent_id,)
    ).fetchall()

    child_data_list = []
    for c in children:
        cdict = dict(c)
        cid   = cdict['id']

        latest_assessment = conn.execute("""
            SELECT verdict, dyslexia_score, dysgraphia_score, created_at FROM assessments
            WHERE child_id=? 
            ORDER BY created_at DESC LIMIT 1
        """, (cid,)).fetchone()

        intervention = conn.execute("""
            SELECT current_week, start_date 
            FROM interventions 
            WHERE child_id=? ORDER BY start_date DESC LIMIT 1
        """, (cid,)).fetchone()

        cdict['dyslexia_score'] = latest_assessment['dyslexia_score'] if latest_assessment else None
        cdict['dysgraphia_score'] = latest_assessment['dysgraphia_score'] if latest_assessment else None
        cdict['latest_verdict'] = latest_assessment['verdict'] if latest_assessment else "N/A"
        cdict['last_tested'] = latest_assessment['created_at'].split()[0] if latest_assessment else "N/A"
        cdict['current_week']   = intervention['current_week'] if intervention else 0
        cdict['start_date']     = intervention['start_date'] if intervention else None

        child_data_list.append(cdict)

    conn.close()

    return render_template('parent_dashboard.html',
                           parent=parent,
                           children=child_data_list)


@app.route('/admin/add_patient_assessment', methods=['GET', 'POST'])
@admin_required
def admin_manual_assessment():
    if request.method == 'POST':
        # 1. Get Form Data
        child_name = request.form.get('child_name')
        child_age = request.form.get('child_age')
        
        # Assessment data
        reference_text = request.form.get('reference_text')
        spoken_text = request.form.get('spoken_text')
        handwriting_image_path = 'static/img/placeholder_handwriting.png' 

        # 2. Perform Detection 
        dyslexia_score = calculate_dyslexia_score(spoken_text, reference_text)
        dysgraphia_label, dysgraphia_score = predict_handwriting(handwriting_image_path)
        
        # 3. Save Child Record (If not already exists) and Assessment
        conn = get_db_connection()
        
        child_row = conn.execute("SELECT id FROM children WHERE name = ?", (child_name,)).fetchone()
        
        if not child_row:
            # Get/Create Admin Parent Placeholder
            parent_row = conn.execute("SELECT id FROM parents WHERE username = 'admin_user'").fetchone()
            if not parent_row:
                conn.execute("INSERT INTO parents (username, password_hash, email) VALUES (?, ?, ?)", 
                             ('admin_user', generate_password_hash('password'), 'admin@example.com'))
                parent_row = conn.execute("SELECT id FROM parents WHERE username = 'admin_user'").fetchone()
            
            # Insert new child
            conn.execute("INSERT INTO children (parent_id, name, age) VALUES (?, ?, ?)", 
                         (parent_row[0], child_name, child_age))
            child_row = conn.execute("SELECT id FROM children WHERE name = ?", (child_name,)).fetchone()

        # 4. Save Assessment Result
        child_id = child_row[0]
        verdict = f"Dyslexia: {dyslexia_score:.2f}, Dysgraphia: {dysgraphia_score:.2f}" # Simple combined verdict
        
        conn.execute("""
            INSERT INTO assessments (child_id, screening_type, dyslexia_score, dysgraphia_score, raw_data_path, reference_text, verdict)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (child_id, 'Manual Admin', dyslexia_score, dysgraphia_score, handwriting_image_path, reference_text, verdict))
        conn.commit()
        conn.close()

        flash(f"Manual assessment for {child_name} saved.", 'success')
        # Redirect to the admin's detailed report view
        return redirect(url_for('child_progress_report', child_id=child_id)) 

    # Assumes admin_manual_assessment.html exists
    return render_template('admin_manual_assessment.html', sentences=get_sentences(3))


@app.route('/child/<int:child_id>/report')
@admin_required
def child_progress_report(child_id):
    """Admin route for the detailed comprehensive report."""
    conn = get_db_connection()
    
    # 1. Fetch Child Details
    child = conn.execute("SELECT id, name, age FROM children WHERE id = ?", (child_id,)).fetchone()
    if not child:
        conn.close()
        flash('Child not found.', 'error')
        return redirect(url_for('admin_dashboard'))
    child = dict(child) 

    # 2. Fetch All Assessments (History)
    all_assessments = conn.execute("""
        SELECT 
            dyslexia_score, 
            dysgraphia_score, 
            raw_data_path, 
            created_at,
            reference_text 
        FROM assessments 
        WHERE child_id = ? 
        ORDER BY created_at DESC
    """, (child_id,)).fetchall()
    
    all_assessments = [dict(a) for a in all_assessments]

    # 3. Get Latest Assessment for Goal Generation
    latest_assessment = all_assessments[0] if all_assessments else None
    
    conn.close()

    # 4. Render Template
    return render_template(
        'child_progress.html',
        child=child,
        latest_assessment=latest_assessment,
        all_assessments=all_assessments,
        # Pass the Gemini function directly to the template
        generate_personalized_curriculum=generate_personalized_curriculum 
    )


# --- Parent's simpler progress view ---
@app.route('/parent/child_progress/<int:child_id>')
@parent_required
def child_progress(child_id):
    conn = get_db_connection()
    
    # 1. Fetch Parent info (To fix the "parent is undefined" error)
    parent = conn.execute("SELECT username FROM parents WHERE id = ?", (session['parent_id'],)).fetchone()
    
    # 2. Security check for parent access
    child_check = conn.execute("SELECT parent_id, name, age FROM children WHERE id = ?", (child_id,)).fetchone()
    
    if not child_check or child_check['parent_id'] != session['parent_id']:
          flash("Unauthorized access.", "error")
          conn.close()
          return redirect(url_for('parent_dashboard'))
    
    child = dict(child_check)
    child['id'] = child_id

    # 3. Fetch assessments
    all_assessments = conn.execute("""
        SELECT dyslexia_score, dysgraphia_score, raw_data_path, created_at, reference_text 
        FROM assessments WHERE child_id = ? ORDER BY created_at DESC
    """, (child_id,)).fetchall()
    
    all_assessments = [dict(a) for a in all_assessments]
    latest_assessment = all_assessments[0] if all_assessments else None
    conn.close()
    print()
    return render_template(
        'child_progress.html',
        parent=parent, # <--- ADD THIS LINE
        child=child,
        latest_assessment=latest_assessment,
        all_assessments=all_assessments,
        generate_personalized_curriculum=generate_personalized_curriculum 
    )


# ==============================================================================
# 4. ASSESSMENT FLOW ROUTES 
# ==============================================================================

@app.route('/select_child')
@parent_required
def select_child():
    parent_id = session['parent_id']
    conn = get_db_connection()
    children = conn.execute("SELECT id, name FROM children WHERE parent_id = ?", (parent_id,)).fetchall()
    conn.close()
    return render_template('select_child.html', children=children)

@app.route('/start_assessment/<int:child_id>', methods=['GET'])
@parent_required
def start_assessment(child_id):
    conn = get_db_connection()
    child = conn.execute("SELECT name, parent_id FROM children WHERE id = ?", (child_id,)).fetchone()
    conn.close()
    
    if not child or child['parent_id'] != session['parent_id']:
        flash("Child not found or unauthorized.", "error")
        return redirect(url_for('parent_dashboard'))
        
    session['current_child_id'] = child_id
    session['current_child_name'] = child['name']
    flash(f"Starting assessment for {child['name']}. Please select a screening type.", "info")
    return render_template('screening_choice.html')

# --- 4.1. SPEECH DETECTION (Dyslexia) ---

@app.route('/start_speech', methods=['POST'])
@parent_required
def start_speech():
    sentences = get_sentences() # Get 3 random sentences
    session['speech_prompts'] = sentences
    session['speech_responses'] = []
    session['speech_scores'] = []
    session['speech_current_idx'] = 0
    return render_template('reading.html', 
        prompt=session['speech_prompts'][0], 
        prompt_num=1, total=len(sentences),
        child_name=session['current_child_name'])

@app.route('/submit_speech_response', methods=['POST'])
@parent_required
def submit_speech_response():
    data = request.get_json()
    spoken = data.get('spoken_text', '')
    
    idx = session.get('speech_current_idx', 0)
    prompts = session.get('speech_prompts', [])
    
    if idx >= len(prompts):
        return jsonify({'error': 'Invalid state or assessment finished.'}), 400
        
    target = prompts[idx]
    score = calculate_dyslexia_score(spoken, target) # 0-1 normalized score
    
    session['speech_responses'].append(spoken)
    session['speech_scores'].append(score)
    session['speech_current_idx'] = idx + 1

    if session['speech_current_idx'] < len(prompts):
        return jsonify({
            'next': True,
            'prompt': session['speech_prompts'][session['speech_current_idx']],
            'prompt_num': session['speech_current_idx'] + 1,
            'total': len(prompts)
        })
    else:
        avg_score = sum(session['speech_scores']) / len(session['speech_scores'])
        verdict = "Likelihood of Dyslexia Detected" if avg_score >= DYSLEXIA_SCORE_THRESHOLD else "No significant signs"
        
        conn = get_db_connection()
        # Save the full reference text used for the assessment
        reference_text_used = " | ".join(prompts) 
        
        conn.execute("""
            INSERT INTO assessments (child_id, parent_id, screening_type, dyslexia_score, verdict, reference_text)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session['current_child_id'], session['parent_id'], 'Speech', avg_score, verdict, reference_text_used))
        conn.commit()
        conn.close()
        
        # Clear session data
        session.pop('speech_prompts', None)
        session.pop('speech_responses', None)
        session.pop('speech_scores', None)
        session.pop('speech_current_idx', None)

        return jsonify({'next': False, 'avg_score': f"{avg_score:.2f}", 'verdict': verdict, 'redirect': url_for('parent_dashboard')})


# --- 4.2. HANDWRITING DETECTION (Dysgraphia) ---

@app.route('/handwriting')
@parent_required
def handwriting_upload():
    if 'current_child_name' not in session:
        flash("Please select a child first.", "error")
        return redirect(url_for('select_child'))
    return render_template('handwriting_upload.html', child_name=session['current_child_name'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload_handwriting', methods=['POST'])
@parent_required
def upload_handwriting():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('handwriting_upload'))
        
    file = request.files['file']
    if file.filename == '' or not allowed_file(file.filename):
        flash('No selected file or file type not allowed.', 'error')
        return redirect(url_for('handwriting_upload'))

    filename = secure_filename(file.filename)
    unique_filename = f"{session['current_child_id']}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(filepath)
    
    try:
        # 🧠 Core Detection Logic
        label, conf = predict_handwriting(filepath)
        
        # Score normalization (assuming 'conf' is confidence in the detected label)
        dysgraphia_score = conf if 'Dysgraphia' in label else (1 - conf) 
        verdict = label
        
        conn = get_db_connection()
        conn.execute("""
            INSERT INTO assessments (child_id, parent_id, screening_type, dysgraphia_score, verdict, raw_data_path)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (session['current_child_id'], session['parent_id'], 'Handwriting', dysgraphia_score, verdict, filepath))
        conn.commit()
        conn.close()
        
        flash(f"Handwriting analyzed. Result: {verdict} (Score: {dysgraphia_score:.2f})", "success")
        return render_template('handwriting_result.html', label=verdict, confidence=f"{conf*100:.2f}%", 
                               image_url=url_for('static', filename=f'uploads/{unique_filename}'), 
                               child_id=session['current_child_id'])
        
    except Exception as e:
        flash(f"Error processing image: {str(e)}", "error")
        if os.path.exists(filepath):
            os.remove(filepath)
        return redirect(url_for('handwriting_upload'))

# ==============================================================================
# 5. INTERVENTION AND PROGRESS ROUTES (Gemini AI)
# ==============================================================================
from services.services import generate_full_4_week_curriculum
import json
from datetime import datetime

@app.route('/generate_goals/<int:child_id>', methods=['POST'])
@parent_required
def generate_goals(child_id):
    conn = get_db_connection()
    
    # 1. Fetch Child & Assessment Data (Needed to re-render the progress page)
    child = conn.execute("SELECT * FROM children WHERE id = ?", (child_id,)).fetchone()
    latest_assessment = conn.execute("""
        SELECT * FROM assessments WHERE child_id = ? 
        ORDER BY created_at DESC LIMIT 1
    """, (child_id,)).fetchone()
    all_assessments = conn.execute("SELECT * FROM assessments WHERE child_id = ? ORDER BY created_at DESC", (child_id,)).fetchall()

    if not latest_assessment:
        flash("No assessment data found. Please complete a screening test first.", "error")
        conn.close()
        return redirect(url_for('child_progress', child_id=child_id))

    # 2. Generate 4-week plan using Gemini
    try:
        # Call the AI service function
        full_curriculum_dict = generate_full_4_week_curriculum(
            child['name'], 
            latest_assessment['dyslexia_score'], 
            latest_assessment['dysgraphia_score']
        )
        
        # Flash a success message
        flash("✅ 4-Week AI Curriculum generated successfully!", "success")
        print(full_curriculum_dict)
        # 3. Render the progress template directly with the generated data
        # We pass 'generated_curriculum' so the HTML knows to show it
        return render_template(
            'child_progress.html', 
            child=child, 
            latest_assessment=latest_assessment,
            all_assessments=all_assessments,
            intervention={'curriculum': full_curriculum_dict, 'current_week': 1}, # Mocking the structure
            progress_percent=0,
            is_preview=True # Flag to indicate this isn't saved yet
        )

    except Exception as e:
        print(f"Gemini Error: {e}")
        flash("AI Service is currently busy. Please try again in a moment.", "danger")
        return redirect(url_for('child_progress', child_id=child_id))
    finally:
        conn.close()

    return redirect(url_for('child_progress', child_id=child_id))

@app.route('/complete_week/<int:child_id>/<int:week_num>', methods=['POST'])
@parent_required
def complete_week(child_id, week_num):
    conn = get_db_connection()
    intervention = conn.execute("SELECT id, current_week FROM interventions WHERE child_id = ? ORDER BY start_date DESC LIMIT 1", (child_id,)).fetchone()
    
    if not intervention:
        flash("No intervention plan found.", "error")
        conn.close()
        return redirect(url_for('child_progress', child_id=child_id))

    intervention_id = intervention['id']
    current_week = intervention['current_week']

    if week_num == current_week:
        next_week = current_week + 1
        
        conn.execute("""
            UPDATE interventions 
            SET current_week = ?, last_completed_at = ?
            WHERE id = ?
        """, (next_week, datetime.now(), intervention_id))
        conn.commit()
        
        if next_week <= 4:
            flash(f"🎉 Week {current_week} completed! Week {next_week} is now scheduled.", "success")
        else:
            flash("✨ Congratulations! The 4-Week Intervention Cycle is complete. Please perform a new assessment.", "success")
    elif week_num < current_week:
        flash(f"Week {week_num} is already completed.", "warning")
    else:
        flash(f"You must complete Week {current_week} before starting Week {week_num}.", "error")

    conn.close()
    return redirect(url_for('child_progress', child_id=child_id))


@app.route('/export_csv')
@admin_required
def export_csv():
    conn = get_db_connection()
    c = conn.cursor()
    c.execute("""
        SELECT 
            c.id, c.name, c.age, p.username AS parent_name, 
            a.screening_type, a.verdict, a.dyslexia_score, a.dysgraphia_score, a.created_at
        FROM children c
        JOIN parents p ON c.parent_id = p.id
        LEFT JOIN assessments a ON c.id = a.child_id
        ORDER BY c.id, a.created_at DESC
    """)
    rows = c.fetchall()
    conn.close()

    output = os.path.join(app.root_path, 'assessment_export.csv')
    with open(output, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Child ID', 'Name', 'Age', 'Parent', 'Screening Type', 'Verdict', 'Dyslexia Score', 'Dysgraphia Score', 'Created At'])
        writer.writerows([tuple(row) for row in rows])
    return send_file(output, as_attachment=True)


# ==============================================================================
# 6. MAIN EXECUTION
# ==============================================================================
# --- Add Child Route ---
@app.route('/parent/add_child', methods=['GET', 'POST'])
@parent_required
def add_child():
    if request.method == 'POST':
        name = request.form.get('name')
        age = request.form.get('age')
        school_class = request.form.get('school_class')
        parent_id = session['parent_id']

        if not name or not age:
            flash("Child's name and age are required.", "danger")
            return redirect(url_for('add_child'))

        conn = get_db_connection()
        try:
            conn.execute('''
                INSERT INTO children (parent_id, name, age, school_class)
                VALUES (?, ?, ?, ?)
            ''', (parent_id, name, age, school_class))
            conn.commit()
            flash(f"Success! {name} has been added to your dashboard.", "success")
            return redirect(url_for('parent_dashboard'))
        except Exception as e:
            flash(f"An error occurred: {str(e)}", "danger")
        finally:
            conn.close()

    return render_template('add_child.html')

# ==============================================================================
# 8. STATIC TO-DO CURRICULUM ROUTES (Separate Database)
# ==============================================================================

@app.route('/todo/start/<int:child_id>', methods=['POST'])
@parent_required
def start_todo_curriculum(child_id):
    """Start a static to-do curriculum for a child"""
    # Get latest scores from main database
    conn = get_db_connection()
    latest_scores = conn.execute("""
        SELECT dyslexia_score, dysgraphia_score 
        FROM assessments 
        WHERE child_id = ? 
        ORDER BY created_at DESC LIMIT 1
    """, (child_id,)).fetchone()
    conn.close()
    
    if not latest_scores:
        flash("No assessment data found. Please complete screening first.", "error")
        return redirect(url_for('child_progress', child_id=child_id))
    
    # Determine if curriculum is needed
    difficulty = get_difficulty_from_scores(
        latest_scores['dyslexia_score'], 
        latest_scores['dysgraphia_score']
    )
    
    # If difficulty is None (scores < 0.3), show congratulations message
    if difficulty is None:
        conn = get_db_connection()
        child = conn.execute("SELECT name FROM children WHERE id = ?", (child_id,)).fetchone()
        conn.close()
        
        flash(f"🎉 Great news! {child['name']}'s scores are excellent! No special curriculum needed - they're doing great!", "success")
        return redirect(url_for('child_progress', child_id=child_id))
    
    # Initialize to-do database (only for medium/high difficulty)
    todo_conn = get_todo_db_connection()
    
    # Check if already has active curriculum
    existing = todo_conn.execute('''
        SELECT id FROM child_curriculum 
        WHERE child_id = ? AND is_active = 1
    ''', (child_id,)).fetchone()
    
    if existing:
        flash("To-do curriculum already active for this child.", "info")
        todo_conn.close()
        return redirect(url_for('view_todo_curriculum', child_id=child_id))
    
    # Assign curriculum to child
    todo_conn.execute('''
        INSERT INTO child_curriculum (child_id, difficulty_level)
        VALUES (?, ?)
    ''', (child_id, difficulty))
    
    # Get activities for this difficulty
    activities = todo_conn.execute('''
        SELECT id, week_number, day_number 
        FROM static_curriculum 
        WHERE difficulty_level = ?
        ORDER BY week_number, day_number
    ''', (difficulty,)).fetchall()
    
    # Initialize progress tracking
    for activity in activities:
        todo_conn.execute('''
            INSERT INTO child_progress_tracking 
            (child_id, week_number, day_number, activity_id)
            VALUES (?, ?, ?, ?)
        ''', (child_id, activity['week_number'], activity['day_number'], activity['id']))
    
    todo_conn.commit()
    
    # Get child name for message
    conn = get_db_connection()
    child = conn.execute("SELECT name FROM children WHERE id = ?", (child_id,)).fetchone()
    conn.close()
    
    flash(f"✅ {difficulty.capitalize()}-difficulty curriculum started for {child['name']}!", "success")
    return redirect(url_for('view_todo_curriculum', child_id=child_id))

@app.route('/todo/<int:child_id>')
@parent_required
def view_todo_curriculum(child_id):
    """View the to-do curriculum"""
    # Security check
    conn = get_db_connection()
    child_check = conn.execute("SELECT parent_id, name, age FROM children WHERE id = ?", (child_id,)).fetchone()
    conn.close()
    
    if not child_check or child_check['parent_id'] != session['parent_id']:
        flash("Unauthorized access.", "error")
        return redirect(url_for('parent_dashboard'))
    
    # Get to-do data
    todo_data = get_child_todo_data(child_id)
    
    if not todo_data:
        # Get latest scores to determine if curriculum is needed
        conn = get_db_connection()
        latest_scores = conn.execute("""
            SELECT dyslexia_score, dysgraphia_score 
            FROM assessments 
            WHERE child_id = ? 
            ORDER BY created_at DESC LIMIT 1
        """, (child_id,)).fetchone()
        conn.close()
        
        if latest_scores:
            difficulty = get_difficulty_from_scores(
                latest_scores['dyslexia_score'], 
                latest_scores['dysgraphia_score']
            )
            
            if difficulty is None:
                # Show congratulations page (no curriculum needed)
                return render_template('todo_congratulations.html',
                                     child_id=child_id,
                                     child_name=child_check['name'],
                                     child_age=child_check['age'],
                                     dyslexia_score=latest_scores['dyslexia_score'],
                                     dysgraphia_score=latest_scores['dysgraphia_score'])
            else:
                # Show start curriculum page
                todo_data = {'difficulty_preview': difficulty}
        else:
            # No assessment data
            return render_template('todo_no_assessment.html',
                                 child_id=child_id,
                                 child_name=child_check['name'])
    
    # If todo_data exists, show the curriculum
    return render_template('todo_curriculum.html',
                         child_id=child_id,
                         child_name=child_check['name'],
                         child_age=child_check['age'],
                         todo_data=todo_data)

@app.route('/todo/toggle/<int:child_id>/<int:activity_id>', methods=['POST'])
@parent_required
def toggle_todo_activity(child_id, activity_id):
    """Toggle activity completion status"""
    # Security check
    conn = get_db_connection()
    child_check = conn.execute("SELECT parent_id FROM children WHERE id = ?", (child_id,)).fetchone()
    conn.close()
    
    if not child_check or child_check['parent_id'] != session['parent_id']:
        return jsonify({'success': False, 'error': 'Unauthorized'})
    
    todo_conn = get_todo_db_connection()
    
    # Get current status
    progress = todo_conn.execute('''
        SELECT id, completed FROM child_progress_tracking 
        WHERE child_id = ? AND activity_id = ?
        ORDER BY id DESC LIMIT 1
    ''', (child_id, activity_id)).fetchone()
    
    if not progress:
        todo_conn.close()
        return jsonify({'success': False, 'error': 'Activity not found'})
    
    # Toggle status
    new_status = not progress['completed']
    
    todo_conn.execute('''
        UPDATE child_progress_tracking 
        SET completed = ?, completed_at = ?
        WHERE id = ?
    ''', (new_status, datetime.now() if new_status else None, progress['id']))
    todo_conn.commit()
    
    # Get updated stats
    todo_data = get_child_todo_data(child_id)
    todo_conn.close()
    
    return jsonify({
        'success': True,
        'completed': new_status,
        'progress_percent': todo_data['progress_percent'],
        'completed_activities': todo_data['completed_activities'],
        'total_activities': todo_data['total_activities']
    })

@app.route('/todo/reset/<int:child_id>', methods=['POST'])
@parent_required
def reset_todo_curriculum(child_id):
    """Reset to-do curriculum progress"""
    # Security check
    conn = get_db_connection()
    child_check = conn.execute("SELECT parent_id FROM children WHERE id = ?", (child_id,)).fetchone()
    conn.close()
    
    if not child_check or child_check['parent_id'] != session['parent_id']:
        flash("Unauthorized.", "error")
        return redirect(url_for('parent_dashboard'))
    
    todo_conn = get_todo_db_connection()
    
    # Reset all progress
    todo_conn.execute('''
        UPDATE child_progress_tracking 
        SET completed = 0, completed_at = NULL
        WHERE child_id = ?
    ''', (child_id,))
    
    # Mark curriculum as inactive and create new one
    todo_conn.execute('''
        UPDATE child_curriculum SET is_active = 0 WHERE child_id = ?
    ''', (child_id,))
    
    todo_conn.commit()
    todo_conn.close()
    
    flash("To-do curriculum progress reset. You can start a new one.", "info")
    return redirect(url_for('child_progress', child_id=child_id))

# ==============================================================================
# 9. LEGAL & SUPPORT PAGES
# ==============================================================================

@app.route('/privacy-policy')
def privacy_policy():
    """Privacy policy page"""
    return render_template('privacy_policy.html')

@app.route('/terms-of-service')
def terms_of_service():
    """Terms of service page"""
    return render_template('terms_of_service.html')

@app.route('/contact-support')
def contact_support():
    """Contact support page"""
    return render_template('contact_support.html')


if __name__ == "__main__":
    init_db()
    init_todo_db()
    app.run(debug=True)