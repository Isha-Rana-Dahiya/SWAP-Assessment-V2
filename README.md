# SWAP-Assessment-V2

Use Streamlit for the UI and Python for the backend. We will add an LLM for generating interpretations. We can use AI for generating a textual interpretation of the scores. We'll use an open-source LLM via Hugging Face or a free API. 

   - With LLM (for interpretation)
   - We can use a free LLM API
   - We can also use a local LLM with llama.cpp 
 We'll use an LLM for generating an interpretation of the scores.
 However, to keep it simple and free, we can use Hugging Face's free Inference API for a small model. We need a Hugging Face account (free). 
 
We can use a local model with llama.cpp
 With AI: We use an LLM to generate an interpretation of the scores.
 Steps for Hugging Face:
   - Create an account on Hugging Face.
   -  - Get an access token (free).
   - Use the `huggingface_hub` library to call the API.
 We'll use a model like "google/flan-t5-small" for text generation.
 Use the Hugging Face Inference API.
 Steps:
 1. Sign up at huggingface.co
 2. Go to settings -> Access Tokens -> create a token (with read access)
 3. Save the token in a `.env` file
    
 We'll create a function that takes the wellness and insomnia scores and returns an interpretation.
 Considering all, I'll provide two versions:
 Version A (with Hugging Face Inference API - requires internet and token):
   - We'll call the API to generate an interpretation.
   
    Therefore, I'll go with Hugging Face Inference API.
 Implementation:
   - We'll create a function `get_interpretation(wellness_score, insomnia_score)` that sends a prompt to the model and returns the response.
   - We'll use the `requests` library to call the Hugging Face Inference API.
   - Model: Let's use "google/flan-t5-base" (or small) because it's good for instruction following.
   - Prompt: "Interpret these wellness assessment scores: Wellness Score: {wellness_score}/24, Insomnia Score: {insomnia_score}/24. Provide a brief professional interpretation in 2 sentences."
   - We'll cache the response to avoid repeated calls for same scores.
   
 Steps:
 1. Install libraries:
    ```
    pip install streamlit pandas python-dotenv requests
     Create a `.env` file with:
    ```
    HF_API_TOKEN=your_token_here
    ```
 3. In the code, load the token and make the API call.
 
 Structure the code:
   - Import libraries
   -  Load environment variables
   - Define all 20 questions (with id, text, Thai text, and options)
   - Streamlit app layout
   - Function to calculate scores
   - Function to get interpretation from Hugging Face model
 Code for `app.py`:
We'll use:
Gradio for the UI (simpler than Streamlit)

Hugging Face Inference API for AI interpretation (no local LLM setup)

Python for the backend

Here's the step-by-step implementation:

Step 1: Install Required Libraries
Python:
pip install gradio requests python-dotenv

Step 2: Create Project Files
Create a folder named wellness-assessment

Inside it, create these files:

app.py (main application)

.env (for API keys)

questions.py (question data)

Step 3: Get Free AI API Key
1. Go to Hugging Face and create a free account
2. Go to API Tokens and create a new token
3. Copy the token

Step 4: Set Up Environment (.env)
HF_API_TOKEN=your_huggingface_token_here

Step 5: Create Question Data (questions.py)
QUESTIONS = [
    # Wellness Questions (1-12)
    {
        "id": 1,
        "text": "Have you been able to concentrate on what you're doing?",
        "thai": "ในระยะสองถึงสามสัปดาห์ที่ผ่านมานี้ทำนสามารถมีสมาวิจดต่อกับสิ่งที่กำลังทำอยู่ได้",
        "options": [
            "Better than usual (ดีกว่าปกติ)",
            "Same as usual (เหมือนปกติ)",
            "Less than usual (น้อยกว่าปกติ)",
            "Much less than usual (น้อยกว่าปกติมาก)"
        ]
    },
    {
        "id": 2,
        "text": "Have you recently lost much sleep over worry?",
        "thai": "ในระยะสองถึงสามสัปดาห์ที่ผ่านมานี้ท่านนอนไม่หลับเพราะกังวลใจ",
        "options": [
            "Not at all (ไมเลย)",
            "No more than usual (ไมมากกว่าปกติ)",
            "Rather more than usual (ค่อนข้างมากกว่าปกติ)",
            "Much more than usual (มากกว่าปกติมาก)"
        ]
    },
    # Add all 20 questions in this format
    # ...
    
    # Example of insomnia question (13-20)
    {
        "id": 13,
        "text": "Sleep Induction (time to fall asleep)",
        "thai": "การเข้านอน (เวลาดั้งแต่ปิดไฟจนท่านหลับ)",
        "options": [
            "No problem (ไม่มีปัญหา)",
            "Slightly delayed (ข้ามลิกน้อย)",
            "Markedly delayed (ข้ามาก)",
            "Very delayed or did not sleep at all (ข้ามากที่สุดหรือไม่หลับเลย)"
        ]
    },
    # Continue adding questions...
]

# Scoring rules
SCORING = {
    "wellness": {
        "range": (1, 12),
        "points": {0: 0, 1: 0, 2: 2, 3: 2}
    },
    "insomnia": {
        "range": (13, 20),
        "points": {0: 1, 1: 1, 2: 3, 3: 3}
    }
}

Step 6: Create Main Application (app.py)
import gradio as gr
import requests
import os
from dotenv import load_dotenv
from questions import QUESTIONS, SCORING

# Load Hugging Face API token
load_dotenv()
HF_TOKEN = os.getenv("HF_API_TOKEN")
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"

# Calculate scores
def calculate_scores(answers):
    wellness_score = 0
    insomnia_score = 0
    
    # Calculate wellness score (Q1-12)
    for q_id in range(SCORING["wellness"]["range"][0], SCORING["wellness"]["range"][1] + 1):
        answer_idx = answers.get(f"q{q_id}", 0)
        points = SCORING["wellness"]["points"].get(answer_idx, 0)
        wellness_score += points
    
    # Calculate insomnia score (Q13-20)
    for q_id in range(SCORING["insomnia"]["range"][0], SCORING["insomnia"]["range"][1] + 1):
        answer_idx = answers.get(f"q{q_id}", 0)
        points = SCORING["insomnia"]["points"].get(answer_idx, 1)
        insomnia_score += points
    
    return wellness_score, insomnia_score

# Get AI interpretation
def get_ai_interpretation(wellness, insomnia):
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    prompt = f"""
    Interpret these wellness assessment scores:
    - Wellness Score: {wellness}/24 (0-12: normal, 13-24: concern)
    - Insomnia Score: {insomnia}/24 (0-7: normal, 8-24: concern)
    
    Provide professional interpretation in 2 sentences. Consider cultural sensitivity for Thai employees.
    """
    
    response = requests.post(API_URL, headers=headers, json={"inputs": prompt})
    try:
        return response.json()[0]['generated_text']
    except:
        return "AI interpretation unavailable. Please check your API key."

# Create form UI
def create_form():
    with gr.Blocks(title="Staff Wellness Assessment") as demo:
        gr.Markdown("# 🩺 Doctor Heals Staff Wellness Assessment")
        gr.Markdown("กรุณาตอบคำถามต่อไปนี้ กรุณาเลือกช่องคำตอบที่ใกล้เคียงกับสภาพของท่านในช่วงสามสัปดาห์ที่ผ่านมามากที่สุด")
        
        # Personal information
        with gr.Row():
            first_name = gr.Textbox(label="First Name (ชื่อจริง)")
            last_name = gr.Textbox(label="Last Name (นามสกุล)")
        company = gr.Textbox(label="Company Name (ชื่อบริษัท)")
        email = gr.Textbox(label="Personal Email (อีเมล)")
        
        # Questions
        answers = {}
        for q in QUESTIONS:
            with gr.Group():
                gr.Markdown(f"### Question {q['id']}: {q['text']}")
                gr.Markdown(q['thai'])
                answers[f"q{q['id']}"] = gr.Radio(
                    choices=q["options"],
                    label="Select your answer",
                    type="index"
                )
        
        # Submit button
        submit_btn = gr.Button("Submit Assessment")
        
        # Results
        wellness_score = gr.Number(label="Wellness Score", visible=False)
        insomnia_score = gr.Number(label="Insomnia Score", visible=False)
        interpretation = gr.Textbox(label="AI Interpretation", visible=False)
        
        # Results display
        with gr.Column(visible=False) as results_section:
            gr.Markdown("## Assessment Results")
            wellness_display = gr.Markdown()
            insomnia_display = gr.Markdown()
            ai_interpretation = gr.Markdown()
        
        # Submit handler
        def submit_form(*args):
            # Extract answers
            answer_dict = {}
            for i, q in enumerate(QUESTIONS):
                answer_dict[f"q{q['id']}"] = args[i]
            
            # Calculate scores
            wellness, insomnia = calculate_scores(answer_dict)
            
            # Get AI interpretation
            ai_text = get_ai_interpretation(wellness, insomnia)
            
            return {
                wellness_score: wellness,
                insomnia_score: insomnia,
                interpretation: ai_text,
                results_section: gr.update(visible=True),
                wellness_display: f"**Wellness Score:** {wellness}/24",
                insomnia_display: f"**Insomnia Score:** {insomnia}/24",
                ai_interpretation: f"**AI Interpretation:**\n\n{ai_text}"
            }
        
        submit_btn.click(
            submit_form,
            inputs=list(answers.values()),
            outputs=[
                wellness_score, 
                insomnia_score, 
                interpretation,
                results_section,
                wellness_display,
                insomnia_display,
                ai_interpretation
            ]
        )
        
        # Privacy notice
        gr.Markdown("---")
        gr.Markdown("""
        **Privacy Notice:**  
        By submitting this form, you consent to Doctor Heals contacting you via provided details.
        We ensure confidentiality per Singapore's PDPA.
        """)
    
    return demo

# Run the application
if __name__ == "__main__":
    app = create_form()
    app.launch(server_name="0.0.0.0", server_port=7860)


  Step 7: Run the Application
  python app.py

  Step 8: Access the Application
The app will run at http://localhost:7860

Open this URL in your web browser

Fill out the form and submit to see scores with AI interpretation

Key Features:
1. Automatic Scoring:

Wellness Score (Q1-12)

Insomnia Score (Q13-20)

2. AI-Powered Interpretation:

Uses Google's Flan-T5 model via Hugging Face

Provides culturally-sensitive feedback

Free tier sufficient for moderate usage

3. Bilingual Interface:

English and Thai support

Clean, professional design
4. Privacy Compliant:

Follows Singapore's PDPA

No data stored permanently

Customization Options:
1. To add all 20 questions:

- Complete the QUESTIONS list in questions.py

- Follow the same dictionary format

2. To change AI model:

Replace model name in API_URL:
Python:
# Other free models:
# "facebook/bart-large-cnn"
# "google/flan-t5-xl"
API_URL = "https://api-inference.huggingface.co/models/google/flan-t5-large"

To deploy online:

Create free Hugging Face Space: https://huggingface.co/spaces

Upload all files

Add HF_API_TOKEN in Space settings

How It Works:
1. User completes the form in their browser
2. Python calculates scores based on your rules
3. Scores are sent to Hugging Face AI for interpretation
4. Results display with scores and AI-generated feedback
