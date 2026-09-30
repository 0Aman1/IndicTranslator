import gradio as gr
import spaces
import json
import tempfile
import pandas as pd
from datetime import datetime
from multilingual_nlp import MultilingualNLPSystem

# Initialize the NLP system
# Note: ZeroGPU will dynamically allocate GPU power when the decorated function is called.
nlp_system = MultilingualNLPSystem()

# Language choices matching your Streamlit app
LANGUAGE_CHOICES = [
    ("English", "en"),
    ("Hindi (हिन्दी)", "hi"),
    ("Tamil (தமிழ்)", "ta"),
    ("Marathi (मराठी)", "mr"),
    ("Gujarati (ગુજરાતી)", "gu"),
    ("Punjabi (ਪੰਜਾਬੀ)", "pa"),
    ("Kannada (ಕನ್ನಡ)", "kn"),
    ("Malayalam (മലയാളം)", "ml"),
    ("Bengali (বাংলা)", "bn")
]

# The @spaces.GPU decorator gives this specific function access to the A100 GPU
@spaces.GPU
def analyze_and_export(text, target_lang):
    if not text.strip():
        raise gr.Error("Please enter some text to analyze.")
        
    try:
        # 1. Process Text using your custom module
        result = nlp_system.process_text(
            text=text,
            target_languages=[target_lang]
        )
        
        # 2. Extract Display Data
        detected_lang = result.get('detected_language', 'Unknown')
        
        sentiment_label = result.get('sentiment', 'Neutral').title()
        sentiment_score = result.get('confidence', 0.0)
        sentiment_display = f"{sentiment_label} (Confidence: {sentiment_score:.1%})"
        
        translated_text = result.get('translations', {}).get(target_lang, "")
        romanized_text = result.get('translations_romanized', {}).get(target_lang, "")
        
        # Handle same-language fallback
        if not translated_text and detected_lang == target_lang:
            translated_text = result.get('native_script', text)
            romanized_text = result.get('native_script_romanized', "")
            
        # 3. Generate Export Files (JSON and CSV)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # JSON Export
        json_file_path = tempfile.mktemp(suffix=".json", prefix=f"analysis_{timestamp}_")
        with open(json_file_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
            
        # CSV Export
        csv_file_path = tempfile.mktemp(suffix=".csv", prefix=f"analysis_{timestamp}_")
        csv_data = pd.DataFrame([{
            'Original Text': text,
            'Detected Language': detected_lang,
            'Sentiment': sentiment_label,
            'Sentiment Score': sentiment_score,
            'Target Language': target_lang,
            'Translated Text': translated_text or "N/A",
            'Translated Romanized': romanized_text or "N/A",
            'Timestamp': timestamp
        }])
        csv_data.to_csv(csv_file_path, index=False)
        
        return (
            detected_lang.upper(),
            sentiment_display,
            translated_text,
            romanized_text,
            result,           # Raw JSON for the UI
            json_file_path,   # File 1
            csv_file_path     # File 2
        )
        
    except Exception as e:
        raise gr.Error(f"Error during analysis: {str(e)}")

# Build the Gradio Interface
with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue", secondary_hue="indigo")) as demo:
    
    gr.Markdown("# 🌐 Multilingual NLP Analysis")
    gr.Markdown("Process text in multiple languages with sentiment analysis and translation, accelerated by Hugging Face ZeroGPU.")
    
    with gr.Row():
        # Left Column: Inputs
        with gr.Column(scale=1):
            gr.Markdown("### Configuration")
            input_text = gr.Textbox(
                lines=6, 
                placeholder="Enter text in English, Hindi, Tamil, Hinglish, or Tanglish...", 
                label="Input Text"
            )
            target_lang_dropdown = gr.Dropdown(
                choices=LANGUAGE_CHOICES, 
                value="en", 
                label="Target Language"
            )
            analyze_btn = gr.Button("🔍 Analyze Text", variant="primary")
            
            with gr.Accordion("🚀 System Capabilities", open=False):
                gr.Markdown("""
                * **🌍 Language Detection:** Auto-detects 9+ Indic languages and English.
                * **😊 Sentiment Analysis:** Real-time emotion detection.
                * **🔄 Translation:** Accurate cross-language translation.
                * **📱 Romanization:** Phonetic representation in English script.
                """)

        # Right Column: Outputs
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Analysis Dashboard")
            
            with gr.Row():
                out_detected = gr.Textbox(label="🌍 Detected Language", interactive=False)
                out_sentiment = gr.Textbox(label="😊 Sentiment", interactive=False)
                
            out_translation = gr.Textbox(lines=3, label="🔄 Translated Text", interactive=False)
            out_romanized = gr.Textbox(lines=2, label="📱 Phonetic Romanization (English Script)", interactive=False)
            
            with gr.Accordion("📄 Raw JSON & Confidence Details", open=False):
                out_json = gr.JSON(label="System Output")
                
            gr.Markdown("### 📤 Export Results")
            with gr.Row():
                out_file_json = gr.File(label="Download JSON")
                out_file_csv = gr.File(label="Download CSV")

    # Connect the button to the function
    analyze_btn.click(
        fn=analyze_and_export,
        inputs=[input_text, target_lang_dropdown],
        outputs=[
            out_detected, out_sentiment, out_translation, 
            out_romanized, out_json, out_file_json, out_file_csv
        ]
    )

if __name__ == "__main__":
    demo.launch()
