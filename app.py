import streamlit as st
import sys
import json
from datetime import datetime
import pandas as pd
from multilingual_nlp import MultilingualNLPSystem

# Page configuration
st.set_page_config(
    page_title="Multilingual NLP System",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Modern CSS styling with improved usability and visual hierarchy
st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Reset and base styles - Clean, professional foundation */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean, professional background */
    .main {
        background: #f8fafc;
        padding: 0 !important;
        color: #1e293b;
    }
    
    /* Professional header hierarchy */
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e293b;
        text-align: left;
        margin: 1.5rem 0 0.5rem 0;
        padding: 0 2rem;
        line-height: 1.2;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #64748b;
        text-align: left;
        margin-bottom: 2rem;
        padding: 0 2rem;
        font-weight: 400;
    }
    
    /* Clean, functional cards without unnecessary styling */
    .functional-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Professional input styling */
    .stTextArea > div > div {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        color: #1e293b;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .stTextArea > div > div:focus-within {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        background: #ffffff;
        color: #1e293b;
    }

    /* Clean input fields */
    .stTextInput > div > div {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        color: #1e293b;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .stTextInput > div > div:focus-within {
        border-color: #3b82f6;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.1);
        background: #ffffff;
        color: #1e293b;
    }

    /* Ensure text in inputs is readable */
    .stTextArea textarea, .stTextInput input {
        color: #1e293b !important;
        background-color: #ffffff !important;
        font-size: 1rem !important;
        line-height: 1.5 !important;
    }
    
    /* Professional buttons */
    .stButton > button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        transition: background-color 0.2s ease;
        font-size: 1rem;
    }
    
    .stButton > button:hover {
        background: #2563eb;
        transform: none;
        box-shadow: none;
    }
    
    /* Clean select boxes */
    .stSelectbox > div > div {
        background: #ffffff;
        border: 1px solid #cbd5e1;
        border-radius: 6px;
        color: #1e293b;
    }
    
    /* Functional checkboxes */
    .stCheckbox > div {
        background: transparent;
        border-radius: 4px;
        padding: 0;
    }
    
    /* Results with clear hierarchy */
    .result-section {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Sentiment styling - clean and professional */
    .sentiment-positive {
        color: #059669;
        font-weight: 600;
    }
    
    .sentiment-negative {
        color: #dc2626;
        font-weight: 600;
    }
    
    .sentiment-neutral {
        color: #6b7280;
        font-weight: 600;
    }
    
    /* Sentiment styling */
    .sentiment-positive {
        background: linear-gradient(135deg, #00b09b, #96c93d);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    .sentiment-negative {
        background: linear-gradient(135deg, #ff416c, #ff4b2b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    .sentiment-neutral {
        background: linear-gradient(135deg, #a8a8a8, #536976);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    /* Animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0% {
            transform: scale(1);
        }
        50% {
            transform: scale(1.05);
        }
        100% {
            transform: scale(1);
        }
    }
    
    /* Loading spinner */
    .stSpinner > div {
        border-color: #667eea !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* JSON output styling */
    .json-output {
        background: rgba(248, 250, 252, 0.95);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 10px;
        padding: 15px;
        color: #1e293b;
        font-family: 'Courier New', monospace;
        font-size: 14px;
        overflow-x: auto;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
        .glass-card {
            padding: 1.5rem;
        }
    }
    
    /* Language badges */
    .lang-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize NLP system
@st.cache_resource
def init_nlp_system():
    return MultilingualNLPSystem()

nlp_system = init_nlp_system()

# Clean header section with professional hierarchy
st.markdown('<h1 class="main-header">Multilingual NLP Analysis</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Process text in multiple languages with sentiment analysis and translation</p>', unsafe_allow_html=True)

# Configuration section - functional and clean
col1, col2, col3 = st.columns([2, 2, 1])

with col1:
    target_language = st.selectbox(
        "Target Language",
        options=["en", "hi", "ta", "mr", "gu", "pa", "kn", "ml", "bn"],
        format_func=lambda x: {
            "en": "English",
            "hi": "Hindi (हिन्दी)",
            "ta": "Tamil (தமிழ்)",
            "mr": "Marathi (मराठी)",
            "gu": "Gujarati (ગુજરાતી)",
            "pa": "Punjabi (ਪੰਜਾਬੀ)",
            "kn": "Kannada (ಕನ್ನಡ)",
            "ml": "Malayalam (മലയാളം)",
            "bn": "Bengali (বাংলা)"
        }[x],
        help="Select target language for translation"
    )

with col2:
    analysis_options = st.multiselect(
        "Analysis Options",
        options=["sentiment", "translation", "romanization", "confidence"],
        default=["sentiment", "translation", "romanization"],
        help="Choose analysis types to perform"
    )

with col3:
    auto_detect = st.toggle("Auto-detect", value=True, help="Auto-detect input language")

# Text input area
st.subheader("Text Input")
input_text = st.text_area(
    "Enter text for analysis:",
    height=150,
    placeholder="Enter text in English, Hindi, Tamil, Marathi, Gujarati, Punjabi, Kannada, Malayalam, Bengali, or transliterated forms..."
)

# System capabilities showcase
with st.expander("🚀 System Capabilities", expanded=False):
    st.markdown("""
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
        <div class="result-card" style="text-align: center; padding: 1rem;">
            <h4>🌍 Language Detection</h4>
            <p>Auto-detect 9+ languages: English, Hindi, Tamil, Marathi, Gujarati, Punjabi, Kannada, Malayalam, Bengali</p>
        </div>
        <div class="result-card" style="text-align: center; padding: 1rem;">
            <h4>😊 Sentiment Analysis</h4>
            <p>Real-time emotion detection with confidence scores</p>
        </div>
        <div class="result-card" style="text-align: center; padding: 1rem;">
            <h4>🔄 Translation</h4>
            <p>Accurate cross-language translation with phonetics</p>
        </div>
        <div class="result-card" style="text-align: center; padding: 1rem;">
            <h4>📱 Romanization</h4>
            <p>Phonetic representation in English script</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Analysis button
if st.button("🔍 Analyze Text", type="primary", use_container_width=True):
    if input_text.strip():
        with st.spinner("Analyzing..."):
            try:
                # Perform analysis
                result = nlp_system.process_text(
                    text=input_text,
                    target_languages=[target_language]
                )
                
                # Display results with modern card layout
                st.markdown('<h2 style="margin: 2rem 0; text-align: center;">📊 Analysis Dashboard</h2>', unsafe_allow_html=True)

                # Create a beautiful results grid
                results_container = st.container()

                with results_container:
                    # Language Detection Card
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.markdown('<h4 style="color: #667eea; margin-bottom: 1rem;">🌍 Language Detection</h4>', unsafe_allow_html=True)
                        
                        detected_lang = result['detected_language']
                        lang_names = {
                            'en': '🇬🇧 English',
                            'hi': '🇮🇳 Hindi',
                            'ta': '🇮🇳 Tamil',
                            'mr': '🇮🇳 Marathi',
                            'gu': '🇮🇳 Gujarati',
                            'pa': '🇮🇳 Punjabi',
                            'kn': '🇮🇳 Kannada',
                            'ml': '🇮🇳 Malayalam',
                            'bn': '🇮🇳 Bengali'
                        }
                        
                        st.markdown(f'<div class="lang-badge">{lang_names.get(detected_lang, detected_lang)}</div>', unsafe_allow_html=True)
                        
                        if result.get('is_transliterated'):
                            st.success("✅ Auto-detected transliteration")
                        
                        if result.get('is_transliterated') and result.get('input_transliteration'):
                            st.markdown('**Native Script:**')
                            st.markdown(f'<span style="font-size: 1.2rem; font-weight: 600;">{result["native_script"]}</span>', unsafe_allow_html=True)
                            if result['native_script_romanized']:
                                st.markdown(f'<span style="color: #666; font-style: italic;">{result["native_script_romanized"]}</span>', unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="result-section">', unsafe_allow_html=True)
                        st.markdown('**Sentiment Analysis**')
                        
                        sentiment_label = result['sentiment']
                        sentiment_score = result['confidence']
                        
                        sentiment_display = {
                            'positive': '😊 Positive',
                            'negative': '😞 Negative', 
                            'neutral': '😐 Neutral'
                        }
                        
                        st.write(f"Result: {sentiment_display.get(sentiment_label, sentiment_label)}")
                        st.write(f"Confidence: {sentiment_score:.1%}")
                        st.progress(sentiment_score)
                        
                        st.markdown('</div>', unsafe_allow_html=True)

                    # Translation Results
                    st.subheader("Translations")

                    translated_text = result['translations'].get(target_language)
                    romanized_text = result['translations_romanized'].get(target_language)

                    if translated_text:
                        if result['detected_language'] == target_language:
                            st.write("**Original Text (Same Language)**")
                            st.write(result["native_script"])
                        else:
                            st.write("**Translated Text**")
                            st.write(translated_text)
                            
                            if romanized_text and romanized_text != translated_text:
                                st.write("**Phonetic (English)**")
                                st.write(romanized_text)

                    # Additional Information
                    if "confidence" in analysis_options:
                        st.markdown('<div class="result-card">', unsafe_allow_html=True)
                        st.markdown('<h4 style="color: #667eea; margin-bottom: 1rem;">📊 Confidence Details</h4>', unsafe_allow_html=True)
                        
                        confidence_data = {
                            'Language Detection': f"{result.get('language_confidence', 0.95):.1%}",
                            'Sentiment Analysis': f"{result['confidence']:.1%}",
                            'Translation': f"{result.get('translation_confidence', 0.90):.1%}"
                        }
                        
                        for key, value in confidence_data.items():
                            st.markdown(f"""
                            <div style="display: flex; justify-content: space-between; align-items: center; margin: 0.5rem 0;">
                                <span>{key}:</span>
                                <span style="font-weight: 600; color: #667eea;">{value}</span>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    # Raw JSON
                    with st.expander("📄 Raw JSON Output"):
                        st.json(result)
                    
                    # Export functionality with modern design
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown('<h4 style="color: #667eea; margin-bottom: 1rem;">📤 Export Results</h4>', unsafe_allow_html=True)

                    # Create modern export buttons
                    export_col1, export_col2, export_col3 = st.columns(3)

                    translated_text = result['translations'].get(target_language)
                    romanized_text = result['translations_romanized'].get(target_language)

                    if not translated_text and result['detected_language'] == target_language:
                        translated_text = result['native_script']
                        romanized_text = result['native_script_romanized']

                    with export_col1:
                        st.download_button(
                            label="📥 JSON",
                            data=json.dumps(result, indent=2, ensure_ascii=False),
                            file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                            mime="application/json",
                            use_container_width=True
                        )

                    with export_col2:
                        csv_data = pd.DataFrame([{
                            'Original Text': input_text,
                            'Native Script': result['native_script'],
                            'Detected Language': result['detected_language'],
                            'Sentiment': result['sentiment'],
                            'Sentiment Score': result['confidence'],
                            'Target Language': target_language,
                            'Translated Text': translated_text or "Not translated",
                            'Translated Romanized': romanized_text or "",
                            'Is Transliterated': "Yes" if result.get('is_transliterated') else "No",
                            'Timestamp': datetime.now()
                        }])
                        
                        st.download_button(
                            label="📊 CSV",
                            data=csv_data.to_csv(index=False),
                            file_name=f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                    with export_col3:
                        if st.button("📋 Copy Summary", use_container_width=True):
                            summary = f"""
Analysis Summary:
- Original: {input_text}
- Detected: {result['detected_language']}
- Sentiment: {result['sentiment']} ({result['confidence']:.1%})
- Translation: {translated_text or 'N/A'}
- Romanized: {romanized_text or 'N/A'}
                            """
                            st.code(summary)

                    st.markdown('</div>', unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.info("Please check your input text and try again.")
    else:
        st.warning("Please enter some text to analyze.")

# Sidebar with clean information
with st.sidebar:
    st.header("About")
    st.write("""
    This Multilingual NLP Studio processes text in multiple languages including:
    
    • English
    • Hindi & Hinglish  
    • Tamil & Tanglish
    
    Features:
    • Language Detection
    • Sentiment Analysis
    • Translation
    • Transliteration
    """)
    
    st.subheader("System Status")
    st.success("All systems operational")

# Clean footer
st.markdown("---")
st.caption("Built with Streamlit | Powered by Advanced NLP Models")