# 🌐 Multilingual NLP System

A comprehensive, plug-and-play multilingual Natural Language Processing system that provides sentiment analysis, translation, and romanization capabilities across multiple Indian and international languages.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Installation Instructions](#installation-instructions)
- [Usage Guide](#usage-guide)
- [System Architecture](#system-architecture)
- [Supported Languages](#supported-languages)
- [API Reference](#api-reference)
- [Web Interface](#web-interface)
- [Dependencies](#dependencies)
- [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Contribution Guidelines](#contribution-guidelines)
- [License Information](#license-information)
- [Acknowledgments](#acknowledgments)

## 🎯 Project Overview

The Multilingual NLP System is a sophisticated yet easy-to-use platform designed to break language barriers in text analysis. Built with state-of-the-art transformer models and modern web technologies, it provides:

- **Real-time sentiment analysis** across multiple languages
- **Accurate translation services** between language pairs
- **Phonetic romanization** for Indian languages
- **Automatic language detection** with high accuracy
- **Modern web interface** for user-friendly interaction
- **Command-line interface** for batch processing and automation

Whether you're a researcher analyzing multilingual data, a business processing customer feedback, or a developer building language-aware applications, this system provides the tools you need with minimal setup.

## ✨ Key Features

### 🧠 Advanced NLP Capabilities
- **Multilingual Sentiment Analysis**: Analyze emotions and opinions in 9+ languages
- **Language Detection**: Automatic identification of input language with 95%+ accuracy
- **Cross-language Translation**: Seamless translation between supported languages
- **Phonetic Romanization**: Convert Indian scripts to readable English phonetics

### 🖥️ Multiple Interfaces
- **Web Application**: Beautiful, responsive Streamlit interface
- **Command Line Tool**: Direct CLI access for scripting and automation
- **Python API**: Programmatic access for integration into larger projects

### 🔧 Technical Excellence
- **Pre-trained Models**: Uses state-of-the-art models from HuggingFace
- **Zero Configuration**: Works out-of-the-box with sensible defaults
- **Caching System**: Intelligent caching for improved performance
- **Error Handling**: Robust error handling with graceful fallbacks
- **Extensible Design**: Easy to add new languages and features

### 🌍 Language Support
- **Indian Languages**: Hindi, Tamil, Telugu, Malayalam, Kannada, Bengali, Gujarati, Marathi, Punjabi
- **International**: English (with more languages planned)
- **Mixed Language Support**: Handles Hinglish, Tanglish, and other hybrid inputs

## 🚀 Installation Instructions

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended (for optimal model performance)
- Internet connection (for initial model downloads)

### Quick Start Installation

1. **Clone or download the project**
   ```bash
   git clone <repository-url>
   cd multilingual-nlp-system
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   # On Windows
   python -m venv venv
   venv\Scripts\activate
   
   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python test_system.py
   ```

### Installation Troubleshooting

If you encounter issues during installation:

1. **PyTorch Installation**: If PyTorch fails to install, try:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

2. **Transformers Library**: For transformer model issues:
   ```bash
   pip install transformers --upgrade
   ```

3. **Google Translate**: If translation features fail:
   ```bash
   pip install googletrans==4.0.0rc1 --force-reinstall
   ```

## 📖 Usage Guide

### Method 1: Web Interface (Recommended for Beginners)

1. **Start the web application**
   ```bash
   streamlit run app.py
   ```

2. **Access the interface**
   - Open your browser to `http://localhost:8501`
   - Enter text in any supported language
   - Select analysis options and target languages
   - Click "Analyze Text" to see results

3. **Features available in web interface**
   - Real-time text analysis
   - Visual sentiment indicators
   - Translation with romanization
   - Language detection with confidence scores
   - Export and share results

### Method 2: Command Line Interface

**Basic sentiment analysis:**
```bash
python multilingual_nlp.py "I love this product!"
```

**Analyze with specific target languages:**
```bash
python multilingual_nlp.py "यह बहुत अच्छा है" -t en ta
```

**Save results to file:**
```bash
python multilingual_nlp.py "This is amazing" -o results.json
```

**Batch processing from file:**
```bash
python multilingual_nlp.py -f input_texts.txt -o batch_results.json
```

### Method 3: Python API Integration

```python
from multilingual_nlp import MultilingualNLPSystem

# Initialize the system
system = MultilingualNLPSystem()

# Process single text
result = system.process_text(
    text="I love this product!",
    target_languages=["hi", "ta"]
)
print(result)

# Batch processing
texts = [
    "This product is excellent",
    "यह उत्पाद बहुत बढ़िया है",
    "இந்த தயாரிப்பு சிறந்தது"
]

results = []
for text in texts:
    result = system.process_text(text, target_languages=["en"])
    results.append(result)

# Save results
import json
with open("batch_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
```

## 🏗️ System Architecture

The system follows a modular architecture with clear separation of concerns:

### Core Components

1. **LanguageDetector**: Detects input language using statistical analysis
2. **SentimentAnalyzer**: Performs emotion analysis using transformer models
3. **TranslatorModule**: Handles translation between languages
4. **RomanizerModule**: Converts scripts to phonetic English
5. **MultilingualNLPSystem**: Orchestrates all components

### Data Flow
```
Input Text → Language Detection → Sentiment Analysis → Translation → Romanization → Output
     ↓              ↓                    ↓              ↓              ↓
   Hinglish    Confidence Score    Emotion + Score   Translated    Phonetic
   Tanglish     Language ID        Positive/Negative  Text         English
```

### Model Architecture
- **Base Models**: RoBERTa-based sentiment analysis models
- **Translation**: Google Translate API with fallback mechanisms
- **Language Detection**: Statistical language detection with confidence scoring
- **Caching**: Local caching for improved performance

## 🌍 Supported Languages

### Primary Languages
| Language | Code | Script | Sentiment | Translation | Romanization |
|----------|------|---------|-----------|-------------|--------------|
| English | en | Latin | ✅ | ✅ | N/A |
| Hindi | hi | Devanagari | ✅ | ✅ | ✅ |
| Tamil | ta | Tamil | ✅ | ✅ | ✅ |
| Telugu | te | Telugu | ✅ | ✅ | ✅ |
| Malayalam | ml | Malayalam | ✅ | ✅ | ✅ |
| Kannada | kn | Kannada | ✅ | ✅ | ✅ |
| Bengali | bn | Bengali | ✅ | ✅ | ✅ |
| Gujarati | gu | Gujarati | ✅ | ✅ | ✅ |
| Marathi | mr | Devanagari | ✅ | ✅ | ✅ |
| Punjabi | pa | Gurmukhi | ✅ | ✅ | ✅ |

### Mixed Language Support
- **Hinglish**: Hindi + English hybrid
- **Tanglish**: Tamil + English hybrid
- **Other Hybrids**: Various Indian language + English combinations

## 📚 API Reference

### MultilingualNLPSystem Class

```python
class MultilingualNLPSystem:
    def __init__(self, cache_enabled: bool = True):
        """Initialize the NLP system with optional caching."""
    
    def process_text(self, text: str, target_languages: List[str] = None) -> Dict:
        """Process text and return analysis results."""
    
    def detect_language(self, text: str) -> str:
        """Detect the language of input text."""
    
    def analyze_sentiment(self, text: str, language: str) -> Dict:
        """Analyze sentiment of text in given language."""
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text between languages."""
    
    def romanize_text(self, text: str, language: str) -> str:
        """Convert text to romanized form."""
```

### Response Format

```json
{
  "original_text": "I love this product!",
  "detected_language": "en",
  "sentiment": "positive",
  "confidence": 0.987,
  "translations": {
    "hi": "मुझे यह उत्पाद पसंद है!",
    "ta": "எனக்கு இந்த தயாரிப்பு பிடிக்கும்!"
  },
  "romanized": {
    "hi": "mujhe yah utpaad pasand hai!",
    "ta": "enakku inda tayarippu piṭikkum!"
  },
  "processing_time": 1.234,
  "model_info": {
    "sentiment_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "translation_service": "google_translate"
  }
}
```

## 🌐 Web Interface

The web interface provides an intuitive way to interact with the NLP system:

### Main Features
- **Real-time Analysis**: Instant results as you type
- **Multi-language Input**: Accept mixed language inputs
- **Visual Feedback**: Color-coded sentiment indicators
- **Translation Display**: Side-by-side original and translated text
- **Romanization**: Phonetic representation for Indian languages
- **Export Options**: Save results in JSON format

### Interface Components
1. **Text Input Area**: Large, resizable text input
2. **Language Selector**: Dropdown for target languages
3. **Analysis Options**: Checkboxes for different analysis types
4. **Results Dashboard**: Organized display of all results
5. **System Capabilities**: Expandable info panel

## 📦 Dependencies

### Core Dependencies
```
transformers>=4.30.0      # State-of-the-art NLP models
torch>=2.0.0              # Deep learning framework
langdetect>=1.0.9         # Language detection
sentencepiece>=0.1.99     # Subword tokenization
indic-transliteration>=2.3.75  # Indian language romanization
googletrans==4.0.0rc1     # Translation services
requests>=2.31.0           # HTTP requests
numpy>=1.24.0             # Numerical computing
pandas>=2.0.0             # Data manipulation
streamlit>=1.28.0          # Web interface framework
click>=8.1.0              # CLI framework
colorama>=0.4.6           # Terminal colors
```

### Optional Dependencies
- **GPU Support**: Install CUDA-enabled PyTorch for faster processing
- **Additional Languages**: Extend with more language models
- **Custom Models**: Integrate domain-specific sentiment models

## ⚙️ Configuration

### Environment Variables
```bash
# Set cache directory for models
export NLP_CACHE_DIR="/path/to/cache"

# Configure logging level
export NLP_LOG_LEVEL="INFO"

# Set translation timeout
export TRANSLATION_TIMEOUT="30"

# Enable/disable GPU acceleration
export CUDA_VISIBLE_DEVICES="0"
```

### Configuration File
Create a `config.json` file for advanced settings:

```json
{
  "cache_enabled": true,
  "cache_directory": "./nlp_cache",
  "model_settings": {
    "sentiment_model": "cardiffnlp/twitter-roberta-base-sentiment-latest",
    "max_sequence_length": 512,
    "confidence_threshold": 0.7
  },
  "translation": {
    "service": "google_translate",
    "timeout": 30,
    "retry_attempts": 3
  },
  "logging": {
    "level": "INFO",
    "file": "nlp_system.log"
  }
}
```

## 🔧 Troubleshooting

### Common Issues and Solutions

**1. Model Download Failures**
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface
python multilingual_nlp.py "test text"
```

**2. Translation Timeouts**
```bash
# Increase timeout in code
system = MultilingualNLPSystem()
system.translation_timeout = 60
```

**3. Memory Issues**
```bash
# Use smaller models or batch processing
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
```

**4. Language Detection Errors**
- Ensure text is at least 10 characters long
- Mixed language inputs may need manual language specification
- Use `-l` flag to specify source language explicitly

**5. Web Interface Won't Load**
```bash
# Check port availability
netstat -an | findstr 8501
# Use different port
streamlit run app.py --server.port 8502
```

### Performance Optimization
- **Enable GPU acceleration** for faster processing
- **Use caching** to avoid repeated model loading
- **Batch processing** for multiple texts
- **Adjust batch size** based on available memory

## 🤝 Contribution Guidelines

We welcome contributions from the community! Here's how you can help:

### Getting Started
1. **Fork the repository** on GitHub
2. **Clone your fork** locally
3. **Create a feature branch** (`git checkout -b feature/amazing-feature`)
4. **Make your changes** following our coding standards
5. **Test thoroughly** using provided test cases
6. **Commit with clear messages** (`git commit -m 'Add amazing feature'`)
7. **Push to your branch** (`git push origin feature/amazing-feature`)
8. **Open a Pull Request** with detailed description

### Areas for Contribution
- **New Language Support**: Add support for additional languages
- **Model Improvements**: Integrate better sentiment models
- **UI/UX Enhancements**: Improve web interface design
- **Performance Optimization**: Speed up processing times
- **Documentation**: Improve guides and examples
- **Bug Fixes**: Report and fix issues
- **Testing**: Add comprehensive test cases

### Coding Standards
- Follow **PEP 8** style guidelines
- Add **docstrings** to all functions and classes
- Include **type hints** where appropriate
- Write **unit tests** for new features
- Update **documentation** for changes
- Use **meaningful variable names**
- Add **error handling** for edge cases

### Testing Guidelines
```bash
# Run all tests
python -m pytest test_system.py

# Run specific test
python -m pytest test_system.py::test_sentiment_analysis

# Test with coverage
python -m pytest --cov=multilingual_nlp test_system.py
```

### Pull Request Process
1. Ensure all tests pass
2. Update documentation for changes
3. Add entry to CHANGELOG.md
4. Request review from maintainers
5. Address review feedback
6. Squash commits if requested

## 📄 License Information

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### License Summary
- ✅ **Commercial Use**: You can use this project for commercial purposes
- ✅ **Modification**: You can modify and distribute this project
- ✅ **Distribution**: You can distribute the original or modified versions
- ✅ **Private Use**: You can use this project privately
- ⚠️ **Liability**: The authors are not liable for any damages
- ⚠️ **Warranty**: No warranty is provided

### Third-Party Licenses
This project uses several open-source libraries:
- **Transformers**: Apache 2.0 License
- **PyTorch**: BSD-style License
- **Streamlit**: Apache 2.0 License
- **Google Translate**: Subject to Google's Terms of Service

## 🙏 Acknowledgments

### Technical Contributions
- **HuggingFace**: For providing state-of-the-art transformer models
- **Google Translate**: For translation services
- **Streamlit**: For the amazing web framework
- **PyTorch**: For the deep learning framework

### Community and Support
- **Open Source Community**: For continuous improvements and feedback
- **Contributors**: All the amazing people who have contributed to this project
- **Users**: Everyone who uses and provides feedback on the system

### Academic References
- **RoBERTa**: Robustly Optimized BERT Pretraining Approach
- **Multilingual BERT**: BERT for multilingual understanding
- **Google Translate**: Neural Machine Translation research

---

## 📞 Support and Contact

For support, questions, or collaboration:
- **Issues**: Report bugs and request features on GitHub Issues
- **Discussions**: Join community discussions
- **Email**: Contact maintainers directly
- **Documentation**: Check the wiki and examples

**⭐ Star this repository** if you find it useful!

---

**Made with ❤️ by the Multilingual NLP Team**  
*Breaking language barriers, one translation at a time.*