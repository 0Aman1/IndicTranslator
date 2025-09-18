# Multilingual NLP System

A plug-and-play multilingual NLP system for sentiment analysis and translation using pretrained models.

## Features
- **Multilingual Sentiment Analysis**: Detect sentiment in English, Hindi, Tamil, and more
- **Multilingual Translation**: Translate between multiple language pairs
- **Pretrained Models**: Uses state-of-the-art models from HuggingFace
- **No Training Required**: Works out-of-the-box
- **CLI Interface**: Easy command-line usage

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Basic usage:
```bash
python multilingual_nlp.py "I love this product!"
```

Specify target languages:
```bash
python multilingual_nlp.py "यह बहुत अच्छा है" -t en ta
```

Save results to file:
```bash
python multilingual_nlp.py "This is amazing" -o results.json
```

### Python API

```python
from multilingual_nlp import MultilingualNLPSystem

system = MultilingualNLPSystem()
result = system.process_text("I love this product!", target_languages=["hi", "ta"])
print(result)
```

## Supported Languages
- English (en)
- Hindi (hi)
- Tamil (ta)
- Telugu (te)
- Malayalam (ml)
- Kannada (kn)
- Bengali (bn)
- Gujarati (gu)
- Marathi (mr)
- Punjabi (pa)

## Example Output
```json
{
  "original_text": "I love this product!",
  "language": "en",
  "sentiment": "positive",
  "confidence": 0.987,
  "translations": {
    "hi": "मुझे यह उत्पाद पसंद है!",
    "ta": "எனக்கு இந்த தயாரிப்பு பிடிக்கும்!"
  }
}
```

## Notes
- Models are downloaded automatically on first use
- Google Translate is used as fallback for unsupported language pairs
- Results are cached locally for faster subsequent runs