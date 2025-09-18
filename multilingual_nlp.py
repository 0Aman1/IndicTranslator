#!/usr/bin/env python3
"""
Multilingual NLP System for Sentiment Analysis and Translation
A plug-and-play system using pretrained models from HuggingFace.
"""

import json
import logging
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings("ignore")

from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM
import torch
import googletrans
from googletrans import Translator

# Add indic-transliteration for romanization
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    INDIC_AVAILABLE = True
except ImportError:
    INDIC_AVAILABLE = False
    logger.warning("indic-transliteration not available, romanization disabled")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageDetector:
    """Language detection module using langdetect."""
    
    def __init__(self):
        self.supported_languages = {'en', 'hi', 'ta', 'te', 'ml', 'kn', 'bn', 'gu', 'mr', 'pa'}
    
    def detect_language(self, text: str) -> str:
        """Detect the language of input text."""
        try:
            detected = detect(text)
            if detected in self.supported_languages:
                return detected
            return 'en'  # Default fallback
        except Exception as e:
            logger.warning(f"Language detection failed: {e}, defaulting to English")
            return 'en'

class SentimentAnalyzer:
    """Sentiment analysis using pretrained models."""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.model_mapping = {
            'en': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'hi': 'cardiffnlp/twitter-roberta-base-sentiment-latest',  # Use multilingual model
            'ta': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'te': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'ml': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'kn': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'bn': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'gu': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'mr': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
            'pa': 'cardiffnlp/twitter-roberta-base-sentiment-latest'
        }
        self.sentiment_labels = {
            'cardiffnlp/twitter-roberta-base-sentiment-latest': ['negative', 'neutral', 'positive']
        }
    
    def load_model(self, language: str):
        """Load sentiment analysis model for given language."""
        if language in self.models:
            return
        
        model_name = self.model_mapping.get(language, self.model_mapping['en'])
        
        try:
            logger.info(f"Loading sentiment model for {language}: {model_name}")
            self.tokenizers[language] = AutoTokenizer.from_pretrained(model_name)
            self.models[language] = AutoModelForSequenceClassification.from_pretrained(model_name)
            logger.info(f"Successfully loaded sentiment model for {language}")
        except Exception as e:
            logger.error(f"Failed to load sentiment model for {language}: {e}")
            # Fallback to English model
            if language != 'en':
                self.load_model('en')
    
    def analyze_sentiment(self, text: str, language: str) -> Dict[str, any]:
        """Analyze sentiment of text in given language."""
        if language not in self.models:
            self.load_model(language)
        
        try:
            model_name = self.model_mapping.get(language, self.model_mapping['en'])
            tokenizer = self.tokenizers.get(language, self.tokenizers.get('en'))
            model = self.models.get(language, self.models.get('en'))
            
            if not model or not tokenizer:
                return {"sentiment": "neutral", "confidence": 0.0, "error": "Model not available"}
            
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            
            with torch.no_grad():
                outputs = model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
                predicted_class = torch.argmax(probabilities, dim=-1).item()
                confidence = probabilities[0][predicted_class].item()
            
            labels = self.sentiment_labels.get(model_name, ['negative', 'neutral', 'positive'])
            sentiment = labels[predicted_class]
            
            return {
                "sentiment": sentiment,
                "confidence": round(confidence, 3)
            }
            
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {e}")
            return {"sentiment": "neutral", "confidence": 0.0, "error": str(e)}

class TranslatorModule:
    """Translation module using Google Translate API."""
    
    def __init__(self):
        pass
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using Google Translate API."""
        try:
            # Using requests to call Google Translate API directly
            import requests
            import urllib.parse
            
            url = "https://translate.googleapis.com/translate_a/single"
            params = {
                'client': 'gtx',
                'sl': source_lang,
                'tl': target_lang,
                'dt': 't',
                'q': text
            }
            
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                result = response.json()
                if result and len(result) > 0 and len(result[0]) > 0:
                    translated_text = ''.join([item[0] for item in result[0]])
                    return translated_text
            
            return f"[{text}] (Translation unavailable)"
            
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return f"[{text}] (Translation error)"

class RomanizerModule:
    """Romanization module using indic-transliteration."""
    
    def __init__(self):
        self.script_mapping = {
            'hi': sanscript.DEVANAGARI,
            'ta': sanscript.TAMIL,
            'te': sanscript.TELUGU,
            'ml': sanscript.MALAYALAM,
            'kn': sanscript.KANNADA,
            'bn': sanscript.BENGALI,
            'gu': sanscript.GUJARATI,
            'mr': sanscript.DEVANAGARI,
            'pa': sanscript.GURMUKHI
        }
    
    def romanize_text(self, text: str, source_language: str) -> str:
        """Convert text from native script to Roman (English) script."""
        if not INDIC_AVAILABLE:
            return "Romanization not available"
        
        try:
            if source_language in self.script_mapping:
                script = self.script_mapping[source_language]
                romanized = transliterate(text, script, sanscript.ITRANS)
                return romanized
            else:
                return text  # Already in English or unsupported
                
        except Exception as e:
            logger.error(f"Romanization failed: {e}")
            return text

    def romanize_translations(self, translations: Dict[str, str]) -> Dict[str, str]:
        """Romanize all translations."""
        if not INDIC_AVAILABLE:
            return {}
        
        romanized = {}
        for lang_code, text in translations.items():
            romanized[lang_code] = self.romanize_text(text, lang_code)
        return romanized

class MultilingualNLPSystem:
    """Main system orchestrating all NLP modules."""
    
    def __init__(self):
        self.language_detector = LanguageDetector()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.translator = TranslatorModule()
        self.romanizer = RomanizerModule()
    
    def process_text(self, text: str, target_languages: List[str] = None) -> Dict[str, any]:
        """Process text for sentiment analysis and translation."""
        if target_languages is None:
            target_languages = ['hi', 'ta']
        
        # First, check if text is transliterated from native language
        transliterated_lang = TransliterationDetector.detect_language_from_transliteration(text)
        
        original_text = text
        source_language = 'en'  # Default
        
        if transliterated_lang:
            # Text is transliterated, convert to native script
            native_text = TransliterationDetector.convert_to_native(text, transliterated_lang)
            source_language = transliterated_lang
            original_text = native_text
            logger.info(f"Detected transliterated {transliterated_lang}: '{text}' -> '{native_text}'")
        else:
            # Regular language detection for non-transliterated text
            source_language = self.language_detector.detect_language(text)
            original_text = text
        
        # Analyze sentiment using the detected language
        sentiment_result = self.sentiment_analyzer.analyze_sentiment(original_text, source_language)
        
        # Translate to target languages
        translations = {}
        for target_lang in target_languages:
            if target_lang != source_language:
                translated = self.translator.translate_text(original_text, source_language, target_lang)
                translations[target_lang] = translated
        
        # Romanize original text
        original_romanized = self.romanizer.romanize_text(original_text, source_language)
        
        # Romanize translations
        translations_romanized = self.romanizer.romanize_translations(translations)
        
        # Also provide transliteration of the original English-keyboard input
        input_transliteration = ""
        if transliterated_lang:
            input_transliteration = TransliterationDetector.convert_to_native(text, transliterated_lang)
        
        return {
            "original_text": text,  # Original input
            "detected_language": source_language,
            "native_script": original_text if transliterated_lang else text,
            "native_script_romanized": original_romanized,
            "input_transliteration": input_transliteration,
            "is_transliterated": bool(transliterated_lang),
            "sentiment": sentiment_result.get("sentiment"),
            "confidence": sentiment_result.get("confidence"),
            "translations": translations,
            "translations_romanized": translations_romanized
        }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multilingual NLP System")
    parser.add_argument("text", help="Text to analyze and translate")
    parser.add_argument("-t", "--targets", nargs="+", default=["hi", "ta"], 
                       help="Target languages for translation")
    parser.add_argument("-o", "--output", help="Output file for JSON results")
    
    args = parser.parse_args()
    
    system = MultilingualNLPSystem()
    result = system.process_text(args.text, args.targets)
    
    print(json.dumps(result, indent=2, ensure_ascii=False))
    
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")


import re
from typing import Dict, List, Optional, Tuple
import requests
import json
import logging
from datetime import datetime

try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    INDIC_TRANSLITERATION_AVAILABLE = True
except ImportError:
    INDIC_TRANSLITERATION_AVAILABLE = False
    print("Warning: indic-transliteration not available. Install with: pip install indic-transliteration")

try:
    from indic_transliteration.sanscript import SchemeMap, SCHEMES
    INDIC_ROMANIZATION_AVAILABLE = True
except ImportError:
    INDIC_ROMANIZATION_AVAILABLE = False

# Add transliteration detection patterns
class TransliterationDetector:
    """Detects and converts English-keyboard transliteration to native scripts"""
    
    @staticmethod
    def detect_language_from_transliteration(text: str) -> Optional[str]:
        """Detect if text is transliterated from a native language"""
        text_lower = text.lower().strip()
        
        # More specific Hindi patterns - prioritize common Hinglish words
        hindi_indicators = [
            'hai', 'tha', 'thi', 'the', 'raha', 'rahi', 'rahe', 'gaya', 'gayi', 'gaye',
            'mujhe', 'tujhe', 'apna', 'mera', 'tera', 'kya', 'kyu', 'kyun', 'kaise',
            'aaj', 'kal', 'sab', 'kuch', 'acha', 'bura', 'pyar', 'dost', 'ghar',
            'zindagi', 'waqt', 'din', 'raat', 'subah', 'shaam', 'khana', 'pani',
            'peene', 'ja', 'rha', 'hu', 'main', 'mai', 'hoon', 'hun', 'tum', 'aap',
            'kaha', 'kahan', 'kab', 'kaun', 'kisko', 'kisne', 'mujhse', 'tumse',
            'dil', 'dimaag', 'soch', 'baat', 'baat', 'kar', 'karo', 'karna', 'kiya'
        ]
        
        # More specific Tamil patterns  
        tamil_indicators = [
            'naan', 'en', 'un', 'avan', 'aval', 'avanga', 'idhu', 'adhu', 'enna',
            'eppadi', 'irukku', 'irundha', 'vandha', 'poren', 'varan', 'sollu',
            'romba', 'nalla', 'ketta', 'kadhal', 'thozhan', 'thozhi', 'veedu',
            'kaadhal', 'kaalam', 'naal', 'iravu', 'kaalai', 'maalai', 'saapadu',
            'thanni', 'evvalavu', 'neraya', 'kammiya', 'chinna', 'periya', 'puthusu',
            'pazhasu', 'azhagu', 'veyyil', 'kulir', 'inippu', 'uppu'
        ]
        
        # Score counting - give higher weight to Hindi indicators for Hinglish
        hindi_score = sum(2 for indicator in hindi_indicators if indicator in text_lower)
        tamil_score = sum(2 for indicator in tamil_indicators if indicator in text_lower)
        
        # Additional Hindi-specific patterns for Hinglish
        hindi_patterns = [
            r'\b[a-z]+ne\b',     # peene, karne, etc.
            r'\b[a-z]+a\b',      # ja, gaya, etc.
            r'\b[a-z]+i\b',      # thi, gai, etc.
            r'\b[a-z]+e\b',      # peene, jaane, etc.
            r'\b[a-z]+hu\b',     # hu, rahu, etc.
            r'\bmai[n]?\b',      # mai, main
            r'\bpaani\b',        # pani/paani
            r'\bja\b'            # ja
        ]
        
        # Additional Tamil-specific patterns
        tamil_patterns = [
            r'\b[a-z]+nga\b',  # avanga, ponga, etc.
            r'\b[a-z]+n\b',    # naan, poren, etc.
            r'\b[a-z]+u\b',    # irukku, sollu, etc.
            r'\b[a-z]+a\b'     # vandha, irundha, etc.
        ]
        
        # Check for Hindi-specific patterns
        for pattern in hindi_patterns:
            import re
            if re.search(pattern, text_lower):
                hindi_score += 3
        
        # Check for Tamil-specific patterns
        for pattern in tamil_patterns:
            import re
            if re.search(pattern, text_lower):
                tamil_score += 2
        
        # Hindi-specific endings with high weight
        hindi_endings = ['hai', 'tha', 'thi', 'the', 'raha', 'rahi', 'rahe', 'gaya', 'gayi', 'gaye', 'hu', 'hun', 'hoon']
        for ending in hindi_endings:
            if text_lower.endswith(ending):
                hindi_score += 4
        
        # Tamil-specific endings
        tamil_endings = ['irukku', 'poren', 'vandha', 'sollu', 'romba']
        for ending in tamil_endings:
            if text_lower.endswith(ending):
                tamil_score += 3
        
        # Determine language based on highest score
        if hindi_score > tamil_score and hindi_score >= 5:
            return 'hi'  # Hindi
        elif tamil_score > hindi_score and tamil_score >= 5:
            return 'ta'  # Tamil
        elif hindi_score > 0 or tamil_score > 0:
            # For close scores, use more specific patterns
            if any(word in text_lower for word in ['naan', 'eppadi', 'irukku', 'romba']):
                return 'ta'
            elif any(word in text_lower for word in ['hai', 'mera', 'kya', 'main', 'mai', 'paani', 'peene']):
                return 'hi'
        
        return None
    
    @staticmethod
    def convert_to_native(text: str, detected_lang: str) -> str:
        """Convert English transliteration to native script"""
        if not INDIC_TRANSLITERATION_AVAILABLE:
            return text
            
        try:
            if detected_lang == 'hi':
                # Convert to Devanagari (Hindi)
                return transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
            elif detected_lang == 'ta':
                # Convert to Tamil
                return transliterate(text, sanscript.ITRANS, sanscript.TAMIL)
            elif detected_lang == 'te':
                # Convert to Telugu
                return transliterate(text, sanscript.ITRANS, sanscript.TELUGU)
            elif detected_lang == 'ml':
                # Convert to Malayalam
                return transliterate(text, sanscript.ITRANS, sanscript.MALAYALAM)
            elif detected_lang == 'kn':
                # Convert to Kannada
                return transliterate(text, sanscript.ITRANS, sanscript.KANNADA)
            elif detected_lang == 'bn':
                # Convert to Bengali
                return transliterate(text, sanscript.ITRANS, sanscript.BENGALI)
            elif detected_lang == 'gu':
                # Convert to Gujarati
                return transliterate(text, sanscript.ITRANS, sanscript.GUJARATI)
            elif detected_lang == 'mr':
                # Convert to Devanagari (Marathi)
                return transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
            elif detected_lang == 'pa':
                # Convert to Gurmukhi (Punjabi)
                return transliterate(text, sanscript.ITRANS, sanscript.GURMUKHI)
            else:
                return text
        except Exception as e:
            logging.warning(f"Transliteration failed: {e}")