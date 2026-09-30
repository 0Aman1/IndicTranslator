#!/usr/bin/env python3
"""
Multilingual NLP System for Sentiment Analysis and Translation
A plug-and-play system using pretrained models from HuggingFace.
"""

import json
import logging
import re
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings("ignore")

from langdetect import detect
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from deep_translator import GoogleTranslator, MyMemoryTranslator

# Add indic-transliteration for romanization
try:
    from indic_transliteration import sanscript
    from indic_transliteration.sanscript import transliterate
    INDIC_AVAILABLE = True
except ImportError:
    INDIC_AVAILABLE = False
    logging.warning("indic-transliteration not available, romanization disabled")

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
            'hi': 'cardiffnlp/twitter-roberta-base-sentiment-latest',
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
    """Translation module using deep-translator."""
    
    def __init__(self):
        self.mymemory_lang_map = {
            'en': 'en-GB', 'hi': 'hi-IN', 'ta': 'ta-IN', 'te': 'te-IN',
            'ml': 'ml-IN', 'kn': 'kn-IN', 'bn': 'bn-IN', 'gu': 'gu-IN',
            'mr': 'mr-IN', 'pa': 'pa-IN'
        }
    
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> str:
        """Translate text using deep-translator API with fallback."""
        try:
            sl = 'auto' if source_lang == 'en' else source_lang
            translated_text = GoogleTranslator(source=sl, target=target_lang).translate(text)
            
            if translated_text:
                return translated_text
            return f"[{text}] (Translation unavailable)"
            
        except Exception as e:
            logger.warning(f"GoogleTranslator failed: {e}. Trying fallback MyMemoryTranslator...")
            try:
                # Fallback to MyMemoryTranslator
                sl_my = self.mymemory_lang_map.get(source_lang, 'en-GB') if source_lang != 'auto' else 'en-GB'
                tl_my = self.mymemory_lang_map.get(target_lang, target_lang)
                
                translated_text = MyMemoryTranslator(source=sl_my, target=tl_my).translate(text)
                if translated_text:
                    return translated_text
                return f"[{text}] (Translation unavailable)"
            except Exception as fallback_e:
                logger.error(f"Fallback translation failed: {fallback_e}")
                return f"[{text}] (Translation error)"

class TransliterationDetector:
    """Detects and converts English-keyboard transliteration to native scripts"""
    
    @staticmethod
    def detect_language_from_transliteration(text: str) -> Optional[str]:
        """Detect if text is transliterated from a native language"""
        text_lower = text.lower().strip()
        
        # Extract exact words to prevent partial matching 
        words = set(re.findall(r'\b[a-z]+\b', text_lower))
        
        # Hindi patterns - prioritize common Hinglish words
        hindi_indicators = {
            'hai', 'tha', 'thi', 'the', 'raha', 'rahi', 'rahe', 'gaya', 'gayi', 'gaye',
            'mujhe', 'tujhe', 'apna', 'mera', 'tera', 'kya', 'kyu', 'kyun', 'kaise',
            'aaj', 'kal', 'sab', 'kuch', 'acha', 'bura', 'pyar', 'dost', 'ghar',
            'zindagi', 'waqt', 'din', 'raat', 'subah', 'shaam', 'khana', 'pani',
            'peene', 'ja', 'rha', 'hu', 'main', 'mai', 'hoon', 'hun', 'tum', 'aap',
            'kaha', 'kahan', 'kab', 'kaun', 'kisko', 'kisne', 'mujhse', 'tumse',
            'dil', 'dimaag', 'soch', 'baat', 'kar', 'karo', 'karna', 'kiya'
        }
        
        # Tamil patterns  
        tamil_indicators = {
            'naan', 'en', 'un', 'avan', 'aval', 'avanga', 'idhu', 'adhu', 'enna',
            'eppadi', 'irukku', 'irundha', 'vandha', 'poren', 'varan', 'sollu',
            'romba', 'nalla', 'ketta', 'kadhal', 'thozhan', 'thozhi', 'veedu',
            'kaadhal', 'kaalam', 'naal', 'iravu', 'kaalai', 'maalai', 'saapadu',
            'thanni', 'evvalavu', 'neraya', 'kammiya', 'chinna', 'periya', 'puthusu',
            'pazhasu', 'azhagu', 'veyyil', 'kulir', 'inippu', 'uppu'
        }
        
        # Score counting: EXACT MATCH ONLY
        hindi_score = sum(2 for w in words if w in hindi_indicators)
        tamil_score = sum(2 for w in words if w in tamil_indicators)
        
        # Check endings safely using actual words
        if words:
            last_word = text_lower.split()[-1]
            last_word = re.sub(r'[^a-z]', '', last_word)
            
            hindi_endings = {'hai', 'tha', 'thi', 'the', 'raha', 'rahi', 'rahe', 'gaya', 'gayi', 'gaye', 'hu', 'hun', 'hoon'}
            tamil_endings = {'irukku', 'poren', 'vandha', 'sollu', 'romba'}
            
            if last_word in hindi_endings:
                hindi_score += 4
            if last_word in tamil_endings:
                tamil_score += 3
        
        # Determine language based on highest score (Require a stronger threshold of 4)
        if hindi_score > tamil_score and hindi_score >= 4:
            return 'hi'
        elif tamil_score > hindi_score and tamil_score >= 4:
            return 'ta'
            
        return None

    @staticmethod
    def convert_to_native(text: str, detected_lang: str) -> str:
        """Convert English transliteration to native script"""
        if not INDIC_AVAILABLE:
            return text
            
        try:
            if detected_lang == 'hi':
                return transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
            elif detected_lang == 'ta':
                return transliterate(text, sanscript.ITRANS, sanscript.TAMIL)
            elif detected_lang == 'te':
                return transliterate(text, sanscript.ITRANS, sanscript.TELUGU)
            elif detected_lang == 'ml':
                return transliterate(text, sanscript.ITRANS, sanscript.MALAYALAM)
            elif detected_lang == 'kn':
                return transliterate(text, sanscript.ITRANS, sanscript.KANNADA)
            elif detected_lang == 'bn':
                return transliterate(text, sanscript.ITRANS, sanscript.BENGALI)
            elif detected_lang == 'gu':
                return transliterate(text, sanscript.ITRANS, sanscript.GUJARATI)
            elif detected_lang == 'mr':
                return transliterate(text, sanscript.ITRANS, sanscript.DEVANAGARI)
            elif detected_lang == 'pa':
                return transliterate(text, sanscript.ITRANS, sanscript.GURMUKHI)
            else:
                return text
        except Exception as e:
            logger.warning(f"Transliteration failed: {e}")
            return text

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
            "original_text": text,  
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
