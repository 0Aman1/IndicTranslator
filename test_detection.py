import multilingual_nlp

system = multilingual_nlp.MultilingualNLPSystem()

# Test cases
test_cases = [
    'aaj mai paani peene ja rha hu',  # Hinglish
    'eppadi irukkinga',               # Tanglish
    'naan saapadu saapidalam',        # Tanglish
    'mera dost bahut acha hai',       # Hinglish
    'kadhal romba azhagu',           # Tanglish
    'kal main school gaya tha'       # Hinglish
]

print('=== Hinglish-Tanglish Detection Test ===')
for text in test_cases:
    result = system.process_text(text, ['en', 'hi', 'ta'])
    detected = result['detected_language']
    native = result['native_script']
    
    lang_name = "Hinglish" if detected == "hi" else "Tanglish" if detected == "ta" else "Other"
    
    print(f'Input: {text}')
    print(f'Detected: {detected} ({lang_name})')
    print(f'Native: {native}')
    print('---')