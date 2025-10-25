from deep_translator import GoogleTranslator
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

class Translate:
    def __init__(self):
        self.translator = GoogleTranslator(source='en')

    def translate_and_transliterate(self, text, target_lang_code):
        """
        Translates English text to target language, then transliterates it into English script.
        
        Args:
            text (str): The English text to translate.
            target_lang_code (str): The language code for translation (e.g., 'hi' for Hindi).
            
        Returns:
            str: Transliterated text in English script.
        """
        try:
            # Translate using deep-translator
            self.translator.target = target_lang_code
            translated_text = self.translator.translate(text)
            
            # Transliterate to English script using indic_nlp
            if translated_text and translated_text != text:  # Only transliterate if translation succeeded
                try:
                    transliterated_text = UnicodeIndicTransliterator.transliterate(
                        translated_text, target_lang_code, 'en'
                    )
                    return transliterated_text
                except:
                    # If transliteration fails, return translated text
                    return translated_text
            return text
            
        except Exception as e:
            print(f"Translation/transliteration error: {str(e)}")
            return text  # Return original text if process fails