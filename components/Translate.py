import asyncio
from googletrans import Translator
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

class Translate:
    def __init__(self):
        self.translator = Translator()

    async def _translate_async(self, text, target_lang_code):
        """Asynchronous translation helper method"""
        try:
            translation = await self.translator.translate(text, dest=target_lang_code)
            return translation.text
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return text

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
            # Create new event loop for this translation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Run translation in event loop
            translated_text = loop.run_until_complete(self._translate_async(text, target_lang_code))
            loop.close()
            
            # Transliterate to English script using indic_nlp
            if translated_text != text:  # Only transliterate if translation succeeded
                transliterated_text = UnicodeIndicTransliterator.transliterate(
                    translated_text, target_lang_code, 'en'
                )
                return transliterated_text
            return text
            
        except Exception as e:
            print(f"Translation/transliteration error: {str(e)}")
            return text  # Return original text if process fails