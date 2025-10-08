import asyncio
from googletrans import Translator
from indicnlp.transliterate.unicode_transliterate import UnicodeIndicTransliterator

class Translate:
    def __init__(self):
        self.translator = Translator()
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    async def _translate_async(self, text, target_lang_code):
        """Asynchronous translation helper method"""
        translation = await self.translator.translate(text, dest=target_lang_code)
        return translation.text

    def translate_and_transliterate(self, text, target_lang_code):
        """
        Translates English text to the target language, then transliterates it into English script.
        
        Args:
            text (str): The English text to translate.
            target_lang_code (str): The language code for translation (e.g., 'hi' for Hindi).
            
        Returns:
            str: Transliterated text in English script.
        """
        try:
            # Run translation in event loop
            translated_text = self.loop.run_until_complete(self._translate_async(text, target_lang_code))
            
            # Transliterate to English script using indic_nlp
            transliterated_text = UnicodeIndicTransliterator.transliterate(translated_text, target_lang_code, 'en')
            return transliterated_text
        except Exception as e:
            print(f"Translation error: {str(e)}")
            return text  # Return original text if translation fails