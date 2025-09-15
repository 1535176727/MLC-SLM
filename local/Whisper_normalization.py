import re
import json
import logging
from langid.langid import LanguageIdentifier, model
from transformers.models.whisper.tokenization_whisper import BasicTextNormalizer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

# 初始化 langid
#lid = LanguageIdentifier.from_modelstring(model, norm_probs=True)
#logging.getLogger('langid').setLevel(logging.INFO)

# 支持语言
#limited_langs = ['en', 'de', 'ja', 'pt', 'ru', 'fr', 'it', 'ko', 'es', 'th', 'vi']
#lid.set_languages(limited_langs)

# 加载英文正则映射
mapping = json.load(open("local/english.json"))


class WhisperNormalizer:
    def __init__(self):
        self.english_normalizer = EnglishTextNormalizer(mapping)
        self.basic_normalizer = BasicTextNormalizer()

    #def detect_language(self, text: str) -> str:
    #   return lid.classify(text)[0]

    def normalize(self, text: str, key:str) -> str:
        if not text.strip():
            return ""

        #lang = self.detect_language(text)
        lang = key.split("-")[0]
        LANG_MAP = {
    "English": "en", "French": "fr", "German": "de", "Italian": "it", "Portuguese": "pt",
    "Spanish": "es", "Japanese": "ja", "Korean": "ko", "Russian": "ru", "Thai": "th", "Vietnamese": "vi"
}
        lang = LANG_MAP.get(lang, 'en')  # 默认英文
        if lang == 'en' and self.english_normalizer is not None:
            text=text
            #text = self.english_normalizer(text)
        else:
            text=text
            #text = self.basic_normalizer(text)

        # 截断循环冗余尾部
        text = self.trim_repeated_suffix(text)

        return text

    def trim_repeated_suffix(self, text: str, min_repeat: int = 5) -> str:
        """
        检测并截断文本中尾部重复的 token 片段，避免 LLM 重复输出。
        """
        tokens = text.strip().split()
        max_pattern_len = min(10, len(tokens) // 2)

        for size in range(1, max_pattern_len + 1):
            pattern = tokens[-size:]
            repeat_count = 1
            i = len(tokens) - size * 2
            while i >= 0 and tokens[i:i + size] == pattern:
                repeat_count += 1
                i -= size
            if repeat_count >= min_repeat:
                # 保留前面的部分，去掉重复
                new_text = tokens[:i + size]
                logging.warning(f"[123Truncated repeated pattern x{repeat_count}]: {' '.join(pattern)}")
                return ' '.join(new_text)
                #return text
        return text
