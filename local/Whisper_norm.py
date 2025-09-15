import re
import json
import logging
import argparse
from typing import List
from transformers.models.whisper.tokenization_whisper import BasicTextNormalizer
from transformers.models.whisper.english_normalizer import EnglishTextNormalizer

# 设置日志格式与等级
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

# 加载英文正则映射
try:
    #with open("local/english.json", "r", encoding="utf-8") as f:
    with open("pre_trained_models/whisper-large-v3/normalizer.json", "r", encoding="utf-8") as f:
        mapping = json.load(f)
except Exception as e:
    logging.error(f"Failed to load english.json: {e}")
    mapping = {}

if not mapping:
    logging.warning("Warning: english.json mapping is empty. English normalization may be incomplete.")

class WhisperNormalizer:
    def __init__(self):
        self.english_normalizer = EnglishTextNormalizer(mapping)
        self.basic_normalizer = BasicTextNormalizer()

    def normalize_line(self, line: str) -> str:
        line = line.strip()
        if not line:
            return ""

        parts = line.split(maxsplit=1)
        key = parts[0]
        text = parts[1] if len(parts) > 1 else ""

        lang = key.split("-")[0]

        if not text.strip():
            return key

        try:
            if lang == 'English':
                text=text
                text = self.english_normalizer(text)
            else:
                text=text
                text = self.basic_normalizer(text)

            #text = self.trim_repeated_suffix(text, key=key)
            return f"{key} {text}"
        except Exception as e:
            logging.error(f"Normalization failed for line: {line}\nError: {e}")
            return key

    def trim_repeated_suffix(self, text: str, key: str = "", min_repeat: int = 5) -> str:
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
                new_tokens = tokens[:i + size]
                logging.warning(f"[{key}] Truncated repeated pattern x{repeat_count}: {' '.join(pattern)}")
                return ' '.join(new_tokens)

        return text

def normalize_text_lines(lines: List[str]) -> List[str]:
    normalizer = WhisperNormalizer()
    return [normalizer.normalize_line(line) for line in lines if line.strip()]

def main():
    parser = argparse.ArgumentParser(description="Normalize key-text lines using Whisper normalization.")
    parser.add_argument("--input", "-i", required=True, help="Path to input text file")
    parser.add_argument("--output", "-o", required=True, help="Path to output text file")
    args = parser.parse_args()

    try:
        with open(args.input, "r", encoding="utf-8") as fin:
            lines = fin.readlines()
    except Exception as e:
        logging.error(f"Failed to read input file: {e}")
        return

    normalized = normalize_text_lines(lines)

    try:
        with open(args.output, "w", encoding="utf-8") as fout:
            for line in normalized:
                fout.write(line.strip() + "\n")
    except Exception as e:
        logging.error(f"Failed to write output file: {e}")
        return

    logging.info(f"Normalization completed. Output written to {args.output}")

if __name__ == "__main__":
    main()
