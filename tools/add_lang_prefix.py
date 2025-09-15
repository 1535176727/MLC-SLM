import sys
import re

# 语言代码与全称的映射
lang_map = {
    "en": "English",
    "es": "Spanish",
    "de": "German",
    "fr": "French",
    "th": "Thai",
    "ja": "Japanese",
    "it": "Italian",
    "ru": "Russian",
    "pt": "Portuguese",
    "vi": "Vietnamese",
    "ko": "Korean",
}

if len(sys.argv) != 3:
    print("用法: python add_lang_prefix.py 输入文件 输出文件")
    sys.exit(1)

input_file = sys.argv[1]
output_file = sys.argv[2]

with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        parts = line.strip().split(maxsplit=1)
        if len(parts) < 2:
            continue
        utt_id, text = parts
        match = re.search(r'_([a-z]{2})_', utt_id)
        if match:
            lang_code = match.group(1)
            lang_full = lang_map.get(lang_code, "Unknown")
            new_utt_id = f"{lang_full}_{utt_id}"
            fout.write(f"{new_utt_id} {text}\n")
