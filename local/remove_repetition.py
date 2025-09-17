import argparse
import logging

def trim_repeated_suffix(text: str, min_repeat: int = 5, key: str = "") -> str:
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
            new_text = tokens[:i + size] + pattern
            logging.warning(f"[{key}] Truncated repeated pattern x{repeat_count} → 1: {' '.join(pattern)}")
            return ' '.join(new_text)

    return text

def process_lines(input_path: str, output_path: str, min_repeat: int = 5):
    with open(input_path, 'r', encoding='utf-8') as fin, open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                logging.warning(f"Skipped line (format error): {line}")
                key = parts[0]
                fout.write(f"{key}\n")
                continue
            key, text = parts
            cleaned_text = trim_repeated_suffix(text, min_repeat, key)
            fout.write(f"{key} {cleaned_text}\n")

def main():
    parser = argparse.ArgumentParser(description="Remove repeated suffixes from text lines.")
    parser.add_argument('--input', '-i', required=True, help="Path to input text file")
    parser.add_argument('--output', '-o', required=True, help="Path to output text file")
    parser.add_argument('--min_repeat', '-r', type=int, default=5, help="Minimum repeat count to trigger truncation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    process_lines(args.input, args.output, args.min_repeat)

if __name__ == "__main__":
    main()
