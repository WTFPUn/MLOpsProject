from pathlib import Path

def get_test(debug_path=None):
    sentences = []
    if debug_path is not None:
        txt_files = list(Path(debug_path).glob("*.txt"))
        for txt_file in txt_files:
            with txt_file.open("r", encoding="utf-8") as f:
                full_text = f.read().strip()
                if full_text:
                    sentences.append(full_text)
        print(f"Loaded {len(sentences)} test documents.")
    return sentences

if __name__ == "__main__":
    get_test(True)