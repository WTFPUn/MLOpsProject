from pathlib import Path
import pandas as pd

def get_test(csv_path, debug_path=None):
    sentences = []
    if debug_path is not None:
        txt_files = list(Path(debug_path).glob("*.txt"))
        for txt_file in txt_files:
            with txt_file.open("r", encoding="utf-8") as f:
                full_text = f.read().strip()
                if full_text:
                    sentences.append(full_text)
        print(f"Loaded {len(sentences)} test documents.")
    else:
        df = pd.read_csv(csv_path)
        df = df.dropna(subset=["content"])
        sentences = df["content"].tolist()
    return sentences

if __name__ == "__main__":
    get_test(debug_path=True)