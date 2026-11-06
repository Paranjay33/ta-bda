from pathlib import Path
import pandas as pd

def process_split(file_path: Path):
    if not file_path.exists():
        print(f"[MISSING] {file_path}")
        return False
    
    rows = []
    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) >= 3:
                label = tokens[-1]
                if label in ['0', '1', '2']:
                    text = " ".join(tokens[1:-1])
                    rows.append({"text": text, "label": int(label)})
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(file_path, index=False, sep='\t', header=True)
        print(f"[DONE] Formatted {file_path} with headers (rows: {len(df)})")
        return True
    else:
        print(f"[ERROR] Could not parse rows from {file_path}")
        return False

if __name__ == "__main__":
    raw_dir = Path("data/raw")
    for fname in ["train.txt", "validation.txt", "test.txt"]:
        fpath = raw_dir / fname
        process_split(fpath)
        if fpath.exists():
            (Path(fname)).write_bytes(fpath.read_bytes())