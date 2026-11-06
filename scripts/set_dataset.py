"""Set CHOSEN_DATASET in configs/config.json:   python scripts/set_dataset.py german|so|acs"""
import json
import sys
from pathlib import Path

path = Path(__file__).resolve().parent.parent / "configs" / "config.json"
ds = sys.argv[1] if len(sys.argv) > 1 else ""
text = path.read_text(encoding="utf-8")
config = json.loads(text)
if ds not in config["DATASETS"]:
    sys.exit(f"usage: set_dataset.py {'|'.join(config['DATASETS'])}")
import re
new = re.sub(r'("CHOSEN_DATASET"\s*:\s*")[a-z]+(")', rf'\g<1>{ds}\2', text, count=1)
path.write_text(new, encoding="utf-8")
print(f"CHOSEN_DATASET = {ds}")
