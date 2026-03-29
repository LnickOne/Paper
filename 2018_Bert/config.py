import os


def _find_paper_root() -> str:
    current = os.path.dirname(os.path.abspath(__file__))
    while True:
        name = os.path.basename(current)
        if len(name) > 5 and name[:4].isdigit() and name[4] == "_":
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return os.path.dirname(os.path.abspath(__file__))
        current = parent


PAPER_ROOT = _find_paper_root()
PAPER_NAME = os.path.basename(PAPER_ROOT)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = os.environ.get("MODEL_NAME", "bert-base-uncased")

# ---- Per-paper data/cache roots ----
PAPER_DATA = os.environ.get("PAPER_DATA", f"/home/lnick/DataSet/Hugging-Face/{PAPER_NAME}")
HF_HOME = os.path.join(PAPER_DATA, "hf_home")
HF_HUB_CACHE = os.path.join(HF_HOME, "hub")
HF_DATASETS_CACHE = os.path.join(HF_HOME, "datasets")
HF_MODULES_CACHE = os.path.join(HF_HOME, "modules")
TRANSFORMERS_CACHE = os.path.join(HF_HOME, "transformers")

# Strict isolation: override any globally-exported HF cache variables.
os.environ["HF_HOME"] = HF_HOME
os.environ["HF_HUB_CACHE"] = HF_HUB_CACHE
os.environ["HF_DATASETS_CACHE"] = HF_DATASETS_CACHE
os.environ["HF_MODULES_CACHE"] = HF_MODULES_CACHE
os.environ["TRANSFORMERS_CACHE"] = TRANSFORMERS_CACHE

# Optional mirror endpoint
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

# ---- Per-paper outputs under /home/lnick/Output/<paper> ----
OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", f"/home/lnick/Output/{PAPER_NAME}")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", os.path.join(OUTPUT_ROOT, "output"))
OUTPUT_BASE = OUTPUT_DIR
LOG_BASE = os.path.join(OUTPUT_ROOT, "logs")

os.makedirs(os.path.join(PAPER_DATA, "data"), exist_ok=True)
os.makedirs(HF_HUB_CACHE, exist_ok=True)
os.makedirs(HF_DATASETS_CACHE, exist_ok=True)
os.makedirs(HF_MODULES_CACHE, exist_ok=True)
os.makedirs(TRANSFORMERS_CACHE, exist_ok=True)
os.makedirs(OUTPUT_BASE, exist_ok=True)
os.makedirs(LOG_BASE, exist_ok=True)
