# modules/layout_manager.py
import json
from pathlib import Path
from datetime import datetime

LAYOUTS_DIR = Path.cwd() / "insightx_data" / "layouts"
LAYOUTS_DIR.mkdir(parents=True, exist_ok=True)

def save_layout(username: str, layout_name: str, layout_json: dict):
    """
    Save a dashboard layout per user.
    """
    path = LAYOUTS_DIR / f"{username}__{layout_name}.json"
    with open(path, "w") as f:
        json.dump({
            "meta": {"saved_at": datetime.utcnow().isoformat()},
            "layout": layout_json
        }, f)

def load_layouts_for_user(username: str):
    """
    Load all saved layouts for a given user.
    Returns a dict: {layout_name: layout_data}.
    """
    files = list(LAYOUTS_DIR.glob(f"{username}__*.json"))
    out = {}
    for fpath in files:
        with open(fpath, "r") as f:
            data = json.load(f)
        name = fpath.stem.split("__", 1)[1]
        out[name] = data
    return out
