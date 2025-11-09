from pathlib import Path
import shutil, json

BASE = Path(__file__).resolve().parent.parent
NB = BASE / "notebooks"
MODELS = BASE / "models"
MODELS.mkdir(parents=True, exist_ok=True)

# copy model + preprocessor if produced in notebook
for name in ("ev_range_model.pkl", "preprocessor.pkl"):
    src = NB / name
    if src.exists():
        dst = MODELS / name
        shutil.copy2(src, dst)
        print("copied", src.name, "->", dst)

# write a simple metrics placeholder (you can update values later in notebook)
metrics = {"train_cv_mae": None, "test_mae": None}
with open(MODELS / "metrics.json", "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
print("wrote metrics.json")