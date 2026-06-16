# check_training_labels.py
from sqlalchemy import text
from modules.configuration.config_env import DatabaseConfig

# Connect to your DB using the existing config
engine = DatabaseConfig().get_engine()

# Query the training_sample table for the first 200 rows
with engine.connect() as conn:
    rows = conn.execute(text(
        "SELECT entities FROM training_sample WHERE sample_type='ner' LIMIT 200"
    )).fetchall()

# Collect all unique labels
labels = set()
for (ents,) in rows:
    if not ents:
        continue
    for e in ents:
        lab = e.get("label") or e.get("entity")
        if lab:
            labels.add(str(lab))

# Print results
print("Distinct labels found:", sorted(labels))
