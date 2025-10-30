# modules/emtac_ai/training_scripts/dataset_gen/updt_generate_drawings_ner_train.py
from __future__ import annotations

import os
import sys
import time
import uuid
import json
from pathlib import Path
from typing import Dict, List, Tuple, Iterable

import pandas as pd
from sqlalchemy import select, text, Table, Column, String, Text, Integer, MetaData
from sqlalchemy.orm import Session
from sqlalchemy.dialects.postgresql import insert

# --- logger ---
from modules.configuration.log_config import (
    logger, info_id, debug_id, warning_id, error_id, get_request_id
)

# --- DB config / models ---
from modules.configuration.config_env import DatabaseConfig
from modules.emtac_ai.models.emtac_ai_db_models import (
    Intent,
    QueryTemplate,
    DatasetSource,
    Placeholder,
    PlaceholderColumnMap,
    TrainingSample,   # ORM for training_sample (provides compute_hash and bulk_upsert)
)

# ==========================================================
# Helpers (mirrors updt_generate_parts_ner_train structure)
# ==========================================================

# Single metadata for temp reflection when needed (public schema)
metadata = MetaData(schema="public")

t_training_data = Table(
    "training_data",
    metadata,
    Column("id", Integer, primary_key=True),
    Column("data_type", String(50), nullable=True),
    Column("data_content", Text, nullable=True),
    Column("label", String(100), nullable=True),
)

def _pick_sheet_name(xls: pd.ExcelFile, desired: str | None) -> str:
    sheets = xls.sheet_names or []
    if desired and desired in sheets:
        return desired
    # case-insensitive try
    if desired:
        want = desired.strip().lower()
        for s in sheets:
            if s.strip().lower() == want:
                return s
    # some common fallbacks
    for cand in ("Drawings", "drawings", "Sheet1", "sheet1"):
        for s in sheets:
            if s.strip().lower() == cand.lower():
                return s
    return sheets[0] if sheets else None

def detect_repo_root(start: Path | None = None) -> Path:
    cur = (start or Path(__file__).resolve())
    for p in [cur] + list(cur.parents):
        if (p / "modules").exists():
            return p
    return Path.cwd()

def resolve_dataset_path(ds_path: str, repo_root: Path) -> Path:
    raw = Path(ds_path)
    candidate = raw if raw.is_absolute() else (repo_root / raw).resolve()
    if candidate.exists():
        return candidate
    # collapse accidental "modules/modules"
    s = str(candidate)
    s2 = s.replace("\\modules\\modules\\", "\\modules\\").replace("/modules/modules/", "/modules/")
    alt = Path(s2)
    return alt if alt.exists() else candidate

def get_engine_with_retry(max_tries: int = 8, delay_seconds: float = 1.5):
    cfg = DatabaseConfig()
    last_err = None
    for attempt in range(1, max_tries + 1):
        try:
            engine = getattr(cfg, "engine", None) or cfg.get_engine()
            with engine.connect() as conn:
                conn.exec_driver_sql("SELECT 1")
            logger.info("Database connection established.")
            return engine
        except Exception as e:
            last_err = e
            if attempt < max_tries:
                logger.warning("Database not ready yet (%s). Retrying... [%d/%d]", str(e), attempt, max_tries)
                time.sleep(delay_seconds)
            else:
                logger.error("Database not reachable after retries.")
    raise last_err

# ==========================================================
# Load everything from DB (Intent/Templates/DatasetSource/PCM)
# ==========================================================

def load_from_db(session: Session) -> Tuple[Intent, List[str], DatasetSource, str, Dict[str, List[str]]]:
    """
    Return (intent_row, active_template_texts, drawings DatasetSource, sheet_name, candidate_map).
    This mirrors the parts loader but for intent 'drawings' and source 'drawings_loadsheet'.
    """
    # Intent
    intent = (
        session.query(Intent)
        .filter(Intent.name == "drawings", Intent.is_active.is_(True))
        .first()
    )
    if not intent:
        raise RuntimeError("No active Intent named 'drawings' found.")

    # Templates (active)
    templates = (
        session.query(QueryTemplate.template_text)
        .filter(QueryTemplate.intent_id == intent.id, QueryTemplate.is_active.is_(True))
        .order_by(QueryTemplate.id.asc())
        .all()
    )
    template_texts = [t[0] for t in templates]
    if not template_texts:
        raise RuntimeError("No active templates for intent 'drawings'.")

    # Dataset source (path to drawings loadsheet, plus extra sheet_name if any)
    ds = session.query(DatasetSource).filter(DatasetSource.name == "drawings_loadsheet").first()
    if not ds:
        raise RuntimeError("DatasetSource 'drawings_loadsheet' not found. Seed it first.")

    sheet_name = None
    if isinstance(ds.extra, dict):
        sheet_name = ds.extra.get("sheet_name") or "Drawings"

    # Placeholder candidates (resolve token -> candidate columns in DS)
    rows = session.execute(
        select(Placeholder.token, PlaceholderColumnMap.column_name)
        .join(PlaceholderColumnMap, Placeholder.id == PlaceholderColumnMap.placeholder_id)
        .where(PlaceholderColumnMap.dataset_id == ds.id)
        .order_by(PlaceholderColumnMap.placeholder_id.asc(), PlaceholderColumnMap.priority.asc())
    ).all()

    cand_map: Dict[str, List[str]] = {}
    for tok, col in rows:
        tok_u = (tok or "").strip().upper()
        if not tok_u or not col:
            continue
        cand_map.setdefault(tok_u, []).append(col.strip())

    return intent, template_texts, ds, sheet_name, cand_map

# ==========================================================
# Data reading / column resolution
# ==========================================================

def read_drawings_frame(ds: DatasetSource, repo_root: Path) -> pd.DataFrame:
    desired = (ds.extra or {}).get("sheet_name") or "Drawings"
    path_abs = resolve_dataset_path(ds.path, repo_root)
    logger.info("Reading drawings loadsheet: path='%s' (exists=%s), desired_sheet='%s'",
                str(path_abs), path_abs.exists(), desired)
    if not path_abs.exists():
        logger.error("Drawings loadsheet not found. ds.path='%s', repo_root='%s', resolved='%s'",
                     ds.path, str(repo_root), str(path_abs))
        raise FileNotFoundError(f"Drawings loadsheet not found at '{path_abs}'")

    xls = pd.ExcelFile(path_abs)
    logger.info("Workbook sheets detected: %s", xls.sheet_names)
    sheet = _pick_sheet_name(xls, desired)
    if not sheet:
        raise ValueError(f"No worksheets found in '{path_abs}'.")
    if sheet != desired:
        logger.warning("Requested sheet '%s' not found; using '%s' instead.", desired, sheet)

    df = pd.read_excel(xls, sheet_name=sheet)
    logger.info("Loaded %d rows from sheet '%s'.", len(df), sheet)
    return df

def resolve_columns_from_pcm(ds: DatasetSource, session: Session, columns: List[str]) -> Dict[str, str]:
    """
    Resolve EQUIPMENT_NUMBER, EQUIPMENT_NAME, DRAWING_NUMBER, DRAWING_NAME, SPARE_PART_NUMBER
    from PCM candidates present in the sheet.
    """
    required = ("EQUIPMENT_NUMBER", "EQUIPMENT_NAME", "DRAWING_NUMBER", "DRAWING_NAME", "SPARE_PART_NUMBER")

    # collect ordered candidates per token
    rows = session.execute(
        select(Placeholder.token, PlaceholderColumnMap.column_name)
        .join(PlaceholderColumnMap, Placeholder.id == PlaceholderColumnMap.placeholder_id)
        .where(PlaceholderColumnMap.dataset_id == ds.id)
        .order_by(PlaceholderColumnMap.placeholder_id.asc(), PlaceholderColumnMap.priority.asc())
    ).all()

    cand_map: Dict[str, List[str]] = {}
    for tok, col in rows:
        tok_u = (tok or "").strip().upper()
        if not tok_u or not col:
            continue
        cand_map.setdefault(tok_u, []).append(col.strip())

    # normalize sheet headers for lookup
    norm_to_orig = {c.strip().upper(): c for c in columns}
    chosen: Dict[str, str] = {}
    for tok in required:
        for cand in cand_map.get(tok, []):
            cand_norm = cand.strip().upper()
            if cand_norm in norm_to_orig:
                chosen[tok] = norm_to_orig[cand_norm]
                break

    # final fallback: if token itself is present as a header
    for tok in required:
        if tok not in chosen and tok in norm_to_orig:
            chosen[tok] = norm_to_orig[tok]

    if chosen:
        logger.info("Resolved columns: %s", ", ".join(f"{k} -> {v}" for k, v in chosen.items()))
    else:
        logger.warning("No columns could be resolved from sheet headers.")

    return chosen

# ==========================================================
# Builders (NER + Intent)
# ==========================================================

def _clean(v) -> str:
    if pd.isna(v) or v is None:
        return ""
    return str(v).strip()

def parse_spare_parts(spn) -> List[str]:
    if not spn or pd.isna(spn):
        return []
    return [p.strip() for p in str(spn).split(",") if p and p.strip()]

def build_ner_samples(df: pd.DataFrame, colmap: Dict[str, str]) -> List[Dict]:
    """
    Build basic NER utterances directly from a row by stitching available fields.
    We favor a consistent surface form to make span finding deterministic.
    """
    samples: List[Dict] = []

    for _, row in df.iterrows():
        eq_num = _clean(row.get(colmap.get("EQUIPMENT_NUMBER", ""), ""))
        eq_nam = _clean(row.get(colmap.get("EQUIPMENT_NAME", ""), ""))
        dwg_no = _clean(row.get(colmap.get("DRAWING_NUMBER", ""), ""))
        dwg_nm = _clean(row.get(colmap.get("DRAWING_NAME", ""), ""))
        spares = parse_spare_parts(row.get(colmap.get("SPARE_PART_NUMBER", ""), ""))

        # Choose one compact utterance that includes whatever exists
        parts = []
        if dwg_nm: parts += [dwg_nm]
        if dwg_no: parts += [(" (" if dwg_nm else ""), dwg_no, (")" if dwg_nm else "")]
        if eq_nam or eq_num: parts += [" for "]
        if eq_nam: parts += [eq_nam]
        if eq_num: parts += [(" " if eq_nam else ""), "#", eq_num]
        if spares: parts += [" with spare ", spares[0]]  # include at most one spare in this base utterance

        utter = "".join(parts).strip()
        if not utter:
            continue

        # spans
        ents: List[Dict] = []

        def add_span(value: str, label: str):
            if not value:
                return
            s = utter.find(value)
            if s >= 0:
                ents.append({"start": s, "end": s + len(value), "label": label, "text": value})

        add_span(eq_num, "EQUIPMENT_NUMBER")
        add_span(eq_nam, "EQUIPMENT_NAME")
        add_span(dwg_no, "DRAWING_NUMBER")
        add_span(dwg_nm, "DRAWING_NAME")
        if spares:
            add_span(spares[0], "SPARE_PART_NUMBER")

        if ents:
            samples.append({"text": utter, "entities": ents})

    logger.info("Built %d NER samples from drawings loadsheet.", len(samples))
    return samples

def build_intent_samples(templates: List[str], df: pd.DataFrame, colmap: Dict[str, str]) -> List[str]:
    """
    Expand your DB-stored templates by substituting real values from the sheet.
    Template tokens expected like:
      "Show me {DRAWING_NUMBER}"  "Print for {EQUIPMENT_NAME}" etc.
    """
    out: List[str] = []

    # small value pools per token
    vals: Dict[str, List[str]] = {k: [] for k in ("EQUIPMENT_NUMBER", "EQUIPMENT_NAME", "DRAWING_NUMBER", "DRAWING_NAME", "SPARE_PART_NUMBER")}
    for tok in vals.keys():
        col = colmap.get(tok)
        if col and col in df.columns:
            # capture top 50 non-null strings
            vals[tok] = [str(v) for v in df[col].dropna().astype(str).head(50).tolist()]

    def subst_one(t: str) -> List[str]:
        # generate multiple variants per template by sampling up to 10 per placeholder
        cands: List[str] = []
        tt = t
        has_any = False
        for tok in vals.keys():
            key = "{%s}" % tok
            if key in tt and vals[tok]:
                has_any = True
                for v in vals[tok][:10]:
                    cands.append(tt.replace(key, v))
        return cands if has_any else [tt]

    for t in templates:
        out.extend(subst_one(t))

    logger.info("Built %d intent samples from templates.", len(out))
    return out

def chunked(iterable: Iterable, size: int) -> Iterable[List]:
    buf = []
    for item in iterable:
        buf.append(item)
        if len(buf) >= size:
            yield buf
            buf = []
    if buf:
        yield buf

# ==========================================================
# DB writers (NER -> training_sample, intents -> training_data)
# ==========================================================

def upsert_ner_into_training_sample(session: Session, ner_samples: List[Dict], intent_name: str = "drawings") -> int:
    """Insert NER samples into training_sample with unique hash dedupe."""
    to_insert = []
    for ex in ner_samples:
        text_ = ex["text"]
        ents_ = ex.get("entities") or []
        h = TrainingSample.compute_hash("ner", text_, intent_name, ents_)
        to_insert.append({
            "sample_type": "drawings_ner",
            "text": text_,
            "intent": intent_name,
            "entities": ents_,
            "source": "loadsheet",
            "hash": h,
            "meta": None,
        })

    inserted = 0
    for batch in chunked(to_insert, 2000):
        inserted += TrainingSample.bulk_upsert(session, batch)
    session.commit()
    logger.info("Inserted %d NER samples into training_sample.", inserted)
    return inserted

def upsert_intents_into_training_data(session: Session, intent_samples: List[str], intent_name: str = "drawings") -> int:
    """
    Insert intent texts into public.training_data with data_type='intent'.
    De-duplicates via UNIQUE (data_type, data_content, "label") using ON CONFLICT DO NOTHING.
    """
    if not intent_samples:
        return 0

    # ensure unique index exists
    session.execute(text(
        'CREATE UNIQUE INDEX IF NOT EXISTS uq_training_data_type_content_label '
        'ON public.training_data (data_type, data_content, "label")'
    ))
    session.commit()

    records = [{"data_type": "intent", "data_content": txt, "label": intent_name} for txt in intent_samples]
    stmt = insert(t_training_data).values(records)
    stmt = stmt.on_conflict_do_nothing(index_elements=["data_type", "data_content", "label"])
    res = session.execute(stmt)
    session.commit()
    return res.rowcount or 0

def save_jsonl(ner_samples: List[Dict], intent_samples: List[str], out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    ner_path = os.path.join(out_dir, "drawings_ner.jsonl")
    with open(ner_path, "w", encoding="utf-8") as f:
        for ex in ner_samples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info("Saved %d NER samples -> %s", len(ner_samples), ner_path)

    intent_path = os.path.join(out_dir, "drawings_intents.jsonl")
    with open(intent_path, "w", encoding="utf-8") as f:
        for ex in intent_samples:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    logger.info("Saved %d intent samples -> %s", len(intent_samples), intent_path)

# ==========================================================
# Main
# ==========================================================

def main():
    req_id = uuid.uuid4().hex[:8]
    logger.info("Starting DRAWINGS dataset generation (req_id=%s)", req_id)

    engine = get_engine_with_retry()
    repo_root = detect_repo_root()

    with Session(engine) as session:
        intent, template_texts, ds, sheet_name, _cand_map = load_from_db(session)

        df = read_drawings_frame(ds, repo_root)
        chosen = resolve_columns_from_pcm(ds, session, list(df.columns))
        # We require at least one strong drawing field to proceed
        if not any(k in chosen for k in ("DRAWING_NUMBER", "DRAWING_NAME")):
            logger.error("Could not resolve required columns (need at least DRAWING_NUMBER or DRAWING_NAME). Aborting.")
            sys.exit(2)

        ner_samples = build_ner_samples(df, chosen)
        intent_samples = build_intent_samples(template_texts, df, chosen)

        # Write JSONL artifacts into a predictable repo path
        out_dir = os.path.join(repo_root, "modules", "emtac_ai", "training_data", "autogen")
        save_jsonl(ner_samples, intent_samples, out_dir)

        # Persist to DB
        n1 = upsert_ner_into_training_sample(session, ner_samples, intent_name=intent.name)
        n2 = upsert_intents_into_training_data(session, intent_samples, intent_name=intent.name)

        logger.info("Done DRAWINGS (req_id=%s). DB inserts: ner=%d, intent=%d", req_id, n1, n2)

if __name__ == "__main__":
    main()
