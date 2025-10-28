# EMTAC Database Maintenance Utilities – User Guide

This guide explains how to run and manage the EMTAC Optimized Database Maintenance Utilities.  
These tools help maintain database consistency, repair associations, and validate data integrity.

---

## 📦 Overview

The utilities support the following tasks:

- Associate **parts** with **images**
- Associate **drawings** with **parts**
- Validate that **images** have corresponding **embeddings**
- Run **all maintenance tasks** in sequence

Each operation:
- Uses **vectorized processing** (fast, scalable)
- Generates detailed **CSV reports** for audit and troubleshooting
- Logs execution time and results

---

## 🚀 Running the Tool

The main entry point is:

```powershell
python run_maintenance.py --task <task>
```

⚠️ **PowerShell Note**: Do **not** type `<task>` literally.  
Replace it with one of the valid task names, e.g.:

```powershell
python run_maintenance.py --task associate-images
python run_maintenance.py --task associate-drawings
python run_maintenance.py --task validate-embeddings
python run_maintenance.py --task run-all
```

### Options

- `--report-dir <path>` → Save reports in a custom directory (default: `db_maint_logs`)  
- `--no-report` → Do not generate report files  
- `--quick` → Minimal console output  

---

## 🛠 Available Tasks

### 1. `associate-images` 🖼
Associates **parts** with matching **images** (based on part number in image titles).

**Reports generated:**
- `optimized_part_image_summary_*.csv`
- `optimized_part_image_details_*.csv`
- `optimized_unmatched_parts_*.csv`

**Example:**
```powershell
python run_maintenance.py --task associate-images
```

---

### 2. `associate-drawings` 📋
Associates **drawings** with **parts** (based on spare part numbers).

**Reports generated:**
- `optimized_drawing_part_summary_*.txt`
- `optimized_drawing_part_matches_*.csv`
- `optimized_drawing_part_unmatched_*.csv`

**Example:**
```powershell
python run_maintenance.py --task associate-drawings
```

---

### 3. `validate-embeddings` 🧩
Validates that all **images** have embeddings and all **embeddings** map to valid images.

**Reports generated:**
- `image_embedding_summary_*.csv`
- `images_missing_embeddings_*.csv`
- `orphan_embeddings_*.csv`

**Examples:**
```powershell
# Run embedding validation only
python run_maintenance.py --task validate-embeddings

# Save reports to a custom folder
python run_maintenance.py --task validate-embeddings --report-dir .\reports

# Run without exporting reports
python run_maintenance.py --task validate-embeddings --no-report
```

---

### 4. `run-all` ⚡
Runs all maintenance tasks in sequence:
- Part–Image association
- Drawing–Part association
- (Optional) Embedding validation can be run separately

**Example:**
```powershell
python run_maintenance.py --task run-all
```

---

## 📂 Report Location

By default, reports are saved in:

```
db_maint_logs/
```

Override this with `--report-dir`.

Reports include:
- **Summary** (counts and timing)
- **Detailed matches**
- **Unmatched or missing items**

---

## 📝 Tips & Guidance

- If no associations are created:
  - Ensure images are loaded in the database.
  - Confirm drawing spare part numbers match parts in the database.
- Use `validate-embeddings` regularly to ensure your embeddings remain consistent with images.
- Each report is timestamped for traceability.
- Quick mode (`--quick`) suppresses detailed console logs if you only need summaries.

---

## ✅ Task Summary

- **`associate-images`** → Link parts to images  
- **`associate-drawings`** → Link drawings to parts  
- **`validate-embeddings`** → Check image–embedding consistency  
- **`run-all`** → Run all operations together  

---
