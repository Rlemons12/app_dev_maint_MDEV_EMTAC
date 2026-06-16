import pandas as pd
from sqlalchemy.orm import DeclarativeMeta
from modules.emtacdb import emtacdb_fts as schema_module

OUTPUT_FILE = "Position_Loadsheet_Template.xlsx"

# Tables that contribute to creating a Position
POSITION_MODELS = [
    "Campus",
    "Building",
    "SiteLocation",
    "Area",
    "EquipmentGroup",
    "Model",
    "AssetNumber",
    "Location",
    "Subassembly",
    "ComponentAssembly",
    "AssemblyView",
    "Position"
]

def get_model(name: str):
    """Fetch model class by name."""
    return getattr(schema_module, name, None)

def get_model_columns(model):
    """Return list of column names for the given SQLAlchemy model."""
    return [col.name for col in model.__table__.columns]

def generate_position_loadsheet():
    """Create an Excel file with each sheet containing only column headers."""
    writer = pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter")

    print("\nGenerating Position loadsheet (column headers only)...")

    for model_name in POSITION_MODELS:
        model = get_model(model_name)

        if model is None:
            print(f"WARNING: Model not found → {model_name}")
            continue

        # Get only the column names
        columns = get_model_columns(model)

        # Create empty DataFrame with only headers
        df = pd.DataFrame(columns=columns)

        sheet_name = model.__tablename__[:31]  # Excel sheet name limit

        print(f"  → Writing sheet: {sheet_name} with {len(columns)} columns")

        df.to_excel(writer, sheet_name=sheet_name, index=False)

    writer.close()
    print(f"\nDONE: {OUTPUT_FILE} created.\n")

if __name__ == "__main__":
    generate_position_loadsheet()
