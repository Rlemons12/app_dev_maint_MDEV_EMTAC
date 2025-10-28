import pandas as pd
from modules.configuration.config_env import DatabaseConfig  # Import your database configuration
from modules.emtacdb.emtacdb_fts import Part, PartsPositionImageAssociation
from modules.emtacdb.utlity.main_database.database import create_position


# Function to read the spreadsheet using a hard-coded file path
def get_spreadsheet_data():
    file_path = r"C:\Users\10169062\Desktop\AuMaintdb\Database\DB_LOADSHEETS\bom_for_AFL31600.xlsx"
    try:
        df = pd.read_excel(file_path)
        return df.to_dict(orient='records')  # Convert the dataframe to a list of dictionaries
    except Exception as e:
        print(f"Error reading the spreadsheet: {e}")
        return None

# Testing function
def handle_position_and_part_upload():
    # Initialize the database configuration
    db_config = DatabaseConfig()

    # Get a session for the main database
    session = db_config.get_main_session()

    # Use the hard-coded spreadsheet path to read data
    spreadsheet_data = get_spreadsheet_data()
    if spreadsheet_data is None:
        print("Failed to read the spreadsheet data. Exiting...")
        return

    try:
        # Hardcoded form data for testing purposes
        form_data = {
            'area_id': 3,
            'equipment_group_id': 4,
            'model_id': 21,
            # Add other form data fields if necessary
        }

        # Create or get the Position based on the form data
        position_id = create_position(area_id=form_data['area_id'], equipment_group_id=form_data['equipment_group_id'],
                                      model_id=form_data['model_id'], asset_number_id=form_data.get('asset_number_id'),
                                      location_id=form_data.get('location_id'),
                                      site_location_id=form_data.get('site_location_id'), session=session)

        if position_id is None:
            print("Failed to create or retrieve Position.")
            return

        # Process spreadsheet data row by row with updated mapping
        for row in spreadsheet_data:
            item_number = row.get('Item Number')  # Maps to part_number
            description = row.get('Description')  # Maps to name
            manufacturer = row.get('Manufacturer')  # Maps to oem_mfg
            mfg_part_number = row.get('Mfg Part Number')  # Maps to model
            long_description = row.get('Long Description')  # Maps to notes
            category = row.get('Category')  # Maps to class_flag in the Part class

            if not item_number:
                print("Skipping row due to missing Item Number.")
                continue

            # Check if Part already exists based on Item Number
            existing_part = session.query(Part).filter_by(part_number=item_number).first()

            if existing_part:
                part_id = existing_part.id
                print(f"Found existing part with ID: {part_id}")
                # Update the existing part with new data from the spreadsheet
                existing_part.name = description
                existing_part.oem_mfg = manufacturer
                existing_part.model = mfg_part_number
                existing_part.notes = long_description
                existing_part.class_flag = category  # Map Category from spreadsheet to class_flag in Part class
                session.commit()  # Commit changes
            else:
                # Create a new Part entry
                new_part = Part(
                    part_number=item_number,
                    name=description,
                    oem_mfg=manufacturer,
                    model=mfg_part_number,
                    notes=long_description,
                    class_flag=category  # Map Category from spreadsheet to class_flag in Part class
                )
                session.add(new_part)
                session.commit()  # Commit to get the new part_id
                part_id = new_part.id
                print(f"Created new part with ID: {part_id}")

            # Create PartsPositionImageAssociation entry
            new_association = PartsPositionImageAssociation(
                part_id=part_id,
                position_id=position_id,
                # Add additional fields if necessary for the association
            )
            session.add(new_association)

        # Final commit to save all associations
        session.commit()

    except Exception as e:
        print(f"An error occurred during the position and part upload process: {e}")
        session.rollback()
    finally:
        session.close()



if __name__ == "__main__":
    test_handle_position_and_part_upload()
