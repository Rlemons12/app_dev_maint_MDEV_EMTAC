from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.orm import declarative_base
from modules.configuration.config import DATABASE_URL
from modules.emtacdb.emtacdb_fts import Area

# Database setup
Base = declarative_base()
engine = create_engine(DATABASE_URL)
Session = sessionmaker(bind=engine)
session = Session()


print(f'start')
# Step 1: Query the record you want to delete
area_to_delete = session.query(Area).filter_by(name='Test Area', description='This is a test area').first()

# Step 2: Mark the record for deletion
if area_to_delete:
    session.delete(area_to_delete)
    session.commit()
    print("Record deleted successfully!")
else:
    print("Record not found.")
    
    