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

# Create a new Area instance
new_area = Area(name='Test Area', description='This is a updated for the description part')

# Add the new_area instance to the session
session.add(new_area)
print(f'about to comit')
# Commit the transaction to persist the changes to the database
session.commit()

# Close the session
session.close()
