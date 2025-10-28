# versiontrack.py
from sqlalchemy import event
from modules.configuration.config import NUM_VERSIONS_TO_KEEP, DATABASE_URL # Import the configuration variable
from sqlalchemy import (DateTime, Column, ForeignKey, Integer, String, create_engine, func)
from sqlalchemy.orm import declarative_base, scoped_session, sessionmaker


# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
Base = declarative_base()
Session = scoped_session(sessionmaker(bind=engine))  # Use scoped_session here
session = Session()

Base = declarative_base()

# Define the version control table
class VersionControl(Base):
    __tablename__ = 'version_control'

    id = Column(Integer, primary_key=True)
    record_id = Column(Integer, nullable=False)
    table_name = Column(String, nullable=False)
    version_number = Column(Integer, nullable=False)
    timestamp = Column(DateTime, nullable=False, server_default=func.now())
    user_id = Column(Integer, ForeignKey('users.id'))  # Foreign key to the user table

# Define the trigger function
def version_control_trigger(target_table):
    def version_control_fn(context):
        # Extract necessary information from the context
        record_id = context.get_current_parameters()['id']
        table_name = target_table.__tablename__
        user_id = current_user.id  # Assuming current_user is available (e.g., through Flask-SQLAlchemy)

        # Determine the current version number
        current_version = session.query(func.max(VersionControl.version_number)).filter_by(record_id=record_id).scalar() or 0
        new_version = current_version + 1

        # Insert a new entry into the version control table
        version_entry = VersionControl(record_id=record_id, table_name=table_name, version_number=new_version, user_id=user_id)
        session.add(version_entry)
        session.commit()

    return version_control_fn


event.listen(MainTable1, 'after_insert', version_control_trigger(MainTable1))
event.listen(MainTable1, 'after_update', version_control_trigger(MainTable1))
event.listen(MainTable1, 'after_delete', version_control_trigger(MainTable1))

event.listen(MainTable2, 'after_insert', version_control_trigger(MainTable2))
event.listen(MainTable2, 'after_update', version_control_trigger(MainTable2))
event.listen(MainTable2, 'after_delete', version_control_trigger(MainTable2))

# Function to delete old versions beyond the specified limit
def delete_old_versions(session, record_id):
    num_versions = session.query(VersionControl).filter_by(record_id=record_id).count()
    num_versions_to_delete = max(0, num_versions - NUM_VERSIONS_TO_KEEP)
    if num_versions_to_delete > 0:
        oldest_versions = session.query(VersionControl).filter_by(record_id=record_id).order_by(VersionControl.version_number).limit(num_versions_to_delete).all()
        for version in oldest_versions:
            session.delete(version)
        session.commit()

# Trigger to execute the delete_old_versions function when a new version is added
def register_version_tracking_trigger():
    @event.listens_for(VersionControl, 'after_insert')
    def after_insert_listener(mapper, connection, target):
        session = sessionmaker(bind=connection.engine)()
        delete_old_versions(session, target.record_id)
        session.close()

Base.metadata.create_all(engine)