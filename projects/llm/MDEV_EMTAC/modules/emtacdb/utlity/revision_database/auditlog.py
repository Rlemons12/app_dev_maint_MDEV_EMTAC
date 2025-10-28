import os
import logging
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, String, JSON, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import scoped_session, sessionmaker
from modules.emtacdb.utlity.revision_database.snapshot_utils import (
    create_snapshot
)
from modules.configuration.config import DATABASE_DIR
from threading import Lock

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Revision control database configuration
REVISION_CONTROL_DB_PATH = os.path.join(DATABASE_DIR, 'emtac_revision_control_db.db')
revision_control_engine = create_engine(f'sqlite:///{REVISION_CONTROL_DB_PATH}')
RevisionControlBase = declarative_base()
RevisionControlSession = scoped_session(sessionmaker(bind=revision_control_engine))  # Use distinct name

# Temporary list to store audit log entries
audit_log_entries = []
audit_log_lock = Lock()

def commit_audit_logs():
    """
    Commit all pending audit log entries to the database.
    This function should be called after all operations have completed.
    """
    with audit_log_lock:  # Ensure thread safety
        if audit_log_entries:
            logger.info(f"Committing {len(audit_log_entries)} audit log entries to the database.")
            with RevisionControlSession() as session:
                try:
                    session.bulk_save_objects(audit_log_entries)
                    session.commit()
                    logger.info(f"Successfully committed {len(audit_log_entries)} audit log entries.")
                    audit_log_entries.clear()  # Clear the entries after committing
                except Exception as e:
                    logger.error(f"Failed to commit audit logs: {e}")
                    session.rollback()  # Rollback in case of error
        else:
            logger.info("No audit log entries to commit.")

def add_audit_log_entry(table_name, operation, record_id, old_data=None, new_data=None, commit_to_db=False):
    entry = {
        'table_name': table_name,
        'operation': operation,
        'record_id': record_id,
        'old_data': old_data,
        'new_data': new_data,
        'changed_at': datetime.utcnow()
    }

    with audit_log_lock:
        # Append to the temporary in-memory list
        audit_log_entries.append(entry)
        logger.info(f"Audit log entry added to memory: {entry}")

        if commit_to_db:
            try:
                # Optionally commit this entry to the database immediately
                with RevisionControlSession() as session:
                    audit_log = AuditLog(
                        table_name=entry['table_name'],
                        operation=entry['operation'],
                        record_id=entry['record_id'],
                        old_data=entry['old_data'],
                        new_data=entry['new_data'],
                        changed_at=entry['changed_at']
                    )
                    session.add(audit_log)
                    session.commit()
                    logger.info(f"Audit log entry committed to database: {entry}")
            except Exception as e:
                logger.error(f"Failed to commit audit log entry to database: {e}")


def get_serializable_data(instance):
    """
    Returns a dictionary of serializable fields from the SQLAlchemy instance.
    Excludes the '_sa_instance_state' and other non-serializable attributes.
    """
    data = instance.__dict__.copy()
    data.pop('_sa_instance_state', None)
    return data

class AuditLog(RevisionControlBase):
    __tablename__ = 'audit_log'

    id = Column(Integer, primary_key=True, autoincrement=True)
    table_name = Column(String, nullable=False)
    operation = Column(String, nullable=False)
    record_id = Column(Integer, nullable=False)
    old_data = Column(JSON, nullable=True)
    new_data = Column(JSON, nullable=True)
    changed_at = Column(DateTime, default=datetime.utcnow, nullable=False)


import traceback

def log_insert(mapper, connection, target, SnapshotClass=None, *args, **kwargs):
    # Log the call stack
    stack = traceback.format_stack()
    logger.info("log_insert called by:\n" + "".join(stack))
    
    table_name = target.__tablename__
    new_data = {c.name: getattr(target, c.name) for c in target.__table__.columns}

    if table_name == 'complete_document':
        target.rev = 'R0'  # Set initial revision number

    try:
        # Add to in-memory audit log
        add_audit_log_entry(
            table_name=table_name,
            operation='INSERT',
            record_id=new_data.get('id'),
            new_data=new_data,
            commit_to_db=False  # Don't commit to DB yet, keep in memory
        )

        if SnapshotClass:
            with RevisionControlSession() as session:
                create_snapshot(target, session, SnapshotClass)

        logger.info(f"Inserted record for {table_name} with data: {new_data}")
    except Exception as e:
        logger.error(f"An error occurred during log_insert: {e}")



def log_update(mapper, connection, target, SnapshotClass=None, *args, **kwargs):
    # Log the call stack
    stack = traceback.format_stack()
    logger.info("log_update called by:\n" + "".join(stack))
    
    # Always use RevisionControlSession
    session = RevisionControlSession()
    
    # Log the session's bound engine (which gives insight into the database)
    logger.info(f"Using session bound to engine: {session.bind}")
    
    # Log the connection's information (to understand which database it is connected to)
    logger.info(f"Using connection: {connection} connected to database: {connection.engine.url}")
    
    table_name = target.__tablename__
    old_instance = session.query(mapper.class_).get(mapper.primary_key_from_instance(target))
    old_data = get_serializable_data(old_instance)
    new_data = {c.name: getattr(target, c.name) for c in target.__table__.columns}

    if table_name == 'complete_document':
        current_rev = target.rev
        if current_rev:
            rev_number = int(current_rev[1:])  
            target.rev = f'R{rev_number + 1}'

    try:
        audit_log = AuditLog(
            table_name=table_name,
            operation='UPDATE',
            record_id=new_data.get('id'),
            old_data=old_data,
            new_data=new_data
        )
        session.add(audit_log)
        session.commit()
        
        if SnapshotClass:
            create_snapshot(target, session, SnapshotClass)
            
        logger.info(f"Updated record in audit_log and created snapshot for {table_name} with data: {new_data}")
    except Exception as e:
        logger.error(f"An error occurred during log_update: {e}")
        session.rollback()
    finally:
        session.close()

def log_delete(mapper, connection, target, SnapshotClass=None, *args, **kwargs):
    # Log the call stack
    stack = traceback.format_stack()
    logger.info("log_delete called by:\n" + "".join(stack))
    
    # Always use RevisionControlSession
    session = RevisionControlSession()
    
    # Log the session's bound engine (which gives insight into the database)
    logger.info(f"Using session bound to engine: {session.bind}")
    
    # Log the connection's information (to understand which database it is connected to)
    logger.info(f"Using connection: {connection} connected to database: {connection.engine.url}")
    
    table_name = target.__tablename__
    old_data = {c.name: getattr(target, c.name) for c in target.__table__.columns}

    try:
        audit_log = AuditLog(
            table_name=table_name,
            operation='DELETE',
            record_id=old_data.get('id'),
            old_data=old_data
        )
        session.add(audit_log)
        session.commit()
        
        if SnapshotClass:
            create_snapshot(target, session, SnapshotClass)
            
        logger.info(f"Deleted record from audit_log and created snapshot for {table_name} with data: {old_data}")
    except Exception as e:
        logger.error(f"An error occurred during log_delete: {e}")
        session.rollback()
    finally:
        session.close()


def log_event_listeners(entity_name):
    """
    Logs the setup of event listeners for a given entity.
    """
    logger.info(f"Setting up event listeners for {entity_name}.")