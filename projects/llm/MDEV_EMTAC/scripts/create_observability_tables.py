from modules.configuration.config_env import get_db_config
from modules.observability.models import Base

db = get_db_config()
engine = db.get_engine()

print("Creating emtac_observability tables...")
Base.metadata.create_all(engine)
print("Done.")