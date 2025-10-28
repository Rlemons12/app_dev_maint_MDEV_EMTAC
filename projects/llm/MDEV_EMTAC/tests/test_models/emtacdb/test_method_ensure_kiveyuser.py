# test_kivy_user.py
from modules.emtacdb.emtacdb_fts import User, KivyUser
from modules.configuration.config_env import DatabaseConfig

# Get a session
db_config = DatabaseConfig()
session = db_config.get_main_session()

# Check if KivyUser class has the ensure_kivy_user method
print("Available methods on KivyUser:", [method for method in dir(KivyUser) if not method.startswith('_')])

# Check if User class has the ensure_kivy_user method (it shouldn't)
print("Available methods on User:", [method for method in dir(User) if not method.startswith('_')])

# Find a regular User
user = session.query(User).first()
if user:
    print(f"Found User: {user.employee_id} (ID: {user.id}, Type: {user.type})")

    # Try to use the method
    try:
        kivy_user = KivyUser.ensure_kivy_user(session, user)
        if kivy_user:
            print(f"Successfully got KivyUser: {kivy_user.employee_id} (ID: {kivy_user.id})")
        else:
            print("ensure_kivy_user returned None")
    except Exception as e:
        print(f"Error calling ensure_kivy_user: {type(e).__name__}: {e}")
else:
    print("No User found in database")