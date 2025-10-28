import time
from datetime import datetime

# Import the new PostgreSQL framework components
from modules.emtacdb.emtacdb_fts import User, KivyUser, UserLevel
from modules.configuration.config import ADMIN_CREATION_PASSWORD
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import (
    debug_id, info_id, warning_id, error_id,
    set_request_id, get_request_id, log_timed_operation
)
from modules.initial_setup.initializer_logger import (
    initializer_logger, close_initializer_logger,
    compress_logs_except_most_recent, LOG_DIRECTORY
)

# Ensure ADMIN_CREATION_PASSWORD is treated as a string
ADMIN_CREATION_PASSWORD = str(ADMIN_CREATION_PASSWORD)


class PostgreSQLAdminCreator:
    """PostgreSQL-enhanced admin user creator with comprehensive validation and user experience."""

    def __init__(self):
        self.request_id = set_request_id()
        self.db_config = DatabaseConfig()
        info_id("Initialized PostgreSQL Admin Creator", self.request_id)

        # Statistics tracking
        self.stats = {
            'users_created': 0,
            'users_already_exist': 0,
            'errors_encountered': 0,
            'processing_time': 0
        }

    def validate_admin_password(self, entered_password):
        """Validate the admin creation password with enhanced checking."""
        info_id("Validating admin creation password", self.request_id)

        if not ADMIN_CREATION_PASSWORD:
            error_id("Admin creation password not configured in settings", self.request_id)
            error_id("Admin creation password is not configured")
            info_id("Please check your configuration settings")
            return False

        # Clean and validate the entered password
        entered_password = str(entered_password).strip()
        expected_password = str(ADMIN_CREATION_PASSWORD).strip()

        debug_id(f"Password validation - Length: {len(entered_password)}, Expected length: {len(expected_password)}",
                 self.request_id)

        if entered_password != expected_password:
            error_id("Incorrect admin creation password provided", self.request_id)
            error_id("Incorrect password")
            return False

        info_id("Admin creation password validated successfully", self.request_id)
        return True

    def check_existing_users(self, session):
        """Check for existing admin users and provide user feedback."""
        try:
            info_id("Checking for existing admin users", self.request_id)
            info_id("Checking for existing admin users...")

            # Check for regular admin users
            existing_admin = session.query(User).filter_by(user_level=UserLevel.ADMIN).first()
            existing_kivy_admin = session.query(KivyUser).filter_by(user_level=UserLevel.ADMIN).first()

            # Count total users
            total_users = session.query(User).count()
            total_kivy_users = session.query(KivyUser).count()

            results = {
                'regular_admin_exists': existing_admin is not None,
                'kivy_admin_exists': existing_kivy_admin is not None,
                'total_users': total_users,
                'total_kivy_users': total_kivy_users,
                'existing_admin': existing_admin,
                'existing_kivy_admin': existing_kivy_admin
            }

            if existing_admin or existing_kivy_admin:
                warning_id("EXISTING ADMIN USERS FOUND")
                warning_id("=" * 35)

                if existing_admin:
                    info_id(f"Regular Admin: {existing_admin.employee_id} ({existing_admin.first_name} {existing_admin.last_name})")

                if existing_kivy_admin:
                    info_id(f"Kivy Admin: {existing_kivy_admin.employee_id} ({existing_kivy_admin.first_name} {existing_kivy_admin.last_name})")

                info_id(f"Total Users: {total_users}")
                if total_kivy_users > 0:
                    info_id(f"Total Kivy Users: {total_kivy_users}")

                info_id(f"Found existing admins - Regular: {bool(existing_admin)}, Kivy: {bool(existing_kivy_admin)}",
                        self.request_id)
            else:
                info_id("No existing admin users found")
                if total_users > 0 or total_kivy_users > 0:
                    info_id(f"Current users: {total_users} regular, {total_kivy_users} kivy users")
                info_id("No existing admin users found", self.request_id)

            return results

        except Exception as e:
            error_id(f"Error checking existing users: {str(e)}", self.request_id)
            error_id(f"Error checking existing users: {str(e)}")
            return None

    def create_regular_admin(self, session, user_check_results):
        """Create regular admin user with enhanced validation."""
        try:
            if user_check_results and user_check_results['regular_admin_exists']:
                existing_admin = user_check_results['existing_admin']
                info_id(f"Regular admin already exists: {existing_admin.employee_id}", self.request_id)
                info_id(f"Regular admin user already exists: {existing_admin.employee_id}")
                self.stats['users_already_exist'] += 1
                return True

            info_id("Creating regular admin user", self.request_id)
            info_id("Creating regular admin user...")

            with log_timed_operation("create_regular_admin", self.request_id):
                # Create the admin user
                admin_user = User(
                    employee_id='admin',
                    first_name='Admin',
                    last_name='User',
                    current_shift='Day',
                    primary_area='Administration',
                    age=30,
                    education_level='Masters',
                    start_date=datetime.utcnow(),
                    user_level=UserLevel.ADMIN
                )

                # Set password with validation
                admin_user.set_password('admin123')
                debug_id("Password set for regular admin user", self.request_id)

                # Add to session
                session.add(admin_user)
                session.flush()  # Get the ID without committing

                info_id(f"Created regular admin user with ID: {admin_user.id}", self.request_id)
                info_id(f"Created regular admin: {admin_user.employee_id}")
                info_id("Default password: admin123")
                info_id("Please change the password after first login")

                self.stats['users_created'] += 1
                return True

        except Exception as e:
            error_id(f"Error creating regular admin user: {str(e)}", self.request_id, exc_info=True)
            error_id(f"Error creating regular admin: {str(e)}")
            self.stats['errors_encountered'] += 1
            return False

    def create_kivy_admin(self, session, user_check_results):
        """Create Kivy admin user with enhanced validation."""
        try:
            if user_check_results and user_check_results['kivy_admin_exists']:
                existing_kivy_admin = user_check_results['existing_kivy_admin']
                info_id(f"Kivy admin already exists: {existing_kivy_admin.employee_id}", self.request_id)
                info_id(f"Kivy admin user already exists: {existing_kivy_admin.employee_id}")
                self.stats['users_already_exist'] += 1
                return True

            info_id("Creating Kivy admin user", self.request_id)
            info_id("Creating Kivy admin user...")

            with log_timed_operation("create_kivy_admin", self.request_id):
                # Create the Kivy admin user
                kivy_admin = KivyUser(
                    employee_id='kivyadmin',
                    first_name='Kivy',
                    last_name='Admin',
                    current_shift='Day',
                    primary_area='Administration',
                    age=30,
                    education_level='Masters',
                    start_date=datetime.utcnow(),
                    user_level=UserLevel.ADMIN
                )

                # Set password with validation
                kivy_admin.set_password('admin123')
                debug_id("Password set for Kivy admin user", self.request_id)

                # Add to session
                session.add(kivy_admin)
                session.flush()  # Get the ID without committing

                info_id(f"Created Kivy admin user with ID: {kivy_admin.id}", self.request_id)
                info_id(f"Created Kivy admin: {kivy_admin.employee_id}")
                info_id("Default password: admin123")
                info_id("Please change the password after first login")

                self.stats['users_created'] += 1
                return True

        except Exception as e:
            error_id(f"Error creating Kivy admin user: {str(e)}", self.request_id, exc_info=True)
            error_id(f"Error creating Kivy admin: {str(e)}")
            self.stats['errors_encountered'] += 1
            return False

    def display_final_summary(self):
        """Display comprehensive admin creation summary."""
        info_id("ADMIN CREATION COMPLETE!")
        info_id("=" * 35)
        info_id("Final Summary:")
        info_id(f"   Users created: {self.stats['users_created']}")
        info_id(f"   Users already existed: {self.stats['users_already_exist']}")

        if self.stats['errors_encountered'] > 0:
            warning_id(f"   Errors encountered: {self.stats['errors_encountered']}")

        if self.stats['processing_time'] > 0:
            info_id(f"   Processing time: {self.stats['processing_time']:.2f}s")

        info_id("=" * 35)

        if self.stats['users_created'] > 0:
            info_id("IMPORTANT SECURITY NOTES:")
            info_id("   - Default password is 'admin123'")
            info_id("   - Change passwords immediately after first login")
            info_id("   - Use strong, unique passwords for production")
            info_id("   - Consider enabling two-factor authentication")

        info_id(f"Admin creation summary: {self.stats}", self.request_id)

    def create_admin_users(self, admin_password):
        """Main method to create admin users with comprehensive handling."""
        try:
            info_id("Initial Admin User Creation")
            info_id("=" * 35)

            start_time = time.time()

            # Validate password
            if not self.validate_admin_password(admin_password):
                return False

            # Get database session using new framework
            with self.db_config.main_session() as session:
                # Check existing users
                user_check_results = self.check_existing_users(session)
                if user_check_results is None:
                    return False

                # Determine what needs to be created
                needs_regular_admin = not user_check_results['regular_admin_exists']
                needs_kivy_admin = not user_check_results['kivy_admin_exists']

                if not needs_regular_admin and not needs_kivy_admin:
                    info_id("All admin users already exist")
                    info_id("All admin users already exist, nothing to create", self.request_id)
                    self.stats['users_already_exist'] = 2
                else:
                    info_id("Creating missing admin users...")

                    # Create regular admin if needed
                    if needs_regular_admin:
                        self.create_regular_admin(session, user_check_results)

                    # Create Kivy admin if needed
                    if needs_kivy_admin:
                        self.create_kivy_admin(session, user_check_results)

                    # Commit all changes
                    try:
                        session.commit()
                        info_id("All admin user changes committed successfully", self.request_id)
                        info_id("All changes saved successfully")
                    except Exception as e:
                        session.rollback()
                        error_id(f"Error committing admin user changes: {str(e)}", self.request_id)
                        error_id(f"Error saving changes: {str(e)}")
                        return False

            # Update final statistics
            self.stats['processing_time'] = time.time() - start_time

            # Display summary
            self.display_final_summary()

            info_id("Admin user creation completed successfully", self.request_id)
            return True

        except Exception as e:
            error_id(f"Admin user creation failed: {str(e)}", self.request_id, exc_info=True)
            error_id(f"Admin creation failed: {str(e)}")
            return False


def prompt_for_admin_password():
    """Enhanced password prompting with validation."""
    info_id("Admin Creation Authentication")
    info_id("=" * 35)
    info_id("Enter the admin creation password to proceed.")
    info_id("This is configured in your application settings.")
    info_id("")

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        admin_password = input(f"Admin creation password (attempt {attempt}/{max_attempts}): ").strip()

        if admin_password:
            return admin_password
        else:
            warning_id("Please enter a password")
            if attempt < max_attempts:
                info_id(f"{max_attempts - attempt} attempts remaining")

    error_id("Maximum attempts exceeded")
    return None


def main():
    """
    Main function to handle the admin creation process using PostgreSQL framework.
    """
    info_id("Starting Initial Admin User Creation")
    info_id("=" * 45)

    creator = None
    try:
        # Initialize the PostgreSQL admin creator
        creator = PostgreSQLAdminCreator()

        # Prompt for admin password
        admin_password = prompt_for_admin_password()
        if not admin_password:
            error_id("Admin creation cancelled - no valid password provided")
            return

        # Create admin users
        success = creator.create_admin_users(admin_password)

        if success:
            info_id("Initial Admin Creation Completed Successfully!")
            info_id("=" * 45)
        else:
            warning_id("Initial Admin Creation Completed with Issues")
            info_id("=" * 45)

    except KeyboardInterrupt:
        warning_id("Admin creation interrupted by user")
        if creator:
            error_id("Admin creation interrupted by user", creator.request_id)
    except Exception as e:
        error_id(f"Admin creation failed: {str(e)}")
        if creator:
            error_id(f"Admin creation failed: {str(e)}", creator.request_id, exc_info=True)
    finally:
        # Cleanup and logging
        try:
            close_initializer_logger()
            compress_logs_except_most_recent(LOG_DIRECTORY)
        except:
            pass


if __name__ == '__main__':
    main()