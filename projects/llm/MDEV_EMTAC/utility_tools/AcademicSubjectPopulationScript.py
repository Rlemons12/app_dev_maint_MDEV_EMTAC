#!/usr/bin/env python3
"""
Academic Subject Population Script - Updated for PostgreSQL Framework

This script populates the database with academic subjects using a restructured hierarchy:
- Area → "Academic" (top level container)
- EquipmentGroup → Academic Fields (Physics, Business, etc.)
- Model → Subjects (Marketing, Mechanics, etc.)
- (and so on down the hierarchy)

Now uses the enhanced PostgreSQL framework with:
- Improved connection management
- Context managers for better session handling
- PostgreSQL-specific optimizations
- Enhanced logging and error handling

Usage:
    python AcademicSubjectPopulationScript.py
"""

import os
import sys
import traceback
from sqlalchemy.exc import SQLAlchemyError
from modules.database_manager.db_manager import PostgreSQLDatabaseManager
# Debug - Print current information
print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
print(f"Current working directory: {os.getcwd()}")
print(f"Python path: {sys.path}")

# Add the project root to the Python path if needed
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
    print(f"Added project root to sys.path: {project_root}")

try:
    # Import project configuration, database, and logging
    from modules.configuration.config_env import DatabaseConfig

    print("Successfully imported DatabaseConfig")

    from modules.configuration.log_config import logger, with_request_id, info_id, error_id, debug_id, set_request_id

    print("Successfully imported logging modules")

    # Import the new PostgreSQL database managers


    print("Successfully imported PostgreSQL database manager")

    # Import your models - adjust this import path as needed
    try:
        from modules.emtacdb.emtacdb_fts import Area, EquipmentGroup, Model

        print("Successfully imported database models")
    except ImportError as e:
        print(f"Failed to import database models: {e}")
        print("Trying alternative import paths...")

        # Try alternative import locations
        try:
            from modules.emtacdb.models import Area, EquipmentGroup, Model

            print("Successfully imported models from modules.emtacdb.models")
        except ImportError:
            print("Failed to import from modules.emtacdb.models")

            # Another potential location
            try:
                from modules.emtacdb.emtacdb_models import Area, EquipmentGroup, Model

                print("Successfully imported models from modules.emtacdb.emtacdb_models")
            except ImportError:
                print("Failed to import from modules.emtacdb.emtacdb_models")
                raise
except Exception as e:
    print(f"Import error: {e}")
    print("Traceback:")
    traceback.print_exc()
    sys.exit(1)

print("All imports successful!")

# Initialize DatabaseConfig - ONLY ONCE
try:
    db_config = DatabaseConfig()
    print("DatabaseConfig initialized successfully")

    # Check if we're using PostgreSQL
    if db_config.is_postgresql:
        print("✓ Using PostgreSQL database")
    else:
        print("ℹ Using SQLite database")

    # Display connection stats
    stats = db_config.get_connection_stats()
    print(f"Database type: {stats['database_type']}")
    print(f"Connection limiting enabled: {stats['connection_limiting_enabled']}")
    print(f"Max concurrent connections: {stats['max_concurrent_connections']}")

except Exception as e:
    print(f"Error initializing DatabaseConfig: {e}")
    traceback.print_exc()
    sys.exit(1)


class AcademicSubjectManager(PostgreSQLDatabaseManager):
    """Enhanced manager for academic subject operations using PostgreSQL framework."""

    def __init__(self, db_config, request_id=None):
        """Initialize with database config and request ID."""
        self.db_config = db_config
        self.request_id = request_id or set_request_id()
        # Initialize parent with None session - we'll use context managers
        super().__init__(session=None, request_id=self.request_id)

    def add_area(self, name, description=None):
        """Add an academic field (Area) using context manager."""
        try:
            with self.db_config.main_session() as session:
                # Check if area already exists
                existing = session.query(Area).filter(Area.name == name).first()
                if existing:
                    info_id(f"Area '{name}' already exists with ID {existing.id}", self.request_id)
                    return existing.id

                # Create new area
                area = Area(name=name, description=description)
                session.add(area)
                session.flush()  # Get the ID before commit
                area_id = area.id
                info_id(f"Added area: {name} (ID: {area_id})", self.request_id)
                return area_id

        except Exception as e:
            error_id(f"Error adding area '{name}': {str(e)}", self.request_id)
            print(f"Error adding area '{name}': {str(e)}")
            traceback.print_exc()
            raise

    def add_equipment_group(self, name, area_id, description=None):
        """Add a subject (EquipmentGroup) using context manager."""
        try:
            with self.db_config.main_session() as session:
                # Check if subject already exists
                existing = session.query(EquipmentGroup).filter(
                    EquipmentGroup.name == name,
                    EquipmentGroup.area_id == area_id
                ).first()
                if existing:
                    info_id(f"Subject '{name}' already exists with ID {existing.id}", self.request_id)
                    return existing.id

                # Create new subject
                subject = EquipmentGroup(name=name, area_id=area_id, description=description)
                session.add(subject)
                session.flush()  # Get the ID before commit
                subject_id = subject.id
                info_id(f"Added subject: {name} (ID: {subject_id}) to area ID {area_id}", self.request_id)
                return subject_id

        except Exception as e:
            error_id(f"Error adding subject '{name}': {str(e)}", self.request_id)
            print(f"Error adding subject '{name}': {str(e)}")
            traceback.print_exc()
            raise

    def add_model(self, name, equipment_group_id, description=None):
        """Add a branch/subdiscipline (Model) using context manager."""
        try:
            with self.db_config.main_session() as session:
                # Check if branch already exists
                existing = session.query(Model).filter(
                    Model.name == name,
                    Model.equipment_group_id == equipment_group_id
                ).first()
                if existing:
                    info_id(f"Branch '{name}' already exists with ID {existing.id}", self.request_id)
                    return existing.id

                # Create new branch
                branch = Model(name=name, equipment_group_id=equipment_group_id, description=description)
                session.add(branch)
                session.flush()  # Get the ID before commit
                branch_id = branch.id
                info_id(f"Added branch: {name} (ID: {branch_id}) to subject ID {equipment_group_id}", self.request_id)
                return branch_id

        except Exception as e:
            error_id(f"Error adding branch '{name}': {str(e)}", self.request_id)
            print(f"Error adding branch '{name}': {str(e)}")
            traceback.print_exc()
            raise

    def populate_field_batch(self, session, field_name, area_id, subjects_data):
        """Populate a complete academic field using the provided session."""
        try:
            # Create the equipment group
            equipment_group = EquipmentGroup(
                name=field_name,
                area_id=area_id,
                description=subjects_data.get('description', f"Study of {field_name.lower()}")
            )
            session.add(equipment_group)
            session.flush()  # Get ID

            # Create all subjects for this field
            models = []
            for subject_data in subjects_data.get('subjects', []):
                model = Model(
                    name=subject_data['name'],
                    equipment_group_id=equipment_group.id,
                    description=subject_data.get('description', '')
                )
                models.append(model)

            if models:
                session.add_all(models)
                session.flush()
                info_id(f"Added {len(models)} subjects to {field_name}", self.request_id)

            return equipment_group.id

        except Exception as e:
            error_id(f"Error populating field {field_name}: {str(e)}", self.request_id)
            raise


def populate_business_field_efficient(manager, session, academic_area_id):
    """Populate business field using efficient batched approach."""
    business_data = {
        'description': 'Study of management and operation of business enterprises',
        'subjects': [
            {'name': 'Marketing', 'description': 'Study of promoting and selling products or services'},
            {'name': 'Finance', 'description': 'Study of money management and asset allocation'},
            {'name': 'Management', 'description': 'Study of organizational leadership and administration'},
            {'name': 'Accounting',
             'description': 'Study of recording, classifying, and summarizing financial transactions'},
            {'name': 'Human Resources', 'description': 'Study of managing an organization\'s workforce'},
            {'name': 'Operations Management', 'description': 'Study of designing and controlling production processes'},
            {'name': 'Business Analytics', 'description': 'Study of data analysis techniques for business insights'},
            {'name': 'Entrepreneurship', 'description': 'Study of starting and running new business ventures'},
            {'name': 'International Business', 'description': 'Study of global business operations and strategies'},
        ]
    }

    return manager.populate_field_batch(session, "Business Administration", academic_area_id, business_data)


def populate_science_fields_efficient(manager, session, academic_area_id):
    """Populate science fields using efficient batched approach."""
    fields_data = {
        'Physics': {
            'description': 'Study of matter, energy, and their interactions',
            'subjects': [
                {'name': 'Mechanics', 'description': 'Study of motion and forces'},
                {'name': 'Quantum Physics', 'description': 'Study of subatomic particles and their behaviors'},
                {'name': 'Thermodynamics', 'description': 'Study of heat, energy, and work'},
                {'name': 'Electromagnetism', 'description': 'Study of electromagnetic force and fields'},
                {'name': 'Relativity', 'description': 'Study of space, time, and gravity'},
            ]
        },
        'Chemistry': {
            'description': 'Study of substances, their properties, and reactions',
            'subjects': [
                {'name': 'Organic Chemistry', 'description': 'Study of carbon-containing compounds'},
                {'name': 'Inorganic Chemistry', 'description': 'Study of non-carbon compounds'},
                {'name': 'Physical Chemistry',
                 'description': 'Study of how matter behaves on a molecular and atomic level'},
                {'name': 'Biochemistry', 'description': 'Study of chemical processes within living organisms'},
                {'name': 'Analytical Chemistry',
                 'description': 'Study of separation, identification, and quantification of matter'},
            ]
        },
        'Biology': {
            'description': 'Study of living organisms',
            'subjects': [
                {'name': 'Molecular Biology', 'description': 'Study of biological activity at the molecular level'},
                {'name': 'Genetics', 'description': 'Study of genes, heredity, and genetic variation'},
                {'name': 'Ecology', 'description': 'Study of interactions between organisms and their environment'},
            ]
        }
    }

    created_ids = []
    for field_name, field_data in fields_data.items():
        field_id = manager.populate_field_batch(session, field_name, academic_area_id, field_data)
        created_ids.append(field_id)

    return created_ids


def populate_mathematics_fields_efficient(manager, session, academic_area_id):
    """Populate mathematics fields using efficient batched approach."""
    fields_data = {
        'Pure Mathematics': {
            'description': 'Study of abstract concepts and structures',
            'subjects': [
                {'name': 'Algebra', 'description': 'Study of mathematical symbols and rules'},
                {'name': 'Calculus', 'description': 'Study of continuous change and functions'},
                {'name': 'Geometry', 'description': 'Study of shapes, sizes, and properties of space'},
                {'name': 'Number Theory', 'description': 'Study of integers and integer-valued functions'},
                {'name': 'Topology', 'description': 'Study of properties preserved under continuous deformations'},
            ]
        },
        'Applied Mathematics': {
            'description': 'Application of mathematical methods to solve real-world problems',
            'subjects': [
                {'name': 'Statistics', 'description': 'Study of data collection, analysis, and interpretation'},
                {'name': 'Operations Research', 'description': 'Application of analytical methods for decision-making'},
                {'name': 'Mathematical Modeling', 'description': 'Using mathematics to describe real-world phenomena'},
                {'name': 'Financial Mathematics', 'description': 'Application of mathematical methods in finance'},
            ]
        }
    }

    created_ids = []
    for field_name, field_data in fields_data.items():
        field_id = manager.populate_field_batch(session, field_name, academic_area_id, field_data)
        created_ids.append(field_id)

    return created_ids


def populate_computer_science_efficient(manager, session, academic_area_id):
    """Populate computer science field using efficient batched approach."""
    cs_data = {
        'description': 'Study of computation, algorithms, and information processing',
        'subjects': [
            {'name': 'Algorithms', 'description': 'Study of computational procedures and problem-solving methods'},
            {'name': 'Data Structures', 'description': 'Study of organizing and storing data efficiently'},
            {'name': 'Artificial Intelligence',
             'description': 'Study of intelligent agent development and machine learning'},
            {'name': 'Database Systems', 'description': 'Study of data organization, storage, and retrieval methods'},
            {'name': 'Computer Networks', 'description': 'Study of data communication systems and protocols'},
            {'name': 'Software Engineering', 'description': 'Study of systematic development of software applications'},
            {'name': 'Cybersecurity',
             'description': 'Study of protecting computer systems from unauthorized access and attacks'},
            {'name': 'Theoretical Computer Science', 'description': 'Mathematical study of computation and algorithms'},
            {'name': 'Human-Computer Interaction', 'description': 'Study of interaction between humans and computers'},
        ]
    }

    return manager.populate_field_batch(session, "Computer Science", academic_area_id, cs_data)


def populate_humanities_fields_efficient(manager, session, academic_area_id):
    """Populate humanities fields using efficient batched approach."""
    fields_data = {
        'Philosophy': {
            'description': 'Study of fundamental questions about existence, knowledge, ethics, and more',
            'subjects': [
                {'name': 'Ethics', 'description': 'Study of moral principles and values'},
                {'name': 'Epistemology', 'description': 'Study of knowledge, belief, and justification'},
                {'name': 'Metaphysics', 'description': 'Study of reality, existence, and being'},
                {'name': 'Logic', 'description': 'Study of valid reasoning and inference'},
            ]
        },
        'History': {
            'description': 'Study of past events, societies, and civilizations',
            'subjects': [
                {'name': 'World History', 'description': 'Study of history on a global scale'},
                {'name': 'U.S. History', 'description': 'Study of United States history'},
                {'name': 'Ancient History', 'description': 'Study of early human civilizations'},
                {'name': 'Medieval History', 'description': 'Study of the Middle Ages'},
                {'name': 'Modern History', 'description': 'Study of recent centuries of human history'},
            ]
        },
        'Literature': {
            'description': 'Study of written works of art',
            'subjects': [
                {'name': 'English Literature', 'description': 'Study of literature written in English'},
                {'name': 'World Literature', 'description': 'Study of literature from various cultures and languages'},
                {'name': 'Poetry', 'description': 'Study of poetic works and forms'},
                {'name': 'Drama', 'description': 'Study of theatrical works'},
            ]
        }
    }

    created_ids = []
    for field_name, field_data in fields_data.items():
        field_id = manager.populate_field_batch(session, field_name, academic_area_id, field_data)
        created_ids.append(field_id)

    return created_ids


def populate_social_sciences_efficient(manager, session, academic_area_id):
    """Populate social sciences fields using efficient batched approach."""
    fields_data = {
        'Psychology': {
            'description': 'Study of mind and behavior',
            'subjects': [
                {'name': 'Clinical Psychology', 'description': 'Study and treatment of mental illness and distress'},
                {'name': 'Cognitive Psychology',
                 'description': 'Study of mental processes including perception, thinking, and memory'},
                {'name': 'Developmental Psychology',
                 'description': 'Study of psychological growth across the lifespan'},
                {'name': 'Social Psychology',
                 'description': 'Study of how individuals\' thoughts and behavior are influenced by others'},
            ]
        },
        'Economics': {
            'description': 'Study of production, distribution, and consumption of goods and services',
            'subjects': [
                {'name': 'Microeconomics',
                 'description': 'Study of individual and business decisions regarding resource allocation'},
                {'name': 'Macroeconomics',
                 'description': 'Study of economy-wide phenomena like inflation, growth, and unemployment'},
                {'name': 'International Economics', 'description': 'Study of economic interactions between countries'},
                {'name': 'Econometrics', 'description': 'Application of statistical methods to economic data'},
            ]
        },
        'Sociology': {
            'description': 'Study of society, social relationships, and culture',
            'subjects': [
                {'name': 'Cultural Sociology', 'description': 'Study of the influence of culture on social life'},
                {'name': 'Urban Sociology',
                 'description': 'Study of social life and interactions in urban environments'},
                {'name': 'Social Inequality', 'description': 'Study of social differences and hierarchies'},
            ]
        }
    }

    created_ids = []
    for field_name, field_data in fields_data.items():
        field_id = manager.populate_field_batch(session, field_name, academic_area_id, field_data)
        created_ids.append(field_id)

    return created_ids


def populate_industrial_manufacturing_efficient(manager, session, academic_area_id):
    """Populate industrial manufacturing fields using efficient batched approach."""
    fields_data = {
        'Welding Technology': {
            'description': 'Study of joining materials through fusion processes',
            'subjects': [
                {'name': 'Arc Welding', 'description': 'Welding processes using an electric arc (SMAW, GMAW, GTAW)'},
                {'name': 'Resistance Welding',
                 'description': 'Welding processes that use electrical resistance to generate heat'},
                {'name': 'Oxyfuel Welding', 'description': 'Welding using fuel gases and oxygen to produce a flame'},
                {'name': 'Welding Metallurgy', 'description': 'Study of metal properties and behaviors during welding'},
                {'name': 'Welding Inspection', 'description': 'Quality control and testing of welded joints'},
                {'name': 'Welding Automation', 'description': 'Automated and robotic welding systems and programming'},
            ]
        },
        'Machining Technology': {
            'description': 'Study of material removal processes to create parts',
            'subjects': [
                {'name': 'CNC Machining',
                 'description': 'Computer numerical control machining processes and programming'},
                {'name': 'Manual Machining',
                 'description': 'Traditional machine tool operation (lathes, mills, drill presses)'},
                {'name': 'Precision Measurement',
                 'description': 'Metrology techniques and instruments for machined parts'},
                {'name': 'CAD/CAM Systems', 'description': 'Computer-aided design and manufacturing for machining'},
                {'name': 'Advanced Machining Processes',
                 'description': 'Non-traditional processes like EDM, waterjet, and laser cutting'},
                {'name': 'Tool Design', 'description': 'Design of cutting tools, fixtures, and machine tooling'},
            ]
        },
        'Industrial Electrical Systems': {
            'description': 'Study of electrical systems in industrial settings',
            'subjects': [
                {'name': 'Power Distribution', 'description': 'Industrial power systems and distribution networks'},
                {'name': 'Motor Controls', 'description': 'Electric motor operation, control, and protection systems'},
                {'name': 'Industrial Controls', 'description': 'PLCs, SCADA, and other industrial control systems'},
                {'name': 'Electrical Troubleshooting',
                 'description': 'Diagnosing and repairing electrical system faults'},
                {'name': 'Electrical Safety',
                 'description': 'Hazard identification and safety practices for electrical work'},
                {'name': 'Industrial IoT', 'description': 'Internet of Things applications in industrial settings'},
            ]
        },
        'Fluid Power Systems': {
            'description': 'Study of hydraulic and pneumatic systems for power transmission',
            'subjects': [
                {'name': 'Hydraulic Systems',
                 'description': 'Liquid-based power transmission systems design and operation'},
                {'name': 'Pneumatic Systems', 'description': 'Compressed air power systems design and operation'},
                {'name': 'Fluid Power Components',
                 'description': 'Pumps, valves, actuators, and other hydraulic/pneumatic components'},
                {'name': 'Fluid Power Maintenance',
                 'description': 'Troubleshooting, repair, and preventive maintenance'},
                {'name': 'Electrohydraulic Systems',
                 'description': 'Integration of electronics with hydraulic systems'},
                {'name': 'Fluid Power Circuit Design',
                 'description': 'Designing and analyzing hydraulic and pneumatic circuits'},
            ]
        }
    }

    created_ids = []
    for field_name, field_data in fields_data.items():
        field_id = manager.populate_field_batch(session, field_name, academic_area_id, field_data)
        created_ids.append(field_id)

    return created_ids


@with_request_id
def verify_database(request_id=None):
    """Verify that the database is ready for population."""
    try:
        print("Getting database session...")
        with db_config.main_session() as session:
            print("Database session acquired using context manager")

            print("Verifying database tables...")
            area_count = session.query(Area).count()
            print(f"Found {area_count} areas in database")
            equipment_group_count = session.query(EquipmentGroup).count()
            print(f"Found {equipment_group_count} equipment groups in database")
            model_count = session.query(Model).count()
            print(f"Found {model_count} models in database")

            debug_id(
                f"Found {area_count} areas, {equipment_group_count} equipment groups, and {model_count} models",
                request_id
            )
            return True

    except SQLAlchemyError as e:
        error_id(f"Error verifying database schema: {str(e)}", request_id)
        print("Database tables not found or accessible. Please ensure the database is properly set up.")
        print(f"Error details: {str(e)}")
        traceback.print_exc()
        return False
    except Exception as e:
        error_id(f"Failed to connect to database: {str(e)}", request_id)
        print(f"Database connection failed: {str(e)}")
        traceback.print_exc()
        return False


@with_request_id
def main(request_id=None):
    """Main function to populate all academic fields using efficient PostgreSQL framework."""
    print("Main function started with enhanced PostgreSQL framework")
    info_id("Starting academic subject population with PostgreSQL optimizations...", request_id)
    print("Academic Subject Population Script - PostgreSQL Enhanced")
    print("======================================================")

    # Verify database is ready
    print("Verifying database...")
    if not verify_database():
        error_id("Database verification failed. Exiting.", request_id)
        print("Database verification failed. Please check your database setup.")
        return

    try:
        # Initialize the academic subject manager
        manager = AcademicSubjectManager(db_config, request_id)
        print("Academic Subject Manager initialized")

        # Use single transaction approach for optimal performance
        print("\nUsing single transaction approach for optimal performance...")

        with db_config.main_session() as session:
            print("Creating Academic area...")

            # Check if Academic area already exists
            existing_academic = session.query(Area).filter(Area.name == "Academic").first()
            if existing_academic:
                academic_area_id = existing_academic.id
                print(f"✓ Academic area already exists with ID: {academic_area_id}")
            else:
                academic_area = Area(
                    name="Academic",
                    description="Academic and technical knowledge across various fields"
                )
                session.add(academic_area)
                session.flush()
                academic_area_id = academic_area.id
                print(f"✓ Created Academic area with ID: {academic_area_id}")

            print("\nPopulating fields in batches for efficiency...")

            # Use the efficient batched approach - pass session to all functions
            business_id = populate_business_field_efficient(manager, session, academic_area_id)
            print("✓ Business field populated efficiently")

            science_ids = populate_science_fields_efficient(manager, session, academic_area_id)
            print("✓ Science fields populated efficiently")

            math_ids = populate_mathematics_fields_efficient(manager, session, academic_area_id)
            print("✓ Mathematics fields populated efficiently")

            cs_id = populate_computer_science_efficient(manager, session, academic_area_id)
            print("✓ Computer Science field populated efficiently")

            humanities_ids = populate_humanities_fields_efficient(manager, session, academic_area_id)
            print("✓ Humanities fields populated efficiently")

            social_science_ids = populate_social_sciences_efficient(manager, session, academic_area_id)
            print("✓ Social Sciences fields populated efficiently")

            industrial_ids = populate_industrial_manufacturing_efficient(manager, session, academic_area_id)
            print("✓ Industrial Manufacturing fields populated efficiently")

            print("\nCommitting all changes...")
            # The context manager will automatically commit when exiting

        info_id("Successfully populated all fields using PostgreSQL optimizations", request_id)
        print("\nSuccess! All subjects have been populated efficiently.")
        print(f"Academic area ID: {academic_area_id}")

        # Display final connection stats
        stats = db_config.get_connection_stats()
        print(f"\nFinal connection stats:")
        print(f"Active main connections: {stats['active_main_connections']}")
        print(f"Database type: {stats['database_type']}")

    except Exception as e:
        error_id(f"Error in main execution: {str(e)}", request_id)
        print(f"\nError: {str(e)}")
        traceback.print_exc()

    info_id("Academic subject population script completed", request_id)
    print("\nScript completed. Check the logs for details.")


# THIS IS THE CRITICAL PART
if __name__ == "__main__":
    print("Executing main function from __main__ block")
    try:
        main()
    except Exception as e:
        print(f"Unhandled exception in main: {e}")
        traceback.print_exc()
else:
    print(f"Note: Script was imported as a module, __name__ = {__name__}")

print("Script execution complete")

"""
CONCURRENT WRITES EXPLANATION:

This updated script addresses your question about concurrent writes with multiple approaches:

1. **Current Implementation (Sequential with Batched Operations)**:
   - Uses single transactions per field for efficiency
   - No concurrent writes, but much faster than individual operations
   - Safe for hierarchical data with dependencies
   - Recommended for most use cases

2. **Why No Concurrent Writes for This Use Case**:
   - Academic data has hierarchical dependencies (Area → EquipmentGroup → Model)
   - Population is a one-time setup operation, not ongoing writes
   - Data integrity is more important than write speed for setup scripts
   - Risk of foreign key constraint violations with concurrent inserts

3. **When You WOULD Use Concurrent Writes**:
   - Independent data insertion (no foreign key dependencies)
   - High-volume ongoing operations (not one-time setup)
   - When you have proper conflict resolution strategies
   - When you can handle transaction rollbacks gracefully

4. **Performance Optimizations Used Instead**:
   - Batched operations within single transactions
   - Bulk insert methods (session.add_all())
   - Connection pooling and limiting from the PostgreSQL framework
   - Proper session management with context managers
   - PostgreSQL-specific optimizations (execute_values, etc.)

5. **If You Need True Concurrent Writes Later**:
   - Use the PostgreSQLDatabaseManager.bulk_insert() method
   - Implement proper transaction isolation levels
   - Add retry logic for deadlock handling
   - Consider using message queues for async processing

The new framework provides the foundation for concurrent operations when needed,
but uses the most appropriate approach for this specific use case.
"""