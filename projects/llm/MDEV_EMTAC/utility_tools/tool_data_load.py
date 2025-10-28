#!/usr/bin/env python3
"""
Script to load industrial maintenance tools data into the database.
Run this script to populate your database with realistic tool data.
"""

import sys
import os
from sqlalchemy.exc import IntegrityError
from sqlalchemy import text

# Add the project root to the path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your existing configuration and logging
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import logger, with_request_id, set_request_id

# Import your models - adjust these imports to match your actual model locations
# You'll need to update these import paths to match your project structure
try:
    from modules.emtacdb.emtacdb_fts import ToolCategory, ToolManufacturer, Tool, ToolPackage,\
                                                        tool_package_association

except ImportError as e:
    logger.error(f"Could not import tool models: {e}")
    logger.error("Please update the import paths in this script to match your project structure")
    sys.exit(1)


@with_request_id
def create_sample_data(db_config):
    """Create and insert all sample data for industrial maintenance tools."""

    # Set a request ID for this operation
    request_id = set_request_id("TOOL_LOAD")
    logger.info(f"Starting tool data loading with request ID: {request_id}")

    try:
        with db_config.main_session() as session:
            # Clear existing data (optional - remove if you want to keep existing data)
            logger.info("Clearing existing tool data...")
            session.execute(text("DELETE FROM tool_package_association"))
            session.execute(text("DELETE FROM tool_package"))
            session.execute(text("DELETE FROM tool"))
            session.execute(text("DELETE FROM tool_manufacturer"))
            session.execute(text("DELETE FROM tool_category"))
            session.commit()
            logger.info("Existing data cleared successfully")

            # 1. Create Tool Categories (Parent categories first)
            logger.info("Creating tool categories...")

            # Main categories
            hand_tools = ToolCategory(
                name="Hand Tools",
                description="Manual tools requiring no external power source"
            )
            power_tools = ToolCategory(
                name="Power Tools",
                description="Tools requiring electrical or pneumatic power"
            )
            measuring_testing = ToolCategory(
                name="Measuring & Testing Equipment",
                description="Precision instruments for measurement and diagnostics"
            )
            safety_equipment = ToolCategory(
                name="Safety Equipment",
                description="Personal protective equipment and safety tools"
            )
            specialty_tools = ToolCategory(
                name="Specialty Maintenance Tools",
                description="Specialized tools for specific maintenance tasks"
            )

            session.add_all([hand_tools, power_tools, measuring_testing, safety_equipment, specialty_tools])
            session.commit()
            logger.info("Main categories created successfully")

            # Subcategories
            # Hand Tools subcategories
            wrenches_spanners = ToolCategory(
                name="Wrenches & Spanners",
                description="Tools for gripping and turning nuts, bolts, and fittings",
                parent=hand_tools
            )
            screwdrivers = ToolCategory(
                name="Screwdrivers",
                description="Tools for turning screws",
                parent=hand_tools
            )
            pliers = ToolCategory(
                name="Pliers",
                description="Gripping, bending, and cutting tools",
                parent=hand_tools
            )
            hammers = ToolCategory(
                name="Hammers",
                description="Impact tools for striking",
                parent=hand_tools
            )
            files_rasps = ToolCategory(
                name="Files & Rasps",
                description="Abrasive tools for material removal",
                parent=hand_tools
            )

            # Power Tools subcategories
            electric_drills = ToolCategory(
                name="Electric Drills",
                description="Rotating tools for drilling holes",
                parent=power_tools
            )
            grinders = ToolCategory(
                name="Grinders",
                description="Abrasive cutting and surface preparation tools",
                parent=power_tools
            )
            saws = ToolCategory(
                name="Saws",
                description="Cutting tools for various materials",
                parent=power_tools
            )
            impact_tools = ToolCategory(
                name="Impact Tools",
                description="High-torque tools for fastener work",
                parent=power_tools
            )

            # Measuring & Testing subcategories
            linear_measurement = ToolCategory(
                name="Linear Measurement",
                description="Tools for measuring length, width, and depth",
                parent=measuring_testing
            )
            electrical_testing = ToolCategory(
                name="Electrical Testing",
                description="Instruments for electrical diagnostics",
                parent=measuring_testing
            )
            pressure_temperature = ToolCategory(
                name="Pressure & Temperature",
                description="Gauges and meters for system monitoring",
                parent=measuring_testing
            )

            # Safety Equipment subcategories
            personal_protection = ToolCategory(
                name="Personal Protection",
                description="Equipment to protect the individual worker",
                parent=safety_equipment
            )
            lockout_tagout = ToolCategory(
                name="Lockout/Tagout",
                description="Energy isolation safety equipment",
                parent=safety_equipment
            )

            # Specialty Tools subcategories
            bearing_pulley_tools = ToolCategory(
                name="Bearing & Pulley Tools",
                description="Tools for bearing and belt/pulley maintenance",
                parent=specialty_tools
            )
            pipe_fitting_tools = ToolCategory(
                name="Pipe & Fitting Tools",
                description="Tools for plumbing and piping work",
                parent=specialty_tools
            )
            lubrication_equipment = ToolCategory(
                name="Lubrication Equipment",
                description="Tools for applying lubricants and greases",
                parent=specialty_tools
            )

            session.add_all([
                wrenches_spanners, screwdrivers, pliers, hammers, files_rasps,
                electric_drills, grinders, saws, impact_tools,
                linear_measurement, electrical_testing, pressure_temperature,
                personal_protection, lockout_tagout,
                bearing_pulley_tools, pipe_fitting_tools, lubrication_equipment
            ])
            session.commit()
            logger.info("Subcategories created successfully")

            # 2. Create Tool Manufacturers
            logger.info("Creating tool manufacturers...")

            manufacturers_data = [
                {
                    'name': 'Snap-on Tools',
                    'description': 'Premium professional tools and equipment',
                    'country': 'USA',
                    'website': 'snapon.com'
                },
                {
                    'name': 'Matco Tools',
                    'description': 'Professional automotive and industrial tools',
                    'country': 'USA',
                    'website': 'matcotools.com'
                },
                {
                    'name': 'Mac Tools',
                    'description': 'Professional hand and power tools',
                    'country': 'USA',
                    'website': 'mactools.com'
                },
                {
                    'name': 'Proto Industrial Tools',
                    'description': 'Industrial-grade hand tools and torque equipment',
                    'country': 'USA',
                    'website': 'proto.stanley.com'
                },
                {
                    'name': 'Klein Tools',
                    'description': 'Electrical and utility tools',
                    'country': 'USA',
                    'website': 'kleintools.com'
                },
                {
                    'name': 'Fluke Corporation',
                    'description': 'Electronic test tools and measurement equipment',
                    'country': 'USA',
                    'website': 'fluke.com'
                },
                {
                    'name': 'Milwaukee Tool',
                    'description': 'Heavy-duty power tools and accessories',
                    'country': 'USA',
                    'website': 'milwaukeetool.com'
                },
                {
                    'name': 'DeWalt',
                    'description': 'Professional power tools and accessories',
                    'country': 'USA',
                    'website': 'dewalt.com'
                },
                {
                    'name': 'Starrett',
                    'description': 'Precision measurement tools',
                    'country': 'USA',
                    'website': 'starrett.com'
                },
                {
                    'name': 'SKF',
                    'description': 'Bearing maintenance tools and equipment',
                    'country': 'Sweden',
                    'website': 'skf.com'
                },
                {
                    'name': 'Ridgid',
                    'description': 'Professional plumbing and pipe tools',
                    'country': 'USA',
                    'website': 'ridgid.com'
                },
                {
                    'name': '3M',
                    'description': 'Safety equipment and protective gear',
                    'country': 'USA',
                    'website': '3m.com'
                },
                {
                    'name': 'Mechanix Wear',
                    'description': 'Professional work gloves and protective equipment',
                    'country': 'USA',
                    'website': 'mechanix.com'
                },
                {
                    'name': 'Lincoln Industrial',
                    'description': 'Lubrication equipment and systems',
                    'country': 'USA',
                    'website': 'lincolnindustrial.com'
                }
            ]

            manufacturers = []
            for mfg_data in manufacturers_data:
                manufacturer = ToolManufacturer(**mfg_data)
                manufacturers.append(manufacturer)

            session.add_all(manufacturers)
            session.commit()
            logger.info(f"Created {len(manufacturers)} manufacturers successfully")

            # Create manufacturer lookup dict
            mfg_lookup = {mfg.name: mfg for mfg in manufacturers}

            # 3. Create Tools
            logger.info("Creating tools...")

            tools_data = [
                {
                    'name': 'Combination Wrench Set',
                    'size': '8mm-32mm (Metric)',
                    'type': 'Combination (Box/Open End)',
                    'material': 'Chrome Vanadium Steel',
                    'description': 'Professional-grade combination wrenches for general fastener work',
                    'category': wrenches_spanners,
                    'manufacturer': 'Proto Industrial Tools'
                },
                {
                    'name': 'Adjustable Wrench Set',
                    'size': '6", 8", 10", 12"',
                    'type': 'Adjustable',
                    'material': 'Chrome Vanadium Steel',
                    'description': 'Variable jaw wrenches for various fastener sizes',
                    'category': wrenches_spanners,
                    'manufacturer': 'Proto Industrial Tools'
                },
                {
                    'name': 'Pipe Wrench Set',
                    'size': '10", 14", 18", 24"',
                    'type': 'Pipe Wrench',
                    'material': 'Cast Iron with Steel Jaws',
                    'description': 'Heavy-duty wrenches for pipe and round stock gripping',
                    'category': wrenches_spanners,
                    'manufacturer': 'Ridgid'
                },
                {
                    'name': 'Phillips Screwdriver Set',
                    'size': '#0, #1, #2, #3, #4',
                    'type': 'Phillips Head',
                    'material': 'Chrome Vanadium Steel Blade, Acetate Handle',
                    'description': 'Professional electrician\'s screwdrivers with cushion-grip handles',
                    'category': screwdrivers,
                    'manufacturer': 'Klein Tools'
                },
                {
                    'name': 'Flathead Screwdriver Set',
                    'size': '1/8", 3/16", 1/4", 5/16", 3/8"',
                    'type': 'Slotted/Flathead',
                    'material': 'Chrome Vanadium Steel',
                    'description': 'Precision flathead screwdrivers for various applications',
                    'category': screwdrivers,
                    'manufacturer': 'Klein Tools'
                },
                {
                    'name': 'Needle Nose Pliers',
                    'size': '6", 8"',
                    'type': 'Long Nose',
                    'material': 'Chrome Nickel Steel',
                    'description': 'Precision gripping and manipulation in tight spaces',
                    'category': pliers,
                    'manufacturer': 'Klein Tools'
                },
                {
                    'name': 'Diagonal Cutting Pliers',
                    'size': '8"',
                    'type': 'Side Cutters',
                    'material': 'High Carbon Steel',
                    'description': 'Clean cutting of wire and small materials',
                    'category': pliers,
                    'manufacturer': 'Klein Tools'
                },
                {
                    'name': 'Cordless Drill/Driver',
                    'size': '1/2" Chuck',
                    'type': 'Brushless Motor',
                    'material': 'Metal Chuck, Polymer Housing',
                    'description': 'High-torque cordless drill for drilling and fastening',
                    'category': electric_drills,
                    'manufacturer': 'Milwaukee Tool'
                },
                {
                    'name': 'Angle Grinder',
                    'size': '4-1/2"',
                    'type': 'Electric Angle',
                    'material': 'Metal Gear Housing',
                    'description': 'Cutting and grinding for metal fabrication and repair',
                    'category': grinders,
                    'manufacturer': 'Milwaukee Tool'
                },
                {
                    'name': 'Steel Rule Set',
                    'size': '6", 12", 24"',
                    'type': 'Flexible Steel Rule',
                    'material': 'Spring Steel',
                    'description': 'Precision measurement rulers with multiple graduation marks',
                    'category': linear_measurement,
                    'manufacturer': 'Starrett'
                },
                {
                    'name': 'Outside Micrometer Set',
                    'size': '0-1", 1-2", 2-3"',
                    'type': 'Outside Micrometer',
                    'material': 'Hardened Steel',
                    'description': 'Precision measurement of external dimensions',
                    'category': linear_measurement,
                    'manufacturer': 'Starrett'
                },
                {
                    'name': 'Digital Multimeter',
                    'size': 'Handheld',
                    'type': 'True RMS Digital',
                    'material': 'Impact-resistant case',
                    'description': 'Electrical measurement and diagnostics instrument',
                    'category': electrical_testing,
                    'manufacturer': 'Fluke Corporation'
                },
                {
                    'name': 'Clamp Meter',
                    'size': '1.18" jaw opening',
                    'type': 'AC/DC Current Clamp',
                    'material': 'Reinforced plastic housing',
                    'description': 'Non-contact current measurement tool',
                    'category': electrical_testing,
                    'manufacturer': 'Fluke Corporation'
                },
                {
                    'name': 'Safety Glasses',
                    'size': 'Universal',
                    'type': 'Wraparound Safety',
                    'material': 'Polycarbonate Lens, Nylon Frame',
                    'description': 'Impact-resistant eye protection with side shields',
                    'category': personal_protection,
                    'manufacturer': '3M'
                },
                {
                    'name': 'Work Gloves',
                    'size': 'Medium, Large, X-Large',
                    'type': 'Mechanic\'s Gloves',
                    'material': 'Synthetic Leather Palm, Spandex Back',
                    'description': 'Dexterous protection for general maintenance work',
                    'category': personal_protection,
                    'manufacturer': 'Mechanix Wear'
                },
                {
                    'name': 'Bearing Puller Set',
                    'size': '2-jaw, 3-jaw configurations',
                    'type': 'Mechanical Puller',
                    'material': 'Drop-forged Steel',
                    'description': 'Safe removal of bearings and pressed-fit components',
                    'category': bearing_pulley_tools,
                    'manufacturer': 'SKF'
                },
                {
                    'name': 'Bearing Installation Kit',
                    'size': 'Various drivers 10mm-100mm',
                    'type': 'Driver Set',
                    'material': 'Heat-treated Steel',
                    'description': 'Proper installation of bearings without damage',
                    'category': bearing_pulley_tools,
                    'manufacturer': 'SKF'
                },
                {
                    'name': 'Grease Gun',
                    'size': '14.5 oz cartridge capacity',
                    'type': 'Lever-action',
                    'material': 'Cast Iron Head, Steel Barrel',
                    'description': 'High-pressure grease application for maintenance lubrication',
                    'category': lubrication_equipment,
                    'manufacturer': 'Lincoln Industrial'
                }
            ]

            tools = []
            for tool_data in tools_data:
                manufacturer_name = tool_data.pop('manufacturer')
                category = tool_data.pop('category')

                tool = Tool(
                    **tool_data,
                    tool_category=category,
                    tool_manufacturer=mfg_lookup[manufacturer_name]
                )
                tools.append(tool)

            session.add_all(tools)
            session.commit()
            logger.info(f"Created {len(tools)} tools successfully")

            # Create tool lookup dict
            tool_lookup = {tool.name: tool for tool in tools}

            # 4. Create Tool Packages
            logger.info("Creating tool packages...")

            # Package 1: Basic Maintenance Kit
            basic_kit = ToolPackage(
                name="Basic Maintenance Kit",
                description="Essential tools for routine maintenance tasks"
            )

            # Package 2: Electrical Maintenance Kit
            electrical_kit = ToolPackage(
                name="Electrical Maintenance Kit",
                description="Specialized tools for electrical system maintenance"
            )

            # Package 3: Mechanical Maintenance Kit
            mechanical_kit = ToolPackage(
                name="Mechanical Maintenance Kit",
                description="Tools for mechanical system maintenance and repair"
            )

            # Package 4: Power Tools Starter Kit
            power_kit = ToolPackage(
                name="Power Tools Starter Kit",
                description="Essential power tools for industrial maintenance"
            )

            # Package 5: Precision Measurement Kit
            measurement_kit = ToolPackage(
                name="Precision Measurement Kit",
                description="Accurate measurement tools for quality maintenance work"
            )

            packages = [basic_kit, electrical_kit, mechanical_kit, power_kit, measurement_kit]
            session.add_all(packages)
            session.commit()
            logger.info(f"Created {len(packages)} tool packages successfully")

            # 5. Create Tool-Package Associations
            logger.info("Creating tool-package associations...")

            # Basic Maintenance Kit associations
            basic_tools = [
                ('Combination Wrench Set', 1),
                ('Phillips Screwdriver Set', 1),
                ('Flathead Screwdriver Set', 1),
                ('Adjustable Wrench Set', 1),
                ('Needle Nose Pliers', 1),
                ('Safety Glasses', 1),
                ('Work Gloves', 1)
            ]

            for tool_name, qty in basic_tools:
                association = tool_package_association.insert().values(
                    tool_id=tool_lookup[tool_name].id,
                    package_id=basic_kit.id,
                    quantity=qty
                )
                session.execute(association)

            # Electrical Maintenance Kit associations
            electrical_tools = [
                ('Digital Multimeter', 1),
                ('Clamp Meter', 1),
                ('Phillips Screwdriver Set', 1),
                ('Diagonal Cutting Pliers', 1),
                ('Needle Nose Pliers', 1),
                ('Safety Glasses', 1)
            ]

            for tool_name, qty in electrical_tools:
                association = tool_package_association.insert().values(
                    tool_id=tool_lookup[tool_name].id,
                    package_id=electrical_kit.id,
                    quantity=qty
                )
                session.execute(association)

            # Mechanical Maintenance Kit associations
            mechanical_tools = [
                ('Bearing Puller Set', 1),
                ('Bearing Installation Kit', 1),
                ('Grease Gun', 1),
                ('Combination Wrench Set', 1),
                ('Steel Rule Set', 1),
                ('Outside Micrometer Set', 1)
            ]

            for tool_name, qty in mechanical_tools:
                association = tool_package_association.insert().values(
                    tool_id=tool_lookup[tool_name].id,
                    package_id=mechanical_kit.id,
                    quantity=qty
                )
                session.execute(association)

            # Power Tools Starter Kit associations
            power_tools_list = [
                ('Cordless Drill/Driver', 1),
                ('Angle Grinder', 1),
                ('Safety Glasses', 1),
                ('Work Gloves', 1)
            ]

            for tool_name, qty in power_tools_list:
                association = tool_package_association.insert().values(
                    tool_id=tool_lookup[tool_name].id,
                    package_id=power_kit.id,
                    quantity=qty
                )
                session.execute(association)

            # Precision Measurement Kit associations
            measurement_tools = [
                ('Steel Rule Set', 1),
                ('Outside Micrometer Set', 1),
                ('Digital Multimeter', 1)
            ]

            for tool_name, qty in measurement_tools:
                association = tool_package_association.insert().values(
                    tool_id=tool_lookup[tool_name].id,
                    package_id=measurement_kit.id,
                    quantity=qty
                )
                session.execute(association)

            session.commit()
            logger.info("Tool-package associations created successfully")

            # Print summary
            category_count = session.query(ToolCategory).count()
            manufacturer_count = session.query(ToolManufacturer).count()
            tool_count = session.query(Tool).count()
            package_count = session.query(ToolPackage).count()

            logger.info("Successfully loaded all tool data!")
            logger.info(f"Summary:")
            logger.info(f"- Created {category_count} tool categories")
            logger.info(f"- Created {manufacturer_count} manufacturers")
            logger.info(f"- Created {tool_count} tools")
            logger.info(f"- Created {package_count} tool packages")

    except IntegrityError as e:
        logger.error(f"Database integrity error: {e}")
        raise
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def main():
    """Main function to set up database connection and load data."""

    try:
        logger.info("Initializing database configuration...")
        db_config = DatabaseConfig()

        # Get connection stats for debugging
        stats = db_config.get_connection_stats()
        logger.info(f"Database connection stats: {stats}")

        # Create tables if they don't exist
        logger.info("Creating database tables if needed...")
        db_config.get_main_base().metadata.create_all(db_config.main_engine)

        logger.info("Starting to load industrial maintenance tools data...")
        create_sample_data(db_config)
        logger.info("Tool data loading completed successfully!")

    except ImportError as e:
        logger.error(f"Import error - please check your model imports: {e}")
        return 1
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())