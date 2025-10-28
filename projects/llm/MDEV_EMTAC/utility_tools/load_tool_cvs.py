#!/usr/bin/env python3
"""
Script to load hand tools from CSV file into the database.
Handles category creation, manufacturer assignment, and tool insertion.
"""

import sys
import os
import csv
import re
from typing import Dict, Any, Optional
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from sqlalchemy import text

# Add the project root to the path if needed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your existing configuration and logging
from modules.configuration.config_env import DatabaseConfig
from modules.configuration.log_config import logger, with_request_id, set_request_id

# Import your models - adjust these imports to match your actual model locations
try:
    from modules.emtacdb.emtacdb_fts import (
        Tool, ToolCategory, ToolManufacturer
    )
except ImportError as e:
    logger.error(f"Could not import tool models: {e}")
    logger.error("Please update the import paths to match your project structure")
    sys.exit(1)


class HandToolsLoader:
    """
    Loads hand tools from CSV file into the database with intelligent
    category and manufacturer mapping.
    """

    def __init__(self, db_config: DatabaseConfig):
        """Initialize the loader with database configuration."""
        self.db_config = db_config
        self.request_id = set_request_id("CSV_LOAD")

        # Cache for database lookups
        self.category_cache = {}
        self.manufacturer_cache = {}

        # Tool type to manufacturer mapping
        self.manufacturer_mapping = {
            # Precision/Electronics tools
            'precision screwdriver': 'Klein Tools',
            'torx screwdriver': 'Klein Tools',
            'hex key': 'Bondhus',
            'needle nose pliers': 'Klein Tools',
            'diagonal pliers': 'Klein Tools',
            'wire strippers': 'Klein Tools',

            # General hand tools
            'combination wrench': 'Proto Industrial Tools',
            'open end wrench': 'Proto Industrial Tools',
            'adjustable wrench': 'Crescent Tool',
            'pipe wrench': 'Ridgid',

            # Hammers
            'ball peen hammer': 'Estwing',
            'claw hammer': 'Estwing',
            'dead blow hammer': 'Snap-On',
            'sledge hammer': 'Truper',

            # Cutting tools
            'hacksaw': 'Stanley',
            'snips': 'Midwest Tool',
            'aviation snips': 'Midwest Tool',
            'utility knife': 'Stanley',
            'box cutter': 'Stanley',

            # Files and shaping
            'mill file': 'Nicholson',
            'flat file': 'Nicholson',
            'round file': 'Nicholson',
            'needle file': 'Grobet',

            # Measuring tools
            'steel rule': 'Starrett',
            'tape measure': 'Stanley',
            'micrometer': 'Starrett',
            'calipers': 'Starrett',
            'combination square': 'Starrett',
            'try square': 'Starrett',

            # Specialty tools
            'pry bar': 'Mayhew Tools',
            'crowbar': 'Stanley',
            'grease gun': 'Lincoln Industrial',
            'oil can': 'Goldenrod',
        }

        # Category hierarchy mapping
        self.category_hierarchy = {
            'Screwdrivers': 'Hand Tools',
            'Wrenches': 'Hand Tools',
            'Pliers': 'Hand Tools',
            'Hammers': 'Hand Tools',
            'Files': 'Hand Tools',
            'Chisels': 'Hand Tools',
            'Punches': 'Hand Tools',
            'Cutting': 'Hand Tools',
            'Marking': 'Hand Tools',
            'Measuring': 'Measuring & Testing Equipment',
            'Prying': 'Hand Tools',
            'Specialty': 'Specialty Maintenance Tools',
            'Lubrication': 'Specialty Maintenance Tools'
        }

        logger.info(f"HandToolsLoader initialized with request ID: {self.request_id}")

    @with_request_id
    def load_csv_file(self, csv_file_path: str, clear_existing: bool = False) -> Dict[str, Any]:
        """
        Load tools from the CSV file into the database.

        Args:
            csv_file_path: Path to the CSV file
            clear_existing: Whether to clear existing tool data first

        Returns:
            Dictionary with loading statistics
        """
        if not os.path.exists(csv_file_path):
            raise FileNotFoundError(f"CSV file not found: {csv_file_path}")

        stats = {
            'tools_loaded': 0,
            'categories_created': 0,
            'manufacturers_created': 0,
            'errors': 0,
            'skipped': 0
        }

        try:
            with self.db_config.main_session() as session:
                # Clear existing data if requested
                if clear_existing:
                    logger.info("Clearing existing tool data...")
                    session.execute(text("DELETE FROM tool WHERE id > 0"))  # Keep existing structure
                    session.commit()
                    logger.info("Existing tool data cleared")

                # Load existing categories and manufacturers
                self._load_caches(session)

                # Read and process CSV file
                logger.info(f"Reading CSV file: {csv_file_path}")
                with open(csv_file_path, 'r', newline='', encoding='utf-8') as csvfile:
                    reader = csv.DictReader(csvfile)

                    # Validate CSV headers
                    expected_headers = ['Tool_Name', 'Size', 'Type', 'Category', 'Material', 'Description',
                                        'Typical_Use']
                    if not all(header in reader.fieldnames for header in expected_headers):
                        raise ValueError(f"CSV missing required headers. Expected: {expected_headers}")

                    # Process each row
                    row_count = 0
                    for row in reader:
                        row_count += 1
                        try:
                            if self._process_tool_row(session, row):
                                stats['tools_loaded'] += 1
                            else:
                                stats['skipped'] += 1

                            # Commit in batches for better performance
                            if row_count % 50 == 0:
                                session.commit()
                                logger.info(f"Processed {row_count} rows...")

                        except Exception as e:
                            logger.error(f"Error processing row {row_count}: {e}")
                            stats['errors'] += 1
                            continue

                    # Final commit
                    session.commit()

                # Update stats
                stats['categories_created'] = len(self.category_cache)
                stats['manufacturers_created'] = len(self.manufacturer_cache)

                logger.info(f"CSV loading completed successfully!")
                logger.info(f"Statistics: {stats}")

                return stats

        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise

    def _load_caches(self, session):
        """Load existing categories and manufacturers into cache."""
        logger.info("Loading existing categories and manufacturers...")

        # Load categories
        categories = session.query(ToolCategory).all()
        for category in categories:
            self.category_cache[category.name] = category

        # Load manufacturers
        manufacturers = session.query(ToolManufacturer).all()
        for manufacturer in manufacturers:
            self.manufacturer_cache[manufacturer.name] = manufacturer

        logger.info(f"Loaded {len(self.category_cache)} categories and {len(self.manufacturer_cache)} manufacturers")

    def _process_tool_row(self, session, row: Dict[str, str]) -> bool:
        """
        Process a single tool row from the CSV.

        Args:
            session: Database session
            row: CSV row data

        Returns:
            True if tool was created, False if skipped
        """
        # Extract and clean data
        tool_name = row['Tool_Name'].strip()
        size = row['Size'].strip() if row['Size'].strip() else None
        tool_type = row['Type'].strip() if row['Type'].strip() else None
        category_name = row['Category'].strip()
        material = row['Material'].strip() if row['Material'].strip() else None
        description = row['Description'].strip() if row['Description'].strip() else None
        typical_use = row['Typical_Use'].strip() if row['Typical_Use'].strip() else None

        # Combine description and typical use
        full_description = description
        if typical_use:
            if full_description:
                full_description += f" - Typical use: {typical_use}"
            else:
                full_description = f"Typical use: {typical_use}"

        # Skip if essential data is missing
        if not tool_name or not category_name:
            logger.warning(f"Skipping row due to missing essential data: {tool_name}")
            return False

        # Get or create category
        category = self._get_or_create_category(session, category_name)
        if not category:
            logger.error(f"Could not create category: {category_name}")
            return False

        # Determine manufacturer
        manufacturer = self._determine_manufacturer(session, tool_name, tool_type, category_name)
        if not manufacturer:
            logger.error(f"Could not determine manufacturer for: {tool_name}")
            return False

        # Check if tool already exists (avoid duplicates)
        existing_tool = session.query(Tool).filter(
            Tool.name == tool_name,
            Tool.size == size,
            Tool.tool_category_id == category.id
        ).first()

        if existing_tool:
            logger.debug(f"Tool already exists, skipping: {tool_name} ({size})")
            return False

        # Create the tool
        new_tool = Tool(
            name=tool_name,
            size=size,
            type=tool_type,
            material=material,
            description=full_description,
            tool_category_id=category.id,
            tool_manufacturer_id=manufacturer.id
        )

        session.add(new_tool)
        logger.debug(f"Created tool: {tool_name} ({size}) - {category.name}")

        return True

    def _get_or_create_category(self, session, category_name: str) -> Optional[ToolCategory]:
        """Get existing category or create new one with proper hierarchy."""
        # Check cache first
        if category_name in self.category_cache:
            return self.category_cache[category_name]

        # Check if category exists in database
        category = session.query(ToolCategory).filter(
            ToolCategory.name == category_name
        ).first()

        if category:
            self.category_cache[category_name] = category
            return category

        # Create new category with parent if specified
        parent_category = None
        if category_name in self.category_hierarchy:
            parent_name = self.category_hierarchy[category_name]
            parent_category = self._get_or_create_category(session, parent_name)

        # Create the category
        new_category = ToolCategory(
            name=category_name,
            description=f"Tools in the {category_name.lower()} category",
            parent=parent_category
        )

        session.add(new_category)
        session.flush()  # Get ID without committing

        self.category_cache[category_name] = new_category
        logger.info(f"Created category: {category_name}")

        return new_category

    def _determine_manufacturer(self, session, tool_name: str, tool_type: str, category: str) -> Optional[
        ToolManufacturer]:
        """Determine the appropriate manufacturer for a tool."""
        manufacturer_name = None

        # Try to match by tool type
        tool_key = tool_type.lower() if tool_type else ''
        if tool_key in self.manufacturer_mapping:
            manufacturer_name = self.manufacturer_mapping[tool_key]
        else:
            # Try to match by tool name
            tool_name_lower = tool_name.lower()
            for key, mfg in self.manufacturer_mapping.items():
                if key in tool_name_lower:
                    manufacturer_name = mfg
                    break

        # Default manufacturer if no match found
        if not manufacturer_name:
            manufacturer_name = "Generic Hand Tools"

        # Get or create manufacturer
        return self._get_or_create_manufacturer(session, manufacturer_name)

    def _get_or_create_manufacturer(self, session, manufacturer_name: str) -> Optional[ToolManufacturer]:
        """Get existing manufacturer or create new one."""
        # Check cache first
        if manufacturer_name in self.manufacturer_cache:
            return self.manufacturer_cache[manufacturer_name]

        # Check if manufacturer exists in database
        manufacturer = session.query(ToolManufacturer).filter(
            ToolManufacturer.name == manufacturer_name
        ).first()

        if manufacturer:
            self.manufacturer_cache[manufacturer_name] = manufacturer
            return manufacturer

        # Create new manufacturer
        new_manufacturer = ToolManufacturer(
            name=manufacturer_name,
            description=f"Hand tool manufacturer - {manufacturer_name}",
            country="USA",  # Default country
            website=self._generate_website(manufacturer_name)
        )

        session.add(new_manufacturer)
        session.flush()  # Get ID without committing

        self.manufacturer_cache[manufacturer_name] = new_manufacturer
        logger.info(f"Created manufacturer: {manufacturer_name}")

        return new_manufacturer

    def _generate_website(self, manufacturer_name: str) -> str:
        """Generate a plausible website URL for a manufacturer."""
        # Clean up the name for URL
        clean_name = re.sub(r'[^a-zA-Z0-9\s]', '', manufacturer_name)
        clean_name = clean_name.replace(' ', '').lower()
        return f"{clean_name}.com"

    @with_request_id
    def get_load_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded tools."""
        try:
            with self.db_config.main_session() as session:
                stats = {}

                # Total tool count
                stats['total_tools'] = session.query(Tool).count()

                # Tools by category
                category_stats = session.query(
                    ToolCategory.name,
                    session.query(Tool).filter(Tool.tool_category_id == ToolCategory.id).count().label('count')
                ).join(Tool, isouter=True).group_by(ToolCategory.name).all()

                stats['tools_by_category'] = {name: count for name, count in category_stats if count > 0}

                # Tools by manufacturer
                manufacturer_stats = session.query(
                    ToolManufacturer.name,
                    session.query(Tool).filter(Tool.tool_manufacturer_id == ToolManufacturer.id).count().label('count')
                ).join(Tool, isouter=True).group_by(ToolManufacturer.name).all()

                stats['tools_by_manufacturer'] = {name: count for name, count in manufacturer_stats if count > 0}

                return stats

        except SQLAlchemyError as e:
            logger.error(f"Database error getting statistics: {e}")
            raise


def main():
    """Main function to load the CSV file."""
    import argparse

    parser = argparse.ArgumentParser(description='Load hand tools from CSV file')
    parser.add_argument('csv_file', help='Path to the CSV file')
    parser.add_argument('--clear', action='store_true', help='Clear existing tools before loading')
    parser.add_argument('--stats', action='store_true', help='Show statistics after loading')

    args = parser.parse_args()

    try:
        # Initialize database configuration
        logger.info("Initializing database configuration...")
        db_config = DatabaseConfig()

        # Create tables if they don't exist
        logger.info("Ensuring database tables exist...")
        db_config.get_main_base().metadata.create_all(db_config.main_engine)

        # Initialize loader
        loader = HandToolsLoader(db_config)

        # Load the CSV file
        logger.info(f"Loading tools from CSV file: {args.csv_file}")
        load_stats = loader.load_csv_file(args.csv_file, clear_existing=args.clear)

        # Display results
        print("\n" + "=" * 50)
        print("LOADING COMPLETED SUCCESSFULLY!")
        print("=" * 50)
        print(f"Tools loaded: {load_stats['tools_loaded']}")
        print(f"Categories created: {load_stats['categories_created']}")
        print(f"Manufacturers created: {load_stats['manufacturers_created']}")
        print(f"Errors: {load_stats['errors']}")
        print(f"Skipped: {load_stats['skipped']}")

        # Show detailed statistics if requested
        if args.stats:
            print("\n" + "=" * 50)
            print("DETAILED STATISTICS")
            print("=" * 50)

            detailed_stats = loader.get_load_statistics()

            print(f"\nTotal tools in database: {detailed_stats['total_tools']}")

            print(f"\nTools by category:")
            for category, count in detailed_stats['tools_by_category'].items():
                print(f"  {category}: {count}")

            print(f"\nTools by manufacturer:")
            for manufacturer, count in detailed_stats['tools_by_manufacturer'].items():
                print(f"  {manufacturer}: {count}")

        logger.info("CSV loading process completed successfully!")
        return 0

    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        print(f"Error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        print(f"Error loading CSV: {e}")
        return 1


# Example usage function
def example_usage():
    """Example of how to use the HandToolsLoader programmatically."""

    # Initialize database configuration
    db_config = DatabaseConfig()

    # Create the loader
    loader = HandToolsLoader(db_config)

    # Load from CSV file
    csv_file_path = "hand_tools_comprehensive.csv"  # Adjust path as needed

    try:
        # Load the tools
        stats = loader.load_csv_file(csv_file_path, clear_existing=False)

        print(f"Loaded {stats['tools_loaded']} tools successfully!")

        # Get detailed statistics
        detailed_stats = loader.get_load_statistics()
        print(f"Total tools now in database: {detailed_stats['total_tools']}")

    except Exception as e:
        logger.error(f"Error in example usage: {e}")


if __name__ == "__main__":
    exit(main())