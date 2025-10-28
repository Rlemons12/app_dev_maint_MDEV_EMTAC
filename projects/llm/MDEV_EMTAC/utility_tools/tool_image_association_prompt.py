# scripts/tool_image_association_prompt.py

import os
import sys
import logging
from sqlalchemy import create_engine, and_
from sqlalchemy.orm import sessionmaker

# Import models
from modules.emtacdb.emtacdb_fts import Base, Image, Tool, ToolImageAssociation
from modules.configuration.config import DATABASE_URL

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('tool_image_script')


def create_session():
    """Create and return a database session"""
    engine = create_engine(DATABASE_URL)
    Session = sessionmaker(bind=engine)
    return Session()


def list_tools(session):
    """List all tools in the database"""
    tools = session.query(Tool).order_by(Tool.name).all()
    if not tools:
        print("No tools found in the database.")
        return

    print("\nAvailable Tools:")
    print("-" * 60)
    print(f"{'ID':<5} | {'Name':<30} | {'Type':<15} | {'Size':<10}")
    print("-" * 60)

    for tool in tools:
        print(f"{tool.id:<5} | {tool.name[:30]:<30} | {tool.type or 'N/A':<15} | {tool.size or 'N/A':<10}")

    print("-" * 60)


def list_images(session):
    """List all images in the database"""
    images = session.query(Image).order_by(Image.title).all()
    if not images:
        print("No images found in the database.")
        return

    print("\nAvailable Images:")
    print("-" * 70)
    print(f"{'ID':<5} | {'Title':<30} | {'Description':<30}")
    print("-" * 70)

    for image in images:
        description = image.description[:30] + "..." if len(image.description) > 30 else image.description
        print(f"{image.id:<5} | {image.title[:30]:<30} | {description:<30}")

    print("-" * 70)


def list_tool_images(session, tool_id):
    """List all images associated with a specific tool"""
    # Use session.get instead of query.get
    tool = session.get(Tool, tool_id)
    if not tool:
        print(f"No tool found with ID {tool_id}")
        return

    print(f"\nImages for Tool: {tool.name}")
    print("-" * 70)

    associations = session.query(ToolImageAssociation).filter_by(tool_id=tool_id).all()
    if not associations:
        print(f"No images associated with this tool.")
        return

    print(f"{'Image ID':<10} | {'Title':<30} | {'Association Description':<30}")
    print("-" * 70)

    for assoc in associations:
        # Use session.get instead of query.get
        image = session.get(Image, assoc.image_id)
        if image:
            description = assoc.description or "No description"
            if len(description) > 30:
                description = description[:27] + "..."
            print(f"{image.id:<10} | {image.title[:30]:<30} | {description:<30}")

    print("-" * 70)


def associate_existing():
    """Associate an existing image with a tool"""
    session = create_session()

    try:
        # List available tools and images
        list_tools(session)
        list_images(session)

        # Get tool ID
        while True:
            try:
                tool_id = int(input("\nEnter Tool ID: "))
                # Use session.get instead of query.get
                tool = session.get(Tool, tool_id)
                if tool:
                    break
                print(f"Tool with ID {tool_id} not found. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

        # Get image ID
        while True:
            try:
                image_id = int(input("Enter Image ID: "))
                # Use session.get instead of query.get
                image = session.get(Image, image_id)
                if image:
                    break
                print(f"Image with ID {image_id} not found. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

        # Get description
        description = input("Enter association description (optional): ")
        if not description.strip():
            description = None

        # Create association
        association = ToolImageAssociation.associate_with_tool(
            session, image_id, tool_id, description
        )

        session.commit()
        print(f"\nSuccess! Associated image '{image.title}' with tool '{tool.name}'")

    except Exception as e:
        session.rollback()
        print(f"Error: {e}")
    finally:
        session.close()


def add_and_associate():
    """Add a new image and associate it with a tool"""
    session = create_session()

    try:
        # List available tools
        list_tools(session)

        # Get tool ID
        while True:
            try:
                tool_id = int(input("\nEnter Tool ID: "))
                # Use session.get instead of query.get
                tool = session.get(Tool, tool_id)
                if tool:
                    break
                print(f"Tool with ID {tool_id} not found. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

        # Get image details - improved file path handling
        while True:
            print("\nPlease enter the full path to the image file.")
            print("Example: C:\\Users\\Username\\Pictures\\tool_image.jpg")
            image_path = input("Image file path: ")

            # Clear any quotes that might be added accidentally
            if image_path.startswith('"') and image_path.endswith('"'):
                image_path = image_path[1:-1]
            elif image_path.startswith("'") and image_path.endswith("'"):
                image_path = image_path[1:-1]

            if os.path.isfile(image_path):
                print(f"File found: {image_path}")
                break
            else:
                print(f"File not found: {image_path}")
                retry = input("Would you like to try again? (y/n): ")
                if retry.lower() != 'y':
                    print("Image addition canceled.")
                    return

        image_title = input("Enter image title: ")
        image_description = input("Enter image description (optional): ")
        association_description = input("Enter association description (optional): ")

        if not association_description.strip():
            association_description = None

        # Confirm before proceeding
        print("\nReady to add image with the following details:")
        print(f"File: {image_path}")
        print(f"Title: {image_title}")
        print(f"Description: {image_description}")
        print(f"Tool ID: {tool_id} ({tool.name})")
        print(f"Association Description: {association_description}")

        confirm = input("\nProceed? (y/n): ")
        if confirm.lower() != 'y':
            print("Operation canceled.")
            return

        # Add image and associate with tool
        image, association = ToolImageAssociation.add_and_associate_with_tool(
            session,
            image_title,
            image_path,
            tool_id,
            description=image_description,
            association_description=association_description
        )

        session.commit()
        print(f"\nSuccess! Added image '{image_title}' and associated with tool '{tool.name}'")

    except Exception as e:
        session.rollback()
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()  # Print detailed error info
    finally:
        session.close()


def view_tool_images():
    """View images associated with a tool"""
    session = create_session()

    try:
        # List available tools
        list_tools(session)

        # Get tool ID
        while True:
            try:
                tool_id = int(input("\nEnter Tool ID: "))
                # Use session.get instead of query.get
                tool = session.get(Tool, tool_id)
                if tool:
                    break
                print(f"Tool with ID {tool_id} not found. Please try again.")
            except ValueError:
                print("Please enter a valid number.")

        # List images for this tool
        list_tool_images(session, tool_id)

    except Exception as e:
        print(f"Error: {e}")
    finally:
        session.close()


def main_menu():
    """Display the main menu and handle user selection"""
    while True:
        print("\n===== Tool-Image Association Tool =====")
        print("1. Associate Existing Image with Tool")
        print("2. Add New Image and Associate with Tool")
        print("3. View Tool Images")
        print("4. Exit")

        choice = input("\nEnter choice (1-4): ")

        if choice == '1':
            associate_existing()
        elif choice == '2':
            add_and_associate()
        elif choice == '3':
            view_tool_images()
        elif choice == '4':
            print("Exiting program. Goodbye!")
            sys.exit(0)
        else:
            print("Invalid choice. Please try again.")


if __name__ == '__main__':
    print("Welcome to the Tool-Image Association Tool")
    main_menu()