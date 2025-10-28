import json
import os
from pathlib import Path
import threading

from kivy.core.window import Window
from kivy.clock import Clock
from kivy.logger import Logger


class FileManager:
    """Manages file operations for the application"""

    def __init__(self, app_instance):
        self.app = app_instance
        self.base_dir = Path.home() / "maintenance_app"
        self.data_dir = self.base_dir / "data"
        self.images_dir = self.base_dir / "images"
        self.layouts_dir = self.base_dir / "layouts"

        # Ensure directories exist
        self._ensure_directories()

    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        for directory in [self.base_dir, self.data_dir, self.images_dir, self.layouts_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def save_layout(self, layout_name, layout_data):
        """Save a layout configuration to a JSON file"""
        try:
            layout_file = self.layouts_dir / f"{layout_name}.json"
            with open(layout_file, 'w') as f:
                json.dump(layout_data, f, indent=2)
            return True, f"Layout saved to {layout_file}"
        except Exception as e:
            Logger.error(f"FileManager: Error saving layout - {str(e)}")
            return False, f"Error saving layout: {str(e)}"

    def load_layout(self, layout_name):
        """Load a layout configuration from a JSON file"""
        try:
            layout_file = self.layouts_dir / f"{layout_name}.json"
            with open(layout_file, 'r') as f:
                return True, json.load(f)
        except FileNotFoundError:
            return False, f"Layout file {layout_name}.json not found"
        except json.JSONDecodeError:
            return False, f"Invalid JSON in layout file {layout_name}.json"
        except Exception as e:
            Logger.error(f"FileManager: Error loading layout - {str(e)}")
            return False, f"Error loading layout: {str(e)}"

    def get_available_layouts(self):
        """Get a list of available saved layouts"""
        try:
            layouts = [f.stem for f in self.layouts_dir.glob("*.json")]
            return layouts
        except Exception as e:
            Logger.error(f"FileManager: Error listing layouts - {str(e)}")
            return []

    def save_data(self, data_type, data):
        """Save data to a JSON file"""
        try:
            data_file = self.data_dir / f"{data_type}.json"
            with open(data_file, 'w') as f:
                json.dump(data, f, indent=2)
            return True, f"Data saved to {data_file}"
        except Exception as e:
            Logger.error(f"FileManager: Error saving data - {str(e)}")
            return False, f"Error saving data: {str(e)}"

    def load_data(self, data_type):
        """Load data from a JSON file"""
        try:
            data_file = self.data_dir / f"{data_type}.json"
            with open(data_file, 'r') as f:
                return True, json.load(f)
        except FileNotFoundError:
            return False, f"Data file {data_type}.json not found"
        except json.JSONDecodeError:
            return False, f"Invalid JSON in data file {data_type}.json"
        except Exception as e:
            Logger.error(f"FileManager: Error loading data - {str(e)}")
            return False, f"Error loading data: {str(e)}"

    def save_image(self, image_data, filename=None):
        """Save an image to the images directory"""
        try:
            if filename is None:
                # Generate a unique filename based on timestamp
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"image_{timestamp}.png"

            image_file = self.images_dir / filename
            with open(image_file, 'wb') as f:
                f.write(image_data)
            return True, str(image_file)
        except Exception as e:
            Logger.error(f"FileManager: Error saving image - {str(e)}")
            return False, f"Error saving image: {str(e)}"

    def get_image_list(self):
        """Get a list of available images with paths"""
        try:
            images = []
            for ext in ['png', 'jpg', 'jpeg', 'gif']:
                images.extend(list(self.images_dir.glob(f"*.{ext}")))

            return [str(img) for img in images]
        except Exception as e:
            Logger.error(f"FileManager: Error listing images - {str(e)}")
            return []


class AsyncTask:
    """Utility for running tasks asynchronously"""

    @staticmethod
    def run(func, callback=None, *args, **kwargs):
        """Run a function in a separate thread and call the callback on completion"""

        def _async_call(func, callback, *args, **kwargs):
            result = func(*args, **kwargs)
            if callback:
                Clock.schedule_once(lambda dt: callback(result), 0)

        thread = threading.Thread(target=_async_call, args=(func, callback) + args, kwargs=kwargs)
        thread.daemon = True
        thread.start()
        return thread


class LayoutManager:
    """Manages UI layouts for the application"""

    def __init__(self, app_instance):
        self.app = app_instance
        self.file_manager = FileManager(app_instance)
        self.layouts = {
            'default': {
                'problem_solution_panel': {'pos': (10, 10), 'size': (Window.width * 0.48 - 15, Window.height - 20)},
                'documents_panel': {'pos': (Window.width * 0.5 + 5, 10),
                                    'size': (Window.width * 0.25 - 15, Window.height * 0.5 - 15)},
                'parts_panel': {'pos': (Window.width * 0.75 + 5, 10),
                                'size': (Window.width * 0.25 - 15, Window.height * 0.5 - 15)},
                'images_panel': {'pos': (Window.width * 0.5 + 5, Window.height * 0.5 + 5),
                                 'size': (Window.width * 0.25 - 15, Window.height * 0.5 - 15)},
                'drawings_panel': {'pos': (Window.width * 0.75 + 5, Window.height * 0.5 + 5),
                                   'size': (Window.width * 0.25 - 15, Window.height * 0.5 - 15)},
            },
            'top_left_parts': {
                'parts_panel': {'pos': (10, 10), 'size': (Window.width * 0.25 - 15, Window.height * 0.5 - 15)},
                'documents_panel': {'pos': (10, Window.height * 0.5 + 5),
                                    'size': (Window.width * 0.25 - 15, Window.height * 0.5 - 15)},
                'problem_solution_panel': {'pos': (Window.width * 0.25 + 5, 10),
                                           'size': (Window.width * 0.5 - 15, Window.height - 20)},
                'images_panel': {'pos': (Window.width * 0.75 + 5, 10),
                                 'size': (Window.width * 0.25 - 15, Window.height * 0.5 - 15)},
                'drawings_panel': {'pos': (Window.width * 0.75 + 5, Window.height * 0.5 + 5),
                                   'size': (Window.width * 0.25 - 15, Window.height * 0.5 - 15)},
            }
        }

    def apply_layout(self, layout_name):
        """Apply a layout by name"""
        if layout_name in self.layouts:
            self._set_layout(self.layouts[layout_name])
            return True
        else:
            # Try to load a custom layout
            success, result = self.file_manager.load_layout(layout_name)
            if success:
                self._set_layout(result)
                return True
            return False

    def _set_layout(self, layout_data):
        """Apply layout data to the UI panels"""
        for panel_id, panel_info in layout_data.items():
            panel = self.app.root.ids.get(panel_id)
            if panel:
                panel.pos = panel_info['pos']
                panel.size = panel_info['size']
                # Make sure expanded panels are collapsed
                if hasattr(panel, 'is_expanded') and panel.is_expanded:
                    panel.toggle_expand()

    def save_current_layout(self, layout_name):
        """Save the current layout of panels"""
        layout_data = {}
        for panel_id in ['problem_solution_panel', 'documents_panel', 'parts_panel', 'images_panel', 'drawings_panel']:
            panel = self.app.root.ids.get(panel_id)
            if panel:
                layout_data[panel_id] = {
                    'pos': panel.pos,
                    'size': panel.size
                }

        success, message = self.file_manager.save_layout(layout_name, layout_data)
        return success, message

    def get_layout_list(self):
        """Get a list of all available layouts"""
        custom_layouts = self.file_manager.get_available_layouts()
        return list(self.layouts.keys()) + custom_layouts


class SearchEngine:
    """Handles search functionality across the application"""

    def __init__(self, app_instance):
        self.app = app_instance

    def search_problems(self, query):
        """Search through problems"""
        if not query:
            return []

        query = query.lower()
        # Get all problem items from the UI
        problem_list = self.app.root.ids.problem_solution_panel.ids.problem_list
        results = []

        for child in problem_list.children:
            if hasattr(child, 'text') and query in child.text.lower():
                results.append({
                    'type': 'problem',
                    'text': child.text,
                    'widget': child
                })

        return results

    def search_solutions(self, query):
        """Search through solutions"""
        if not query:
            return []

        query = query.lower()
        # Get all solution items from the UI
        solution_list = self.app.root.ids.problem_solution_panel.ids.solution_list
        results = []

        for child in solution_list.children:
            if hasattr(child, 'text') and query in child.text.lower():
                results.append({
                    'type': 'solution',
                    'text': child.text,
                    'widget': child
                })

        return results

    def global_search(self, query):
        """Search across all content"""
        # Combine results from all search methods
        problem_results = self.search_problems(query)
        solution_results = self.search_solutions(query)

        # Additional search areas could be added here

        return problem_results + solution_results

    def highlight_result(self, result):
        """Highlight a search result in the UI"""
        if 'widget' in result:
            # Access the relevant panel
            panel_id = None
            if result['type'] == 'problem':
                panel_id = 'problem_solution_panel'
            elif result['type'] == 'solution':
                panel_id = 'problem_solution_panel'

            if panel_id:
                panel = self.app.root.ids.get(panel_id)
                if panel and not panel.is_expanded:
                    panel.toggle_expand()

                # Scroll to and highlight the widget
                result['widget'].background_color = (0.2, 0.8, 1, 0.3)  # Highlight color

                # Schedule the highlight to be removed after a delay
                def remove_highlight(dt, widget=result['widget']):
                    widget.background_color = (0, 0, 0, 0)  # Remove highlight

                Clock.schedule_once(remove_highlight, 3)  # Remove after 3 seconds