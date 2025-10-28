# Standard library imports
import os
from datetime import datetime
from functools import partial

# Kivy imports
from kivy.metrics import dp
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.image import AsyncImage
from kivy.uix.label import Label
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.uix.scrollview import ScrollView
from kivy.uix.stencilview import StencilView
from kivy.uix.tabbedpanel import TabbedPanel, TabbedPanelItem
from kivy.uix.widget import Widget

# KivyMD imports
from kivymd.app import MDApp
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.button import MDFlatButton, MDRaisedButton
from kivymd.uix.dialog import MDDialog
from kivymd.uix.label import MDLabel
from kivymd.uix.list import IconLeftWidget, OneLineListItem, TwoLineListItem, ThreeLineListItem
from kivymd.uix.snackbar import MDSnackbar
from kivymd.uix.tab import MDTabs
from kivymd.uix.card import MDCard
# Note: MDSeparator removed - not available in all KivyMD versions

# App-specific imports
from modules.configuration.config import DATABASE_PATH_IMAGES_FOLDER
from modules.configuration.log_config import logger
from pop_widgets import Popup, ImageDetailsPopup, PartDetailsPopup
class NonDraggableMDRaisedButton(MDRaisedButton):
    """Button that prevents dragging behavior from parent containers"""

    def on_touch_down(self, touch):
        if instance.collide_point(*touch.pos):
            # Set a flag to track this touch
            touch.grab(self)
            logger.debug(f"NonDraggableMDRaisedButton: on_touch_down at {touch.pos}")
            return super().on_touch_down(touch)
        return False

    def on_touch_move(self, touch):
        if touch.grab_current is self:
            # Prevent propagation to parent
            logger.debug(f"NonDraggableMDRaisedButton: on_touch_move at {touch.pos}")
            return True
        return False

    def on_touch_up(self, touch):
        if touch.grab_current is self:
            touch.ungrab(self)
            logger.debug(f"NonDraggableMDRaisedButton: on_touch_up at {touch.pos}, triggering on_release")
            if self.collide_point(*touch.pos):
                # Manually dispatch the on_release event
                self.dispatch('on_release')
            return True
        return False

# Assuming ScrollableLabel is defined similarly elsewhere in your code:
class ScrollableLabel(Label):
    """Label that properly handles text wrapping and height adjustment."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.size_hint_y = None
        self.height = dp(30)
        self.bind(width=self._update_text_size, texture_size=self._update_height)
        self.halign = kwargs.get('halign', 'left')
        self.valign = kwargs.get('valign', 'top')
    def _update_text_size(self, *args):
        self.text_size = (self.width, None)
    def _update_height(self, *args):
        if self.texture_size[1] > 0:
            self.height = self.texture_size[1] + dp(10)

class ProblemSolutionContent(MDBoxLayout):
    """
    A panel widget split into four sections:
      - Top left: Problems list
      - Top right: Solutions list
      - Bottom left: Tasks list
      - Bottom right: Task Details + Parts Tabs
    """

    def __init__(self, drawing_content, parts_content, **kwargs):
        logger.debug("Entering ProblemSolutionContent.__init__ with kwargs: %s", kwargs)
        super().__init__(**kwargs)

        self.drawing_content = drawing_content  # For updating drawing content
        self.parts_content = parts_content  # For updating parts content

        self.size_hint = (1, 1)
        self.orientation = 'vertical'

        # Create a 2x2 grid layout
        grid = GridLayout(cols=2, rows=2, padding=dp(10), spacing=dp(10))
        grid.size_hint = (1, 1)

        # --- Top Left: Problems ---
        self.problem_box = MDBoxLayout(orientation='vertical', size_hint=(1, 1))
        self.problem_label = Label(text="Problems:", size_hint_y=None, height=dp(40), bold=True)
        self.problem_box.add_widget(self.problem_label)

        self.problem_container = MDBoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(5))
        self.problem_container.bind(minimum_height=self.problem_container.setter('height'))
        self.problem_scroll = ScrollView(size_hint=(1, 1), do_scroll_x=False)
        self.problem_scroll.add_widget(self.problem_container)
        self.problem_box.add_widget(self.problem_scroll)
        grid.add_widget(self.problem_box)

        # --- Top Right: Solutions ---
        self.solution_box = MDBoxLayout(orientation='vertical', size_hint=(1, 1))
        self.solution_label = Label(text="Solutions:", size_hint_y=None, height=dp(40), bold=True)
        self.solution_box.add_widget(self.solution_label)

        self.solution_container = MDBoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(5))
        self.solution_container.bind(minimum_height=self.solution_container.setter('height'))
        self.solution_scroll = ScrollView(size_hint=(1, 1), do_scroll_x=False)
        self.solution_scroll.add_widget(self.solution_container)
        self.solution_box.add_widget(self.solution_scroll)
        grid.add_widget(self.solution_box)

        # --- Bottom Left: Tasks ---
        self.task_box = MDBoxLayout(orientation='vertical', size_hint=(1, 1))
        self.task_label = Label(text="Tasks:", size_hint_y=None, height=dp(40), bold=True)
        self.task_box.add_widget(self.task_label)

        self.task_container = MDBoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(5))
        self.task_container.bind(minimum_height=self.task_container.setter('height'))
        self.task_scroll = ScrollView(size_hint=(1, 1), do_scroll_x=False)
        self.task_scroll.add_widget(self.task_container)
        self.task_box.add_widget(self.task_scroll)
        grid.add_widget(self.task_box)

        # --- Bottom Right: Task Details with Tabs ---
        self.task_details_box = MDBoxLayout(orientation='vertical', size_hint=(1, 1))
        self.task_details_label = Label(text="Task Details:", size_hint_y=None, height=dp(40), bold=True)
        self.task_details_box.add_widget(self.task_details_label)

        # Create a ScreenManager for tabbed content
        self.tab_screen_manager = ScreenManager()

        # Create task details tab content
        self.task_details_screen = Screen(name="task_details")
        self.task_details = ScrollableLabel(text="")
        task_details_scroll = ScrollView(size_hint=(1, 1), do_scroll_x=False)
        task_details_scroll.add_widget(self.task_details)
        self.task_details_screen.add_widget(task_details_scroll)

        # Create position tab content
        self.position_screen = Screen(name="position")
        self.position_details = ScrollableLabel(text="Position information will appear here.")
        position_scroll = ScrollView(size_hint=(1, 1), do_scroll_x=False)
        position_scroll.add_widget(self.position_details)
        self.position_screen.add_widget(position_scroll)

        # Create tools tab content using a ScrollableLabel for consistency
        self.tools_screen = Screen(name="tools")
        self.tools_details = ScrollableLabel(text="Suggested tools will appear here.")
        tools_scroll = ScrollView(size_hint=(1, 1), do_scroll_x=False)
        tools_scroll.add_widget(self.tools_details)
        self.tools_screen.add_widget(tools_scroll)

        # Add all screens to the screen manager
        self.tab_screen_manager.add_widget(self.task_details_screen)
        self.tab_screen_manager.add_widget(self.position_screen)
        self.tab_screen_manager.add_widget(self.tools_screen)

        # Create tab buttons
        tab_buttons = MDBoxLayout(orientation='horizontal', size_hint_y=None, height=dp(48), spacing=dp(2))

        self.details_tab_btn = MDRaisedButton(
            text="Task Details",
            size_hint_x=1,
            md_bg_color=[0.2, 0.6, 0.8, 1]
        )
        self.details_tab_btn.bind(on_release=lambda x: self.switch_tab("task_details"))

        self.position_tab_btn = MDRaisedButton(
            text="Position",
            size_hint_x=1,
            md_bg_color=[0.5, 0.5, 0.5, 1]
        )
        self.position_tab_btn.bind(on_release=lambda x: self.switch_tab("position"))

        self.tools_tab_btn = MDRaisedButton(
            text="Suggested Tools",
            size_hint_x=1,
            md_bg_color=[0.5, 0.5, 0.5, 1]
        )
        self.tools_tab_btn.bind(on_release=lambda x: self.switch_tab("tools"))

        tab_buttons.add_widget(self.details_tab_btn)
        tab_buttons.add_widget(self.position_tab_btn)
        tab_buttons.add_widget(self.tools_tab_btn)

        # Add tab buttons and screen manager to the task details box
        self.task_details_box.add_widget(tab_buttons)
        self.task_details_box.add_widget(self.tab_screen_manager)

        # Add task details box to the grid
        grid.add_widget(self.task_details_box)

        # Add grid to the main widget
        self.add_widget(grid)

        # Track the current active tab
        self.current_tab = "task_details"

        self.app = MDApp.get_running_app()
        logger.debug("Finished ProblemSolutionContent.__init__")

    def switch_tab(self, tab_name):
        logger.debug("Switching to tab: %s", tab_name)
        self.tab_screen_manager.current = tab_name
        self.current_tab = tab_name

        # Update button colors
        active_color = [0.2, 0.6, 0.8, 1]
        inactive_color = [0.5, 0.5, 0.5, 1]

        self.details_tab_btn.md_bg_color = active_color if tab_name == "task_details" else inactive_color
        self.position_tab_btn.md_bg_color = active_color if tab_name == "position" else inactive_color
        self.tools_tab_btn.md_bg_color = active_color if tab_name == "tools" else inactive_color

    def test_direct_click(self, instance):
        logger.debug("Entering test_direct_click with instance: %s", instance)
        print("=" * 50)
        print("TEST DIRECT CLICK - START")
        print("=" * 50)
        try:
            logger.debug("Attempting to show dialog in test_direct_click")
            from kivymd.uix.dialog import MDDialog
            from kivymd.uix.button import MDFlatButton

            dialog = MDDialog(
                title="Test Button Clicked",
                text="The test button click event was detected!",
                buttons=[
                    MDFlatButton(
                        text="OK",
                        on_release=lambda x: dialog.dismiss()
                    )
                ]
            )
            dialog.open()
            logger.debug("Dialog opened successfully in test_direct_click")
            self.task_details.text = "Test button was clicked at " + str(datetime.utcnow())
            logger.debug("Updated task details text in test_direct_click")
        except Exception as e:
            logger.exception("ERROR in test_direct_click:")

    """def test_click(self, instance):
        logger.debug("Entering test_click with instance: %s", instance)
        print("=" * 50)
        print("TEST PROBLEM BUTTON CLICK - START")
        print("=" * 50)
        try:
            logger.debug("Attempting to show snackbar in test_click")
            from kivymd.uix.snackbar import MDSnackbar
            snackbar = MDSnackbar(text="Problem test button clicked!")
            snackbar.open()
            logger.debug("Snackbar opened successfully in test_click")
            self.test_direct_solution_display()
        except Exception as e:
            logger.exception("ERROR in test_click:")

    def test_solution_click(self, instance):
        logger.debug("Entering test_solution_click with instance: %s", instance)
        print("Solution test button clicked!")
        logger.debug("Solution test button clicked!")
        try:
            from kivymd.uix.snackbar import MDSnackbar
            snackbar = MDSnackbar()
            snackbar.text = "Solution test button clicked!"
            snackbar.open()
            logger.debug("Snackbar opened successfully in test_solution_click")
            from modules.emtacdb.emtacdb_fts import Solution
            solution = self.app.data_service.session.query(Solution).filter(Solution.id == 1).first()
            if solution:
                logger.debug("Found solution: %s, manually showing tasks", solution.name)
                print(f"Found solution: {solution.name}, manually showing tasks")
                self.on_solution_button_click(solution)
            else:
                logger.debug("Could not find solution with ID 1")
                print("Could not find solution with ID 1")
        except Exception as e:
            logger.exception("Error in test_solution_click:")"""

    def update_position_details_by_task(self, task_id):
        """
        Updates the position tab with details from the Position record associated with the given task_id.
        It calls the data service's get_position_by_task_id method to retrieve the Position.
        """
        # Use the data service method now
        position = self.app.data_service.get_position_by_task_id(task_id)
        if position:
            def get_detail(record):
                if record:
                    # For AssetNumber, use 'number' if available; otherwise, use 'name'
                    detail = record.name if hasattr(record, "name") else getattr(record, "number", "N/A")
                    description = getattr(record, "description", "")
                    return f"{detail}{(' - ' + description) if description else ''}"
                return "N/A"

            site_location = get_detail(position.site_location)
            area = get_detail(position.area)
            equipment_group = get_detail(position.equipment_group)
            model = get_detail(position.model)
            asset_number = get_detail(position.asset_number)
            location = get_detail(position.location)  # Added location
            subassembly = get_detail(position.subassembly)
            component_assembly = get_detail(position.component_assembly)
            assembly_view = get_detail(position.assembly_view)

            position_info = (
                f"Position ID: {position.id}\n\n"
                f"Site Location: {site_location}\n\n"
                f"Area: {area}\n\n"
                f"Equipment Group: {equipment_group}\n\n"
                f"Model: {model}\n\n"
                f"Asset Number: {asset_number}\n\n"
                f"Location: {location}\n\n"  # Added location to output
                f"Subassembly: {subassembly}\n\n"
                f"Component Assembly: {component_assembly}\n\n"
                f"Assembly View: {assembly_view}\n\n"
            )
            self.position_details.text = position_info
        else:
            self.position_details.text = f"No position information found for Task ID: {task_id}"

    def simple_button_click(self, instance):
        print("Simple button click fired")
        logger.debug("Simple button click fired for instance: %s", instance)

    def debug_button_click(self, button):
        logger.debug("Entering debug_button_click with button: %s", button)
        print("DEBUG BUTTON CLICKED")
        try:
            problem_id = getattr(button, 'problem_id', None)
            if problem_id:
                logger.debug("Problem button clicked with ID: %s", problem_id)
                self.solution_container.clear_widgets()
                self.solution_container.add_widget(
                    Label(text=f"Problem {problem_id} button clicked!",
                          color=(0, 1, 0, 1), size_hint_y=None, height=dp(30))
                )
                from modules.emtacdb.emtacdb_fts import Solution
                solutions = self.app.data_service.session.query(Solution).filter(
                    Solution.problem_id == problem_id).all()
                logger.debug("Found %d solutions for problem ID %s", len(solutions), problem_id)
                self.solution_container.add_widget(
                    Label(text=f"Found {len(solutions)} solutions",
                          color=(0, 1, 0, 1), size_hint_y=None, height=dp(30))
                )
                for solution in solutions:
                    from kivy.uix.button import Button
                    sol_btn = Button(
                        text=f"SOLUTION: {solution.name}",
                        size_hint_y=None,
                        height=dp(50),
                        background_color=(0, 0, 1, 1)
                    )
                    self.solution_container.add_widget(sol_btn)
                    logger.debug("Added solution button for: %s (ID: %s)", solution.name, solution.id)
            else:
                logger.debug("No problem_id found in debug_button_click")
        except Exception as e:
            logger.exception("Error in debug_button_click:")
            self.solution_container.clear_widgets()
            self.solution_container.add_widget(
                Label(text=f"Error: {str(e)}",
                      color=(1, 0, 0, 1), size_hint_y=None, height=dp(30))
            )

    def debug_show_solutions(self, button):
        logger.debug("Entering debug_show_solutions with button: %s", button)
        print("DEBUG SHOW SOLUTIONS BUTTON CLICKED")
        try:
            self.solution_container.clear_widgets()
            self.solution_container.add_widget(
                Label(text="DIRECT TEST SOLUTIONS",
                      color=(1, 1, 0, 1), size_hint_y=None, height=dp(30))
            )
            from kivy.uix.button import Button
            test_btn = Button(
                text="TEST SOLUTION BUTTON",
                size_hint_y=None,
                height=dp(50),
                background_color=(0, 0, 1, 1)
            )
            self.solution_container.add_widget(test_btn)
            from modules.emtacdb.emtacdb_fts import Solution
            solution = self.app.data_service.session.query(Solution).filter(Solution.id == 1).first()
            if solution:
                self.solution_container.add_widget(
                    Label(text=f"Found solution: {solution.name}",
                          color=(0, 1, 0, 1), size_hint_y=None, height=dp(30))
                )
                sol_btn = Button(
                    text=f"REAL SOLUTION: {solution.name}",
                    size_hint_y=None,
                    height=dp(50),
                    background_color=(0, 1, 0, 1)
                )
                self.solution_container.add_widget(sol_btn)
                logger.debug("Displayed real solution button for: %s (ID: %s)", solution.name, solution.id)
            else:
                self.solution_container.add_widget(
                    Label(text="No solution found with ID 1",
                          color=(1, 0, 0, 1), size_hint_y=None, height=dp(30))
                )
                logger.debug("No solution found with ID 1 in debug_show_solutions")
        except Exception as e:
            logger.exception("Error in debug_show_solutions:")
            self.solution_container.clear_widgets()
            self.solution_container.add_widget(
                Label(text=f"Error: {str(e)}",
                      color=(1, 0, 0, 1), size_hint_y=None, height=dp(30))
            )

    def on_problem_button_click(self, button):
        logger.debug("Entering on_problem_button_click with button: %s", button)
        logger.debug(f"PROBLEM BUTTON CLICKED: {button}")
        logger.debug(f"Has problem attribute: {hasattr(button, 'problem')}")
        try:
            if hasattr(button, 'problem'):
                problem = button.problem
                logger.debug(f"Problem found: {problem.name} (ID: {problem.id})")
            else:
                problem = button
                logger.debug(f"Using button as problem: {problem}")
            if problem:
                logger.debug("Problem button clicked: %s (ID: %s)", problem.name, problem.id)
                self.solution_container.clear_widgets()
                self.solution_container.add_widget(Label(text=f"Loading solutions for {problem.name}...",
                                                         size_hint_y=None, height=dp(30)))
                solutions = self.app.data_service.get_solutions_by_problem(problem.id)
                logger.debug("Found %d solutions for problem ID %s", len(solutions), problem.id)
                for i, solution in enumerate(solutions):
                    logger.debug("Solution %d: ID=%s, Name=%s", i + 1, solution.id, solution.name)
                self.display_solutions(problem, solutions)
            else:
                logger.debug("Problem button clicked but no problem object found")
        except Exception as e:
            logger.exception("Error in on_problem_button_click:")
            self.solution_container.clear_widgets()
            self.solution_container.add_widget(
                Label(text=f"Error loading solutions: {e}",
                      color=(1, 0, 0, 1), size_hint_y=None, height=dp(30))
            )

    def display_solutions(self, problem, solutions):
        try:
            self.solution_container.clear_widgets()
            self.solution_container.add_widget(
                Label(
                    text=f"Solutions for: {problem.name}",
                    size_hint_y=None,
                    height=dp(40),
                    bold=True
                )
            )
            if not solutions:
                self.solution_container.add_widget(
                    Label(
                        text="No solutions found for this problem",
                        size_hint_y=None,
                        height=dp(30)
                    )
                )
                return
            for solution in solutions:
                solution_btn = MDRaisedButton(
                    text=f"Solution: {solution.name}",
                    size_hint_x=0.9,
                    size_hint_y=None,
                    height=dp(60),
                    pos_hint={"center_x": 0.5},
                    md_bg_color=[0.2, 0.6, 0.8, 1]
                )
                solution_btn.solution = solution
                solution_btn.bind(on_release=self.on_solution_button_click)
                self.solution_container.add_widget(solution_btn)
            self.solution_container.do_layout()
        except Exception as e:
            logger.exception("Error displaying solutions:")
            from kivymd.uix.snackbar import MDSnackbar
            snackbar = MDSnackbar()
            snackbar.text = f"Error displaying solutions: {e}"
            snackbar.open()

    def on_solution_button_click(self, button):
        logger.debug("Entering on_solution_button_click with button: %s", button)
        if hasattr(button, 'solution'):
            solution = button.solution
        else:
            solution = button
        if solution:
            logger.debug("Solution button clicked: %s (ID: %s)", solution.name, solution.id)
            self.task_container.clear_widgets()
            self.task_container.add_widget(Label(text=f"Loading tasks for {solution.name}...",
                                                 size_hint_y=None, height=dp(30)))
            tasks = self.app.data_service.get_tasks_by_solution(solution.id)
            logger.debug("Found %d tasks for solution ID %s", len(tasks), solution.id)
            self.display_tasks(solution, tasks)
        else:
            logger.debug("Solution button clicked but no solution object found")

    def display_tasks(self, solution, tasks):
        logger.debug("Entering display_tasks for solution: %s (ID: %s)", solution.name, solution.id)
        self.task_container.clear_widgets()
        self.task_container.add_widget(Label(text=f"Tasks for: {solution.name}",
                                             size_hint_y=None, height=dp(40), bold=True))
        if not tasks:
            self.task_container.add_widget(
                OneLineListItem(text=f"No tasks found for this solution",
                                text_color=[1, 0, 0, 1])
            )
            logger.debug("No tasks found for solution in display_tasks")
            return
        for i, task in enumerate(tasks):
            logger.debug("Adding task button for: %s (ID: %s)", task.name, task.id)
            task_btn = MDRaisedButton(
                text=f"Task: {task.name}",
                size_hint_x=0.9,
                size_hint_y=None,
                height=dp(50),
                pos_hint={"center_x": 0.5},
                md_bg_color=[0.1, 0.7, 0.3, 1]
            )
            task_btn.task = task
            task_btn.bind(on_release=self.on_task_button_click)
            self.task_container.add_widget(task_btn)
            self.task_container.add_widget(Widget(size_hint_y=None, height=dp(5)))
        self.task_container.height = self.task_container.minimum_height
        logger.debug("Finished displaying tasks in display_tasks")

    def on_task_button_click(self, button):
        try:
            task = button.task
            logger.debug(f"Task button clicked: {task.name} (ID: {task.id})")

            # Update Task Details text
            task_info = (
                f"Task: {task.name}\n\n"
                f"Description: {task.description or 'No description'}"
            )
            self.task_details.text = task_info

            # Switch to the Task Details tab
            self.switch_tab("task_details")

            # Update the Tools tab
            self.update_tools_tab(task)

            # Update the Position details tab
            self.update_position_details_by_task(task.id)

            # Update the Drawing content
            self.drawing_content.update_for_task(task.id)

            # Update the Parts content
            self.parts_content.update_for_task(task.id)

            # Update Documents panel
            MDApp.get_running_app().documents_content.update_for_task(task.id)

            # --- NEW CODE: Update Images content ---
            # Fetch position ID from task using DataService
            position = self.app.data_service.get_position_by_task_id(task.id)
            position_id = position.id if position else None

            image_content_widget = MDApp.get_running_app().image_content_widget

            # Update Task Images Tab
            image_content_widget.update_for_task(task.id)

            # Update Position Images Tab (if position ID is available)
            if position_id:
                image_content_widget.update_for_position(position_id)
            else:
                image_content_widget.position_images_list.clear_widgets()
                image_content_widget.position_images_layout.add_widget(
                    MDLabel(
                        text="No position associated with this task.",
                        size_hint_y=None,
                        height=dp(30)
                    )
                )

        except Exception as e:
            logger.exception("Error processing task button click:")
            from kivymd.uix.snackbar import MDSnackbar
            snackbar = MDSnackbar()
            snackbar.text = f"Error processing task: {e}"
            snackbar.open()

    def update_tools_tab(self, task):
        """Update the Suggested Tools tab to display tools linked to the task."""
        try:
            tools = self.app.data_service.get_tools_by_task(task.id)
            tool_text = f"Suggested Tools for {task.name}:\n\n"
            if tools and len(tools) > 0:
                for tool in tools:
                    tool_text += f"- {tool.name}: {tool.description}\n"
            else:
                tool_text += "No tools found for this task."
            self.tools_details.text = tool_text
        except Exception as e:
            self.tools_details.text = f"Error loading tools: {e}"

    def show_solutions_for_problem(self, problem):
        logger.debug("Entering show_solutions_for_problem for problem: %s", problem.name)
        print(f"Legacy show_solutions_for_problem called for {problem.name}")
        self.on_problem_button_click(problem)

    def show_tasks_for_solution(self, solution):
        logger.debug("Entering show_tasks_for_solution for solution: %s", solution.name)
        print(f"Legacy show_tasks_for_solution called for {solution.name}")
        self.on_solution_button_click(solution)

    def show_task_details(self, task):
        logger.debug("Entering show_task_details for task: %s", task.name)
        print(f"Legacy show_task_details called for {task.name}")
        dummy_button = type('obj', (object,), {'task': task})
        self.on_task_button_click(dummy_button)

    def test_direct_solution_display(self):
        logger.debug("Entering test_direct_solution_display")
        try:
            print("=" * 50)
            print("DIRECT SOLUTION DISPLAY TEST")
            print("=" * 50)
            self.problem_container.clear_widgets()
            self.solution_container.clear_widgets()
            self.task_container.clear_widgets()
            self.problem_container.add_widget(
                Label(text="DIRECT TEST MODE", color=(1, 1, 0, 1),
                      size_hint_y=None, height=dp(30))
            )
            from kivy.uix.button import Button
            basic_btn = Button(
                text="BASIC BUTTON TEST",
                size_hint_y=None,
                height=dp(50),
                background_color=(1, 0, 0, 1)
            )
            basic_btn.bind(on_press=self._on_basic_test)
            self.problem_container.add_widget(basic_btn)
            from modules.emtacdb.emtacdb_fts import Problem, Solution
            problem = self.app.data_service.session.query(Problem).filter(Problem.id == 1).first()
            if problem:
                solutions = self.app.data_service.session.query(Solution).filter(
                    Solution.problem_id == problem.id).all()
                self.problem_container.add_widget(
                    Label(text=f"Problem: {problem.name} has {len(solutions)} solutions",
                          color=(0, 1, 0, 1), size_hint_y=None, height=dp(40))
                )
                self.solution_container.clear_widgets()
                self.solution_container.add_widget(
                    Label(text="DIRECT SOLUTIONS TEST", color=(1, 1, 0, 1),
                          size_hint_y=None, height=dp(40))
                )
                for solution in solutions:
                    from kivy.uix.button import Button
                    sol_btn = Button(
                        text=f"SOLUTION: {solution.name}",
                        size_hint_y=None,
                        height=dp(60),
                        background_color=(0, 0, 1, 1)
                    )
                    self.solution_container.add_widget(sol_btn)
                    self.solution_container.add_widget(Widget(size_hint_y=None, height=dp(10)))
                logger.debug("Displayed direct solution test for problem: %s", problem.name)
            else:
                self.problem_container.add_widget(
                    Label(text="ERROR: Could not find Problem ID 1",
                          color=(1, 0, 0, 1), size_hint_y=None, height=dp(40))
                )
                logger.debug("Problem ID 1 not found in test_direct_solution_display")
        except Exception as e:
            logger.exception("Error in test_direct_solution_display:")
            self.problem_container.add_widget(
                Label(text=f"Error: {str(e)}", color=(1, 0, 0, 1),
                      size_hint_y=None, height=dp(40))
            )

    def _on_basic_test(self, instance):
        logger.debug("Entering _on_basic_test with instance: %s", instance)
        print("Basic button clicked")
        self.task_details.text = "Basic button was clicked successfully!"
        logger.debug("Finished _on_basic_test")

class PartsContent(MDBoxLayout):
    """Content for Parts panel with two tabs: Task Parts and Position Parts."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint = (1, 1)  # Fill available space - SAME AS DOCUMENTS

        # Reference to app
        self.app = MDApp.get_running_app()

        # Create a TabbedPanel for parts - EXACT SAME AS DOCUMENTS
        self.tabs = TabbedPanel(
            do_default_tab=False,
            tab_pos='top_mid',
            size_hint=(1, 1)  # This is the key difference!
        )
        self.tabs.tab_width = dp(150)  # Same as Documents

        # Create TabbedPanelItems
        self.task_tab = TabbedPanelItem(text="Task Parts")
        self.position_tab = TabbedPanelItem(text="Position Parts")

        # Build scrollable containers for each tab
        self._build_task_tab()
        self._build_position_tab()

        self.tabs.add_widget(self.task_tab)
        self.tabs.add_widget(self.position_tab)
        self.tabs.default_tab = self.task_tab

        self.add_widget(self.tabs)

    def _build_task_tab(self):
        scroll = ScrollView(do_scroll_x=False)
        container = MDBoxLayout(
            orientation='vertical',
            size_hint_y=None,
            padding=dp(10),
            spacing=dp(5)
        )
        container.bind(minimum_height=container.setter('height'))
        scroll.add_widget(container)
        self.task_parts_list = container
        self.task_tab.add_widget(scroll)

    def _build_position_tab(self):
        scroll = ScrollView(do_scroll_x=False)
        container = MDBoxLayout(
            orientation='vertical',
            size_hint_y=None,
            padding=dp(10),
            spacing=dp(5)
        )
        container.bind(minimum_height=container.setter('height'))
        scroll.add_widget(container)
        self.position_parts_list = container
        self.position_tab.add_widget(scroll)

    def update_for_task(self, task_id):
        """Update the Task Parts tab based on the selected task."""
        self.task_parts_list.clear_widgets()

        if not task_id:
            self.task_parts_list.add_widget(
                OneLineListItem(text="No task selected")
            )
            return

        try:
            parts = self.app.data_service.get_parts_by_task(task_id)

            if not parts:
                self.task_parts_list.add_widget(
                    OneLineListItem(text="No parts linked to this task")
                )
                return

            for part in parts:
                # Use TwoLineListItem like Documents does
                item = TwoLineListItem(
                    text=part.part_number,
                    secondary_text=f"{part.name or 'No name'} - Type: {part.type or 'N/A'}"
                )
                item.part = part
                item.bind(on_release=lambda x, p=part: self.on_part_selected(p))
                self.task_parts_list.add_widget(item)

        except Exception as e:
            logger.error(f"Error updating task parts: {e}")
            self.task_parts_list.add_widget(
                OneLineListItem(text=f"Error: {str(e)}")
            )

    def update_for_position(self, position_id):
        """Update the position parts list based on a given position_id."""
        self.position_parts_list.clear_widgets()

        if not position_id:
            self.position_parts_list.add_widget(
                OneLineListItem(text="No position selected")
            )
            return

        try:
            parts = self.app.data_service.get_parts_by_position(position_id)

            if not parts:
                self.position_parts_list.add_widget(
                    OneLineListItem(text="No parts linked to this position")
                )
                return

            for part in parts:
                item = TwoLineListItem(
                    text=part.part_number,
                    secondary_text=f"{part.name or 'No name'} - Type: {part.type or 'N/A'}"
                )
                item.part = part
                item.bind(on_release=lambda x, p=part: self.on_part_selected(p))
                self.position_parts_list.add_widget(item)

        except Exception as e:
            logger.error(f"Error loading position parts: {e}")
            self.position_parts_list.add_widget(
                OneLineListItem(text=f"Error: {e}")
            )

    def on_part_selected(self, part):
        """Handle part selection"""
        try:
            logger.debug(f"Opening popup for part: {part.part_number}")

            # Use the PartDetailsPopup from pop_widgets
            content = PartDetailsPopup(part)

            popup = Popup(
                title=f"Part Details: {part.part_number}",
                content=content,
                size_hint=(0.8, 0.8),
                auto_dismiss=True
            )
            popup.open()

        except Exception as e:
            logger.error(f"Error selecting part: {e}")
            # Show error in snackbar
            snackbar = MDSnackbar(
                text=f"Error opening part details: {str(e)}"
            )
            snackbar.open()

class DocumentsContent(MDBoxLayout):
    """Content for Documents panel with two tabs:
       - Task Documents
       - Position Documents.

       Document links are clickable and open a popup showing the original document format.
    """

    @property
    def app(self):
        return MDApp.get_running_app()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint = (1, 1)  # Fill available space

        # Create a TabbedPanel for documents
        self.tabs = TabbedPanel(
            do_default_tab=False,
            tab_pos='top_mid',
            size_hint=(1, 1)
        )
        self.tabs.tab_width = dp(150)

        # Create TabbedPanelItems
        self.task_tab = TabbedPanelItem(text="Task Documents")
        self.position_tab = TabbedPanelItem(text="Position Documents")

        # Build scrollable containers for each tab
        self._build_task_tab()
        self._build_position_tab()

        self.tabs.add_widget(self.task_tab)
        self.tabs.add_widget(self.position_tab)
        self.tabs.default_tab = self.task_tab

        self.add_widget(self.tabs)

    def _build_task_tab(self):
        scroll = ScrollView(do_scroll_x=False)
        container = MDBoxLayout(
            orientation='vertical',
            size_hint_y=None,
            padding=dp(10),
            spacing=dp(5)
        )
        container.bind(minimum_height=container.setter('height'))
        scroll.add_widget(container)
        # Store reference for updating task documents
        self.task_docs_list = container
        self.task_tab.add_widget(scroll)

    def _build_position_tab(self):
        scroll = ScrollView(do_scroll_x=False)
        container = MDBoxLayout(
            orientation='vertical',
            size_hint_y=None,
            padding=dp(10),
            spacing=dp(5)
        )
        container.bind(minimum_height=container.setter('height'))
        scroll.add_widget(container)
        # Store reference for updating position documents
        self.position_docs_list = container
        self.position_tab.add_widget(scroll)

    def update_for_task(self, task_id):
        """Update the Task Documents tab based on the selected task."""
        # Clear any existing document items from the task documents container
        self.task_docs_list.clear_widgets()

        if not task_id:
            self.task_docs_list.add_widget(
                OneLineListItem(text="No task selected")
            )
            return

        try:
            # Retrieve task documents using your DataService method
            documents = self.app.data_service.get_documents_by_task_id(task_id)

            if not documents:
                self.task_docs_list.add_widget(
                    OneLineListItem(text="No documents linked to this task")
                )
                return

            # Iterate through the retrieved documents and add them as list items.
            for doc in documents:
                # Create a list item for each document.
                # You may choose TwoLineListItem or any other widget style.
                item = TwoLineListItem(
                    text=doc.title or f"Document {doc.id}",
                    secondary_text=(f"Rev: {doc.rev}" if getattr(doc, "rev", None) else "No revision")
                )
                # Store the document id on the widget for later use.
                item.doc_id = doc.id
                # Bind a callback to open the document's details when the item is clicked.
                item.bind(on_release=self.on_document_selected)
                self.task_docs_list.add_widget(item)

        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Error updating task documents: {e}")
            self.task_docs_list.add_widget(
                OneLineListItem(text=f"Error: {str(e)}")
            )

    def update_for_position(self, position_id):
        """Update the position documents list based on a given position_id."""
        self.position_docs_list.clear_widgets()
        if not position_id:
            self.position_docs_list.add_widget(OneLineListItem(text="No position selected"))
            return
        try:
            # Retrieve position documents using the DataService method.
            documents = self.app.data_service.get_documents_by_position(position_id)
            if not documents:
                self.position_docs_list.add_widget(OneLineListItem(text="No documents linked to this position"))
                return

            for doc in documents:
                doc_item = TwoLineListItem(
                    text=doc.title or f"Document {doc.id}",
                    secondary_text=f"Rev: {doc.rev}" if getattr(doc, "rev", None) else "No revision"
                )
                doc_item.doc_id = doc.id
                doc_item.bind(on_release=self.on_document_selected)
                self.position_docs_list.add_widget(doc_item)
        except Exception as e:
            logger.error(f"Error loading position documents: {e}")
            self.position_docs_list.add_widget(OneLineListItem(text=f"Error: {e}"))

    def on_document_selected(self, list_item):
        try:
            doc_id = getattr(list_item, 'doc_id', None)
            if not doc_id:
                return
            # Retrieve document details using your DataService method.
            doc = self.app.data_service.get_document_details(doc_id)
            if not doc:
                return
            # Build content for the popup.
            content = MDBoxLayout(orientation='vertical', spacing=dp(10), padding=dp(10))
            content.add_widget(Label(text=doc.content or "Document content not available"))
            # If a file path exists, add an Open Document button.
            if doc.file_path and os.path.exists(doc.file_path):
                open_btn = MDRaisedButton(
                    text="Open Document",
                    size_hint=(None, None),
                    size=(dp(150), dp(40)),
                    pos_hint={'center_x': 0.5}
                )
                open_btn.doc_path = doc.file_path
                open_btn.bind(on_release=self.open_document)
                content.add_widget(open_btn)
            popup = Popup(
                title=doc.title or f"Document {doc.id}",
                content=content,
                size_hint=(0.8, 0.8),
                auto_dismiss=True
            )
            popup.open()
        except Exception as e:
            logger.error(f"Error selecting document: {e}")

    def open_document(self, button):
        try:
            doc_path = getattr(button, 'doc_path', None)
            if not doc_path:
                return
            if platform.system() == 'Windows':
                os.startfile(doc_path)
            elif platform.system() == 'Darwin':
                subprocess.call(['open', doc_path])
            else:
                subprocess.call(['xdg-open', doc_path])
        except Exception as e:
            logger.error(f"Error opening document: {e}")
            snackbar = MDSnackbar()
            snackbar.text = f"Error opening document: {e}"
            snackbar.open()

class ImagesContent(MDBoxLayout):
    """Content for Images panel with two tabs:
       - Task Images
       - Position Images
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.size_hint = (1, 1)
        self.app = MDApp.get_running_app()
        self._current_image_obj = None
        self._current_full_image_path = ""

        self.tabs = TabbedPanel(
            do_default_tab=False,
            tab_pos='top_mid',
            size_hint=(1, 1),
            tab_width=dp(200)
        )

        self.task_tab = TabbedPanelItem(text="Task Images")
        self.position_tab = TabbedPanelItem(text="Position Images")

        self.task_images_list = self._create_image_list_container()
        self.position_images_list = self._create_image_list_container()

        self.task_tab.add_widget(self.task_images_list.parent)
        self.position_tab.add_widget(self.position_images_list.parent)

        self.tabs.add_widget(self.task_tab)
        self.tabs.add_widget(self.position_tab)
        self.tabs.default_tab = self.task_tab
        self.add_widget(self.tabs)

        # Preview + Description panel
        self.preview_label = MDLabel(text="Image Preview:", size_hint_y=None, height=dp(30), bold=True)
        self.image_preview = AsyncImage(
            source='',
            allow_stretch=True,
            keep_ratio=True,
            size_hint=(1, 1)  # let it take all available space in whatever parent layout
        )

        self.image_preview.bind(on_touch_down=self._on_image_preview_touch)

        self.description_label = MDLabel(text="Description:", size_hint_y=None, height=dp(30), bold=True)
        self.image_description = ScrollableLabel(text="")
        self.description_scroll = ScrollView(size_hint=(1, None), height=dp(100))
        self.description_scroll.add_widget(self.image_description)

        self.add_widget(self.preview_label)
        self.add_widget(self.image_preview)
        self.add_widget(self.description_label)
        self.add_widget(self.description_scroll)

    def _create_image_list_container(self):
        scroll = ScrollView(do_scroll_x=False, size_hint=(1, None), height=dp(150))
        container = MDBoxLayout(orientation='vertical', size_hint_y=None, spacing=dp(5))
        container.bind(minimum_height=container.setter('height'))
        scroll.add_widget(container)
        container.parent = scroll
        return container

    def update_for_task(self, task_id):
        self._update_image_list(task_id, is_task=True)

    def update_for_position(self, position_id):
        self._update_image_list(position_id, is_task=False)

    def _update_image_list(self, obj_id, is_task=True):
        container = self.task_images_list if is_task else self.position_images_list
        container.clear_widgets()

        if not obj_id:
            container.add_widget(MDLabel(text="Nothing selected", size_hint_y=None, height=dp(30)))
            return

        try:
            images = self.app.data_service.get_images_by_task(
                obj_id) if is_task else self.app.data_service.get_images_by_position(obj_id)
            if not images:
                container.add_widget(MDLabel(text="No images found", size_hint_y=None, height=dp(30)))
                return

            for image in images:
                item = TwoLineListItem(
                    text=image.get('title', f"Image"),
                    secondary_text=(image.get('description', '')[:50] + "...") if image.get('description') and len(
                        image.get('description')) > 50 else image.get('description', 'No description')
                )
                item.image_obj = image  # still store the dict
                item.bind(on_release=self.on_image_selected)
                container.add_widget(item)

        except Exception as e:
            logger.error(f"Error updating images: {e}")
            container.add_widget(MDLabel(text=f"Error: {e}", size_hint_y=None, height=dp(30)))

    def on_image_selected(self, list_item):
        try:
            image = getattr(list_item, 'image_obj', None)
            if not image:
                return

            file_path = image.get('file_path')
            full_path = os.path.abspath(os.path.join(DATABASE_PATH_IMAGES_FOLDER, file_path)).replace('\\', '/')
            self._current_full_image_path = full_path
            self._current_image_obj = image

            if os.path.exists(full_path):
                self.image_preview.source = full_path
                self.image_description.text = image.get('description', "No description")
            else:
                self.image_preview.source = ''
                self.image_description.text = "Image file not found"

        except Exception as e:
            logger.error(f"Error selecting image: {e}")
            self.image_preview.source = ''
            self.image_description.text = f"Error: {e}"

    def _on_image_preview_touch(self, instance, touch):
        if instance.collide_point(*touch.pos):
            self.open_image_popup()

    def open_image_popup(self):
        if not self._current_full_image_path:
            return

        image = self._current_image_obj or {}

        content = MDBoxLayout(orientation='vertical', spacing=dp(10), padding=dp(10))

        if os.path.exists(self._current_full_image_path):
            img = AsyncImage(
                source=self._current_full_image_path,
                allow_stretch=True,
                keep_ratio=True,
                size_hint=(1, 0.8)
            )
            content.add_widget(img)
        else:
            content.add_widget(MDLabel(text="Image not found", size_hint_y=None, height=dp(40)))

        desc = MDLabel(text=image.get('description', "No description"), size_hint_y=None, height=dp(60))
        content.add_widget(desc)

        popup = Popup(
            title=image.get('title', "Image Preview"),
            content=content,
            size_hint=(0.85, 0.85),
            auto_dismiss=True
        )
        popup.open()

class DrawingsContent(MDBoxLayout):
    """Content for Drawings panel - read only"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = 'vertical'
        self.adaptive_height = False

        # Create a standard Kivy TabbedPanel instead of MDTabs
        self.tabs = TabbedPanel(
            do_default_tab=False,
            tab_pos='top_mid',
            size_hint_y=None,
            height=dp(400)  # Adjust height as needed
        )

        # Create tab items
        self.tabs.tab_width = dp(200)
        self.task_tab = TabbedPanelItem(text="Task Prints")
        self.position_tab = TabbedPanelItem(text="Position Prints")

        # Create content layouts for each tab
        self.task_content = MDBoxLayout(orientation='vertical', spacing=dp(10), padding=dp(10))
        self.position_content = MDBoxLayout(orientation='vertical', spacing=dp(10), padding=dp(10))

        # Set up content for each tab
        self._setup_tab_content(self.task_content, "Task Prints")
        self._setup_tab_content(self.position_content, "Position Prints")

        # Add content to tabs
        self.task_tab.add_widget(self.task_content)
        self.position_tab.add_widget(self.position_content)

        # Add tabs to tab panel
        self.tabs.add_widget(self.task_tab)
        self.tabs.add_widget(self.position_tab)

        # Set default tab
        self.tabs.default_tab = self.task_tab

        # Add the tabbed panel to the main layout
        self.add_widget(self.tabs)

        self.app = MDApp.get_running_app()

    def _setup_tab_content(self, container, tab_name):
        # Create a scroll view for the content
        scroll_view = ScrollView(
            do_scroll_x=False,
            do_scroll_y=True,
            bar_width=dp(5)
        )

        # Create the content container
        content_container = MDBoxLayout(
            orientation='vertical',
            spacing=dp(10),
            padding=dp(10),
            size_hint_y=None
        )
        content_container.bind(minimum_height=content_container.setter('height'))

        # Add drawings section
        drawings_label = Label(
            text=f"{tab_name} Drawings:",
            size_hint_y=None,
            height=dp(40),
            bold=True
        )
        content_container.add_widget(drawings_label)

        # Create drawings container
        drawings_container = MDBoxLayout(
            orientation='vertical',
            size_hint_y=None,
            spacing=dp(5)
        )
        drawings_container.bind(minimum_height=drawings_container.setter('height'))

        # Store references to these containers based on tab name
        if tab_name == "Task Prints":
            self.task_prints_container = drawings_container
        else:
            self.position_prints_container = drawings_container

        drawings_scroll = ScrollView(
            size_hint=(1, None),
            height=dp(150),
            do_scroll_x=False
        )
        drawings_scroll.add_widget(drawings_container)
        content_container.add_widget(drawings_scroll)



        # Add details section
        details_label = Label(
            text="Drawing Details:",
            size_hint_y=None,
            height=dp(40),
            bold=True
        )
        content_container.add_widget(details_label)

        details_layout = GridLayout(
            cols=2,
            size_hint_y=None,
            row_default_height=dp(40),
            row_force_default=True
        )
        details_layout.bind(minimum_height=details_layout.setter('height'))

        # Store references to details layout based on tab name
        if tab_name == "Task Prints":
            self.task_details_layout = details_layout
        else:
            self.position_details_layout = details_layout

        # Set a common details layout (will be updated when tabs change)
        self.details_layout = details_layout

        details_scroll = ScrollView(
            size_hint=(1, None),
            height=dp(150),
            do_scroll_x=False
        )
        details_scroll.add_widget(details_layout)
        content_container.add_widget(details_scroll)

        # Add the content container to the scroll view
        scroll_view.add_widget(content_container)

        # Add the scroll view to the tab container
        container.add_widget(scroll_view)

    def on_tab_switch(self, instance, value):
        """Called when switching tabs."""
        if value.text == "Task Prints":
            self.drawings_container = self.task_prints_container
            self.drawing_preview = self.task_drawing_preview
            self.details_layout = self.task_details_layout
        else:
            self.drawings_container = self.position_prints_container
            self.drawing_preview = self.position_drawing_preview
            self.details_layout = self.position_details_layout

    def update_for_position(self, position_id):
        self.task_prints_container.clear_widgets()
        self.position_prints_container.clear_widgets()
        drawings = self.app.data_service.get_drawings_by_position(position_id)
        if not drawings:
            self.task_prints_container.add_widget(OneLineListItem(text="No drawings found"))
            self.position_prints_container.add_widget(OneLineListItem(text="No drawings found"))
            return

        # Separate drawings into task-related and position-related
        task_drawings = [d for d in drawings if d.drawing_task]
        position_drawings = [d for d in drawings if d.drawing_position]

        for drawing in task_drawings:
            item = OneLineListItem(
                text=drawing.drw_name or f"Drawing {drawing.id}",
                on_release=lambda *args, d=drawing: self.show_drawing_details(d)
            )
            item.drawing_id = drawing.id  # if needed later
            self.task_prints_container.add_widget(item)

        for drawing in position_drawings:
            item = OneLineListItem(
                text=drawing.drw_name or f"Drawing {drawing.id}",
                on_release=lambda *args, d=drawing: self.show_drawing_details(d)
            )
            item.drawing_id = drawing.id
            self.position_prints_container.add_widget(item)

    def on_drawing_selected(self, list_item):
        try:
            drawing_id = getattr(list_item, 'drawing_id', None)
            if not drawing_id:
                return
            drawing = self.app.data_service.get_drawing_details(drawing_id)
            if not drawing:
                return

            # Update drawing preview image if file path is valid
            if drawing.file_path and os.path.exists(drawing.file_path):
                self.drawing_preview.source = drawing.file_path
            else:
                self.drawing_preview.source = ''

            # Clear any existing details
            self.details_layout.clear_widgets()

            # Define only the fields you want to display:
            fields = [
                ("Equipment:", drawing.drw_equipment_name or ""),
                ("Drawing Number:", drawing.drw_number or ""),
                ("Drawing Name:", drawing.drw_name or "")
            ]

            # Add each field as a label/value pair to the details layout
            for label_text, value in fields:
                label = Label(
                    text=label_text,
                    halign='right',
                    valign='middle',
                    text_size=(None, dp(40)),
                    size_hint_y=None,
                    height=dp(40)
                )
                # Use a ScrollableLabel or any widget suitable for displaying long text
                value_label = ScrollableLabel(
                    text=str(value),
                    halign='left',
                    valign='middle'
                )
                self.details_layout.add_widget(label)
                self.details_layout.add_widget(value_label)

        except Exception as e:
            logger.error(f"Error selecting drawing: {e}")

    def update_for_task(self, task_id):
        """Update the Task Prints tab based on the selected task."""
        # Ensure we're using the details layout for the task tab
        self.details_layout = self.task_details_layout

        self.task_prints_container.clear_widgets()
        self.task_details_layout.clear_widgets()

        if not task_id:
            self.task_prints_container.add_widget(
                OneLineListItem(text="No task selected")
            )
            return

        try:
            # Query the drawings related to this task
            from modules.emtacdb.emtacdb_fts import DrawingTaskAssociation
            task_associations = (
                self.app.data_service.session.query(DrawingTaskAssociation)
                .filter(DrawingTaskAssociation.task_id == task_id)
                .all()
            )

            drawings = [assoc.drawing for assoc in task_associations if assoc.drawing]

            if not drawings:
                self.task_prints_container.add_widget(
                    OneLineListItem(text="No drawings linked to this task")
                )
                return

            for drawing in drawings:
                item = OneLineListItem(
                    text=drawing.drw_name or f"Drawing {drawing.id}",
                    on_release=lambda *args, d=drawing: self.show_drawing_details(d)
                )
                item.drawing_id = drawing.id
                self.task_prints_container.add_widget(item)

        except Exception as e:
            import traceback
            traceback.print_exc()
            logger.error(f"Error updating task drawings: {e}")
            self.task_prints_container.add_widget(
                OneLineListItem(text=f"Error: {str(e)}")
            )

    def show_drawing_details(self, drawing_obj):
        if not drawing_obj:
            return

        # Manually set the correct details layout based on active tab
        current_tab = self.tabs.current_tab.text
        if current_tab == "Task Prints":
            self.details_layout = self.task_details_layout
        elif current_tab == "Position Prints":
            self.details_layout = self.position_details_layout

        # Clear and populate the details layout
        self.details_layout.clear_widgets()

        fields = [
            ("Equipment:", drawing_obj.drw_equipment_name or ""),
            ("Drawing Number:", drawing_obj.drw_number or ""),
            ("Drawing Name:", drawing_obj.drw_name or "")
        ]
        for label_text, value in fields:
            label = Label(text=label_text, size_hint_y=None, height=dp(40))
            value_label = Label(text=value, size_hint_y=None, height=dp(40))
            self.details_layout.add_widget(label)
            self.details_layout.add_widget(value_label)




