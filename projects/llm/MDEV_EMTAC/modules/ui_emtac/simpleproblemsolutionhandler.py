from kivy.metrics import dp
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivymd.uix.button import MDRaisedButton


class SimpleProblemSolutionHandler:
    """
    A utility class that can be used to test problem-solution interaction
    in a simplified way, separate from the main UI flow.
    """

    def test_it(self, problem_container, solution_container, app):
        """Test the problem-solution flow directly"""
        problem_container.clear_widgets()
        solution_container.clear_widgets()

        # Add test header
        problem_container.add_widget(
            Label(text="DIRECT TEST SETUP", size_hint_y=None, height=dp(30), color=(1, 1, 0, 1))
        )

        # Add a standard Kivy button first to test basic event handling
        basic_btn = Button(
            text="BASIC BUTTON (Click Me)",
            size_hint_y=None,
            height=dp(50),
            background_color=(1, 0, 0, 1)
        )
        basic_btn.bind(on_press=lambda x: self._on_basic_button(solution_container))
        problem_container.add_widget(basic_btn)

        # Add spacer
        problem_container.add_widget(Widget(size_hint_y=None, height=dp(20)))

        # Now try direct problem handling
        # Fetch problem from database
        from modules.emtacdb.emtacdb_fts import Problem, Solution
        problem = app.data_service.session.query(Problem).filter(Problem.id == 1).first()

        if problem:
            # Create problem button with direct handling
            prob_btn = MDRaisedButton(
                text=f"DIRECT PROBLEM: {problem.name}",
                size_hint_x=1.0,
                size_hint_y=None,
                height=dp(60),
                md_bg_color=(0.9, 0.2, 0.2, 1)
            )
            # Directly use a callback with problem ID hardcoded
            prob_btn.bind(on_release=lambda x: self._handle_problem_click(problem.id, solution_container, app))
            problem_container.add_widget(prob_btn)

            # Log it
            print(f"Added direct test button for problem: {problem.name}")
        else:
            problem_container.add_widget(
                Label(text="ERROR: Could not find Problem ID 1",
                      size_hint_y=None, height=dp(40), color=(1, 0, 0, 1))
            )

    def _on_basic_button(self, solution_container):
        """Handle basic button click to test event system"""
        print("BASIC BUTTON CLICKED - Testing event system")

        # Clear and update solution container
        solution_container.clear_widgets()
        solution_container.add_widget(
            Label(text="BASIC BUTTON CLICKED!",
                  size_hint_y=None, height=dp(40), color=(0, 1, 0, 1))
        )

    def _handle_problem_click(self, problem_id, solution_container, app):
        """Handle problem button click with hardcoded problem ID"""
        print(f"DIRECT HANDLER: Problem {problem_id} clicked")

        # Clear solution container
        solution_container.clear_widgets()
        solution_container.add_widget(
            Label(text=f"Loading solutions for problem {problem_id}...",
                  size_hint_y=None, height=dp(40), color=(0, 1, 1, 1))
        )

        # Fetch solutions directly
        from modules.emtacdb.emtacdb_fts import Solution
        solutions = app.data_service.session.query(Solution).filter(Solution.problem_id == problem_id).all()

        print(f"DIRECT HANDLER: Found {len(solutions)} solutions")

        if not solutions:
            solution_container.add_widget(
                Label(text="No solutions found for this problem",
                      size_hint_y=None, height=dp(40), color=(1, 0, 0, 1))
            )
            return

        # Add a label showing how many solutions we found
        solution_container.add_widget(
            Label(text=f"Found {len(solutions)} solutions",
                  size_hint_y=None, height=dp(40), color=(0, 1, 0, 1))
        )

        # Add each solution as a basic button
        for solution in solutions:
            # Use basic Button for simplicity
            sol_btn = Button(
                text=f"SOLUTION: {solution.name}",
                size_hint_y=None,
                height=dp(60),
                background_color=(0, 0.7, 1, 1)
            )
            sol_btn.solution_id = solution.id
            sol_btn.bind(on_press=lambda x: self._on_solution_clicked(x, solution_container))
            solution_container.add_widget(sol_btn)

            # Add some spacing
            solution_container.add_widget(Widget(size_hint_y=None, height=dp(10)))

    def _on_solution_clicked(self, button, solution_container):
        """Handle solution button click"""
        solution_id = getattr(button, 'solution_id', None)
        if solution_id:
            print(f"Solution button pressed: {solution_id}")

            # Add a success indicator
            solution_container.add_widget(
                Label(text=f"You clicked solution ID: {solution_id}!",
                      color=(0, 1, 0, 1), size_hint_y=None, height=dp(30))
            )