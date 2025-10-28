// pst_troubleshooting_task.js

document.addEventListener('DOMContentLoaded', () => {
    'use strict';

    const GET_TASKS_URL = '/pst_troubleshooting_solution/get_tasks/';
    const ADD_TASK_URL = '/pst_troubleshooting_solution/add_task/';
    const GET_TASK_DETAILS_URL = '/pst_troubleshooting_solution/get_task_details/';  // Ensure you have this route in your Flask app if needed.
    let currentSolutionId = null;

    /**
     * Displays an alert message to the user.
     * @param {string} message - The message to display.
     * @param {string} category - The Bootstrap alert category (e.g., 'success', 'warning', 'danger').
     */
    function showAlert(message, category) {
        const alertContainer = document.getElementById('alertContainer');
        if (alertContainer) {
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${category} alert-dismissible fade show`;
            alertDiv.role = 'alert';
            alertDiv.innerHTML = `
                ${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>
            `;
            alertContainer.appendChild(alertDiv);

            setTimeout(() => {
                alertDiv.remove();
            }, 5000);
        }
    }

    /**
     * Fetches tasks for a selected solution and populates the Task tab.
     * @param {number} solutionId - The ID of the selected solution.
     */
    function fetchTasksForSolution(solutionId) {
        currentSolutionId = solutionId;

        fetch(`${GET_TASKS_URL}${solutionId}`)
            .then(response => response.json())
            .then(data => {
                if (data && data.tasks) {
                    populateTasksList(data.tasks);
                } else {
                    showAlert('No tasks found for this solution. You can add new tasks below.', 'info');
                    populateTasksList([]); // Clear tasks if no data
                }
            })
            .catch(error => {
                showAlert('Error fetching tasks: ' + error.message, 'danger');
                console.error('Error fetching tasks:', error);
            });
    }

    /**
     * Populates the Task tab list with fetched tasks.
     * @param {Array} tasks - An array of task objects.
     */
    function populateTasksList(tasks) {
        const tasksList = document.getElementById('tasks_list');
        tasksList.innerHTML = ''; // Clear existing tasks

        tasks.forEach(task => {
            const listItem = document.createElement('li');
            listItem.className = 'list-group-item';
            listItem.textContent = `${task.name} - ${task.description || 'No description provided.'}`;
            listItem.dataset.taskId = task.id; // Store task ID for easy reference
            listItem.addEventListener('dblclick', () => openTaskDetails(task.id)); // Double-click event
            tasksList.appendChild(listItem);
        });
    }

    /**
     * Opens task details when a task is double-clicked.
     * Switches to the "Edit Task" tab or loads a modal with task details.
     * @param {number} taskId - The ID of the selected task.
     */
    function openTaskDetails(taskId) {
        fetch(`${GET_TASK_DETAILS_URL}${taskId}`)
            .then(response => response.json())
            .then(data => {
                if (data && data.task) {
                    populateEditTaskForm(data.task); // Populate form with task details
                } else {
                    clearEditTaskForm(); // Clear the form if no data is found
                }
                activateTab('edit-task-tab'); // Switch to Edit Task tab in both cases
            })
            .catch(error => {
                showAlert('Error loading task details: ' + error.message, 'danger');
                console.error('Error loading task details:', error);
            });
    }

    /**
     * Populates the edit form with task details.
     * @param {Object} task - The task object containing details.
     */
    function populateEditTaskForm(task) {
        document.getElementById('pst_task_edit_task_name').value = task.name || '';
        document.getElementById('pst_task_edit_task_description').value = task.description || '';
        document.getElementById('edit_task_id').value = task.id; // Hidden input for task ID
        // Populate additional fields if necessary
    }

    /**
     * Clears the edit form fields when no task data is available.
     */
    function clearEditTaskForm() {
        document.getElementById('pst_task_edit_task_name').value = '';
        document.getElementById('pst_task_edit_task_description').value = '';
        document.getElementById('edit_task_id').value = ''; // Clear hidden input for task ID
        // Clear additional fields if necessary
    }

    /**
     * Activates a specific tab.
     * @param {string} tabId - The ID of the tab to activate.
     */
    function activateTab(tabId) {
        const tabLink = document.getElementById(tabId);
        if (tabLink) {
            new bootstrap.Tab(tabLink).show();
        }
    }

    // Initialize event listeners and load initial data
    fetchTasksForSolution(currentSolutionId); // Fetch tasks when a solution is selected
});
