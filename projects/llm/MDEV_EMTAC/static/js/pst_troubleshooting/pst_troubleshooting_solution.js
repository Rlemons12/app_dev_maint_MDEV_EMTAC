// pst_troubleshooting_solution.js

document.addEventListener('DOMContentLoaded', () => {
    'use strict';

    // Define backend endpoint URLs
    const GET_SOLUTIONS_URL = '/pst_troubleshooting_solution/get_solutions/';
    const ADD_SOLUTION_URL = '/pst_troubleshooting_solution/add_solution/';
    const REMOVE_SOLUTIONS_URL = '/pst_troubleshooting_solution/remove_solutions/';
    const GET_TASKS_URL_BASE = '/pst_troubleshooting_solution/get_tasks/';
    const ADD_TASK_URL = '/pst_troubleshooting_solution/add_task/';

    let currentProblemId = null;
    let currentSolutionId = null;
    let solutionsToRemove = [];

    function showAlert(message, category) {
        const alertContainer = document.getElementById('alertContainer');
        if (alertContainer) {
            const alertDiv = document.createElement('div');
            alertDiv.className = `alert alert-${category} alert-dismissible fade show`;
            alertDiv.innerHTML = `${message}
                <button type="button" class="btn-close" data-bs-dismiss="alert" aria-label="Close"></button>`;
            alertContainer.appendChild(alertDiv);
            setTimeout(() => alertDiv.remove(), 5000);
        }
    }

    function fetchSolutions(problemId) {
        currentProblemId = problemId;
        showAlert('Loading solutions...', 'info');

        fetch(`${GET_SOLUTIONS_URL}${encodeURIComponent(problemId)}`)
            .then(response => {
                if (!response.ok) {
                    return response.json().then(errorData => {
                        throw new Error(errorData.error || 'Failed to fetch solutions.');
                    });
                }
                return response.json();
            })
            .then(response => {
                populateSolutionsDropdown(response.solutions);
                updateProblemName(response.problem_name);
                activateTab('solution-tab');
            })
            .catch(error => showAlert('Error fetching solutions: ' + error.message, 'danger'));
    }

    function populateSolutionsDropdown(solutions) {
        const solutionsDropdown = document.getElementById('existing_solutions');
        solutionsDropdown.innerHTML = solutions.map(solution =>
            `<option value="${solution.id}">${solution.name} - ${solution.description || 'No description provided.'}</option>`
        ).join('');

        // Reset currentSolutionId when solutions are repopulated
        currentSolutionId = null;

        // Remove existing event listener to prevent duplication
        const newSolutionsDropdown = solutionsDropdown.cloneNode(true);
        solutionsDropdown.parentNode.replaceChild(newSolutionsDropdown, solutionsDropdown);

        // Attach single-click event to update selection
        newSolutionsDropdown.addEventListener('click', (event) => {
            const selectedOption = newSolutionsDropdown.value;
            if (selectedOption) {
                currentSolutionId = selectedOption;
                showAlert(`Solution "${event.target.selectedOptions[0].text}" selected.`, 'info');
            } else {
                currentSolutionId = null;
            }
        });

        // Attach double-click event to trigger an action
        newSolutionsDropdown.addEventListener('dblclick', (event) => {
            const selectedOption = newSolutionsDropdown.value;
            if (selectedOption) {
                currentSolutionId = selectedOption;
                fetchTasksForSolution(selectedOption); // Fetch tasks related to the double-clicked solution
                showAlert(`Action triggered for solution "${event.target.selectedOptions[0].text}".`, 'success');
            }
        });
    }



    function updateProblemName(problemName) {
        const header = document.getElementById('selected-problem-name');
        if (header) header.textContent = `Problem Solutions for: ${problemName}`;
    }

    function activateTab(tabId) {
        const tabLink = document.querySelector(`a[href="#${tabId.replace('-tab', '')}"]`);
        if (tabLink) {
            const tab = new bootstrap.Tab(tabLink);
            tab.show();
        }
    }

    function addNewSolution(problemId, solutionName, solutionDescription) {
        fetch(ADD_SOLUTION_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ problem_id: problemId, name: solutionName, description: solutionDescription })
        })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(errorData => {
                        throw new Error(errorData.error || 'Failed to add solution.');
                    });
                }
                return response.json();
            })
            .then(() => {
                showAlert('Solution added successfully.', 'success');
                fetchSolutions(problemId);
                clearInputFields('new_solution_name', 'new_solution_description');
            })
            .catch(error => showAlert('Error adding solution: ' + error.message, 'danger'));
    }

    function removeSolutions(problemId, solutionIds) {
        fetch(REMOVE_SOLUTIONS_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ problem_id: problemId, solution_ids: solutionIds })
        })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(errorData => {
                        throw new Error(errorData.error || 'Failed to remove solutions.');
                    });
                }
                return response.json();
            })
            .then(() => {
                showAlert('Selected solutions removed successfully.', 'success');
                fetchSolutions(problemId);
            })
            .catch(error => showAlert('Error removing solutions: ' + error.message, 'danger'));
    }

    function fetchTasksForSolution(solutionId) {
        if (!solutionId) {
            console.warn("Invalid solutionId passed to fetchTasksForSolution.");
            return;
        }

        fetch(`${GET_TASKS_URL_BASE}${encodeURIComponent(solutionId)}`)
            .then(response => {
                if (!response.ok) {
                    return response.json().then(errorData => {
                        throw new Error(errorData.error || `HTTP error! Status: ${response.status}`);
                    });
                }
                return response.json();
            })
            .then(data => {
                if (data && data.tasks) {
                    populateTasksDropdown(data.tasks);
                    activateTab('task-tab');
                } else {
                    showAlert('No tasks found for this solution.', 'info');
                    populateTasksDropdown([]); // Clear dropdown if no tasks
                }
            })
            .catch(error => {
                console.error('Error fetching tasks:', error);
                showAlert(`Error fetching tasks: ${error.message || 'Unknown error occurred'}`, 'danger');
                activateTab('task-tab');
            });
    }

    // Populate tasks in a dropdown list instead of a list-group
    function populateTasksDropdown(tasks) {
        const tasksDropdown = document.getElementById('existing_tasks');
        tasksDropdown.innerHTML = tasks.map(task =>
            `<option value="${task.id}">${task.name} - ${task.description || 'No description provided.'}</option>`
        ).join('');

        // Remove existing event listener to prevent duplication
        const newTasksDropdown = tasksDropdown.cloneNode(true);
        tasksDropdown.parentNode.replaceChild(newTasksDropdown, tasksDropdown);

        // Attach new event listener
        newTasksDropdown.addEventListener('change', (event) => {
            const selectedTaskId = newTasksDropdown.value;
            if (selectedTaskId) openTaskDetails(selectedTaskId);
        });
    }

    function clearTasksDropdown() {
        const tasksDropdown = document.getElementById('existing_tasks');
        tasksDropdown.innerHTML = '';
    }

    function openTaskDetails(taskId) {
        fetch(`/pst_troubleshooting_task/get_task_details/${encodeURIComponent(taskId)}`)
            .then(response => {
                if (!response.ok) {
                    return response.json().then(errorData => {
                        throw new Error(errorData.error || 'Failed to load task details.');
                    });
                }
                return response.json();
            })
            .then(data => {
                if (data && data.task) {
                    populateEditTaskForm(data.task);
                    activateTab('edit-task-tab');
                } else {
                    clearEditTaskForm();
                }
            })
            .catch(error => showAlert('Error loading task details: ' + error.message, 'danger'));
    }

    function populateEditTaskForm(task) {
        document.getElementById('pst_task_edit_task_name').value = task.name || '';
        document.getElementById('pst_task_edit_task_description').value = task.description || '';
        document.getElementById('edit_task_id').value = task.id;
    }

    function clearEditTaskForm() {
        document.getElementById('pst_task_edit_task_name').value = '';
        document.getElementById('pst_task_edit_task_description').value = '';
        document.getElementById('edit_task_id').value = '';
    }

    function clearInputFields(...fieldIds) {
        fieldIds.forEach(id => {
            const field = document.getElementById(id);
            if (field) field.value = '';
        });
    }

    function initializeEventListeners() {
        const addSolutionBtn = document.getElementById('addSolutionBtn');
        if (addSolutionBtn) {
            addSolutionBtn.addEventListener('click', () => {
                const name = document.getElementById('new_solution_name').value.trim();
                const description = document.getElementById('new_solution_description')?.value.trim();
                if (name && currentProblemId) {
                    addNewSolution(currentProblemId, name, description);
                } else {
                    showAlert('Solution name cannot be empty or no problem selected.', 'warning');
                }
            });
        }

        const removeSolutionsBtn = document.getElementById('removeSolutionsBtn');
        if (removeSolutionsBtn) {
            removeSolutionsBtn.addEventListener('click', () => {
                const selectedOptions = Array.from(document.getElementById('existing_solutions').selectedOptions);
                if (selectedOptions.length && currentProblemId) {
                    solutionsToRemove = selectedOptions.map(option => option.value);
                    const confirmModal = new bootstrap.Modal(document.getElementById('confirmModal'));
                    confirmModal.show();
                } else {
                    showAlert('Please select at least one solution to remove.', 'warning');
                }
            });
        }

        const confirmRemoveBtn = document.getElementById('confirmRemoveBtn');
        if (confirmRemoveBtn) {
            confirmRemoveBtn.addEventListener('click', () => {
                const confirmModal = bootstrap.Modal.getInstance(document.getElementById('confirmModal'));
                confirmModal.hide();
                if (solutionsToRemove.length && currentProblemId) {
                    removeSolutions(currentProblemId, solutionsToRemove);
                }
            });
        }

        const addTaskBtn = document.getElementById('addTaskBtn');
        if (addTaskBtn) {
            addTaskBtn.addEventListener('click', () => {
                const name = document.getElementById('new_task_name').value.trim();
                const description = document.getElementById('new_task_description').value.trim();
                if (name && currentSolutionId) {
                    addNewTask(currentSolutionId, name, description);
                } else {
                    showAlert('Task name cannot be empty or no solution selected.', 'warning');
                }
            });
        }
    }

    // Define the addNewTask function since it's referenced but not defined
    function addNewTask(solutionId, taskName, taskDescription) {
        fetch(ADD_TASK_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ solution_id: solutionId, name: taskName, description: taskDescription })
        })
            .then(response => {
                if (!response.ok) {
                    return response.json().then(errorData => {
                        throw new Error(errorData.error || 'Failed to add task.');
                    });
                }
                return response.json();
            })
            .then(() => {
                showAlert('Task added successfully.', 'success');
                fetchTasksForSolution(solutionId);
                clearInputFields('new_task_name', 'new_task_description');
            })
            .catch(error => showAlert('Error adding task: ' + error.message, 'danger'));
    }

    initializeEventListeners();
});
