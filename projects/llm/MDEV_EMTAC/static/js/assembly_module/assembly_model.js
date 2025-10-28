document.addEventListener('DOMContentLoaded', () => {
    const sidebarCollapse = document.getElementById('sidebarCollapse');
    const sidebar = document.querySelector('.sidebar');
    const mainContainer = document.querySelector('.main-container');
    const closeSidebarBtn = document.getElementById('closeSidebar');

    sidebarCollapse.addEventListener('click', () => {
        mainContainer.classList.toggle('collapsed');
    });

    closeSidebarBtn.addEventListener('click', () => {
        mainContainer.classList.add('collapsed');
    });

    // Comment Popup Toggle
    const openCommentBtn = document.getElementById('openComment');
    const commentPopup = document.getElementById('commentPopup');
    const submitCommentBtn = document.getElementById('submitComment');

    openCommentBtn.addEventListener('click', () => {
        commentPopup.style.display = 'block';
    });

    submitCommentBtn.addEventListener('click', () => {
        const comment = document.getElementById('commentText').value;
        // Handle the comment submission (e.g., send to server)
        console.log('Comment Submitted:', comment);
        commentPopup.style.display = 'none';
        document.getElementById('commentText').value = '';
    });

    // Close popup when clicking outside of it
    window.addEventListener('click', (event) => {
        if (event.target == commentPopup) {
            commentPopup.style.display = 'none';
        }
    });

    // Handle Form Submission
    const assemblyForm = document.getElementById('assemblyForm');
    assemblyForm.addEventListener('submit', function(event) {
        event.preventDefault(); // Prevent default form submission

        // Gather form data
        const assemblyName = document.getElementById('assemblyName').value;
        const assemblyView = document.getElementById('assemblyView').value;
        const subAssemblyName = document.getElementById('subAssemblyName').value;

        // Create data object
        const data = {
            name: assemblyName,
            assembly_view: assemblyView,
            subassembly_name: subAssemblyName
        };

        // Send data to server
        fetch('/submit-assembly', { // Update this if you have a URL prefix
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        })
        .then(response => response.json())
        .then(data => {
            console.log('Success:', data);
            alert('Subassembly submitted successfully!');
            assemblyForm.reset(); // Reset the form
        })
        .catch((error) => {
            console.error('Error:', error);
            alert('There was an error submitting the assembly.');
        });
    });
});
