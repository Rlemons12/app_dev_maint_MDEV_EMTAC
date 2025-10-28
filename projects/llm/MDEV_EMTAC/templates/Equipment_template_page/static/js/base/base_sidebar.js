// Sidebar toggle functionality
document.addEventListener('DOMContentLoaded', function() {
    const sidebar = document.getElementById('sidebar');
    const sidebarToggle = document.getElementById('sidebarToggle');
    const mainContainer = document.getElementById('main-container');

    if (sidebar && sidebarToggle && mainContainer) {
        // Check for saved state in localStorage
        const sidebarState = localStorage.getItem('sidebarState');

        // Apply saved state if it exists
        if (sidebarState === 'retracted') {
            sidebar.classList.add('retracted');
            mainContainer.classList.add('sidebar-retracted');
        }

        sidebarToggle.addEventListener('click', function() {
            sidebar.classList.toggle('retracted');
            mainContainer.classList.toggle('sidebar-retracted');

            // Save state to localStorage
            if (sidebar.classList.contains('retracted')) {
                localStorage.setItem('sidebarState', 'retracted');
            } else {
                localStorage.setItem('sidebarState', 'expanded');
            }
        });
    }
});

// Dropdown widget logic
document.addEventListener('DOMContentLoaded', function() {
    document.querySelectorAll('.dropdown-btn').forEach(btn => {
        btn.addEventListener('click', function() {
            const parent = this.parentElement;
            parent.classList.toggle('open');
        });
    });
});

// Filtering logic
document.addEventListener('DOMContentLoaded', function() {
    const sidebarFilter = document.getElementById('sidebarFilter');
    const sidebarLinks = document.getElementById('sidebarLinks');

    if (sidebarFilter && sidebarLinks) {
        sidebarFilter.addEventListener('input', function() {
            const val = this.value.toLowerCase();
            sidebarLinks.querySelectorAll('.sidebar-link').forEach(link => {
                link.style.display = link.textContent.toLowerCase().includes(val) ? '' : 'none';
            });
        });
    }
});

// Active link highlighting
document.addEventListener('DOMContentLoaded', function() {
    function highlightActiveSidebarLink() {
        const links = document.querySelectorAll('.sidebar-link[href^="#"]');
        let lastActive = null;
        for (let link of links) {
            const section = document.querySelector(link.getAttribute('href'));
            if (section && window.scrollY >= section.offsetTop - 80) {
                lastActive = link;
            }
            link.classList.remove('active');
        }
        if (lastActive) lastActive.classList.add('active');
    }

    window.addEventListener('scroll', highlightActiveSidebarLink);
    highlightActiveSidebarLink(); // initial run
});