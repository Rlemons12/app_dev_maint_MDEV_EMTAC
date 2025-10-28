document.addEventListener("DOMContentLoaded", function() {
    // Get DOM elements
    const sidebarToggleBtn = document.getElementById("sidebarCollapse");
    const closeSidebarBtn = document.getElementById("closeSidebar");
    const mainContainer = document.querySelector(".main-container");
    const sidebar = document.getElementById("mainSidebar");
    const toggleVoiceBtn = document.getElementById("toggle-voice");
    const toggleTextToSpeechBtn = document.getElementById("toggle-text-to-speech");
    const content = document.querySelector(".content");

    // Toggle the sidebar collapse state
    const toggleSidebar = () => {
        if (mainContainer) {
            mainContainer.classList.toggle("collapsed");

            // For mobile view, toggle the active class on sidebar
            if (window.innerWidth <= 768 && sidebar) {
                sidebar.classList.toggle("active");
            } else if (sidebar) {
                // For desktop view
                sidebar.classList.toggle("collapsed");

                if (content) {
                    content.classList.toggle("sidebar-collapsed");
                }
            }

            // Update aria-expanded attribute for accessibility
            if (sidebarToggleBtn) {
                const isCollapsed = mainContainer.classList.contains("collapsed");
                sidebarToggleBtn.setAttribute("aria-expanded", !isCollapsed);
            }
        }
    };

    // Sidebar collapse button
    if (sidebarToggleBtn) {
        sidebarToggleBtn.onclick = toggleSidebar;
    }

    // Close sidebar button (if one exists in your sidebar template)
    if (closeSidebarBtn) {
        closeSidebarBtn.onclick = toggleSidebar;
    }

    // Close sidebar when clicking outside on mobile
    document.addEventListener('click', function(event) {
        if (window.innerWidth <= 768 &&
            sidebar &&
            mainContainer &&
            mainContainer.classList.contains("collapsed") &&
            !sidebar.contains(event.target) &&
            event.target !== sidebarToggleBtn) {
            mainContainer.classList.remove("collapsed");
            sidebar.classList.remove("active");
        }
    });

    // Toggle Voice button
    if (toggleVoiceBtn) {
        toggleVoiceBtn.onclick = function() {
            this.classList.toggle("active");
            console.log("Voice recognition toggled");
            // Actual implementation would be added here
        };
    }

    // Toggle Text-to-Speech button
    if (toggleTextToSpeechBtn) {
        toggleTextToSpeechBtn.onclick = function() {
            this.classList.toggle("active");
            console.log("Text-to-speech toggled");
            // Actual implementation would be added here
        };
    }

    // Function to show specific forms - make it globally available
    window.showForm = function(formId) {
        // Hide all forms first
        const forms = document.querySelectorAll('.form-container');
        forms.forEach(form => {
            form.style.display = 'none';
        });

        // Show results container for search related forms
        const resultsContainer = document.getElementById('results-container');
        if (resultsContainer) {
            if (formId.includes('search')) {
                resultsContainer.style.display = 'block';
            } else {
                resultsContainer.style.display = 'none';
            }
        }

        // Show the selected form
        const selectedForm = document.getElementById(formId);
        if (selectedForm) {
            selectedForm.style.display = 'block';

            // Scroll to form on mobile
            if (window.innerWidth <= 768) {
                selectedForm.scrollIntoView({ behavior: 'smooth' });

                // Close sidebar on mobile after selection
                if (sidebar && mainContainer && mainContainer.classList.contains("collapsed")) {
                    mainContainer.classList.remove("collapsed");
                    sidebar.classList.remove("active");
                }
            }
        }
    };

    // Initialize voice selection dropdown if available
    initializeVoiceSelection();
});

// Function to initialize voice selection dropdown
function initializeVoiceSelection() {
    const voiceSelection = document.getElementById('voice-selection');
    if (voiceSelection && window.speechSynthesis) {
        // Get available voices and populate the dropdown
        function populateVoiceList() {
            const voices = window.speechSynthesis.getVoices();

            // Clear existing options
            voiceSelection.innerHTML = '';

            // Add a default option
            const defaultOption = document.createElement('option');
            defaultOption.textContent = 'Select Voice';
            voiceSelection.appendChild(defaultOption);

            // Add all available voices
            voices.forEach(voice => {
                const option = document.createElement('option');
                option.textContent = `${voice.name} (${voice.lang})`;
                option.setAttribute('data-lang', voice.lang);
                option.setAttribute('data-name', voice.name);
                voiceSelection.appendChild(option);
            });

            // Select previously used voice if any
            const savedVoice = localStorage.getItem('selectedVoice');
            if (savedVoice) {
                for (let i = 0; i < voiceSelection.options.length; i++) {
                    if (voiceSelection.options[i].getAttribute('data-name') === savedVoice) {
                        voiceSelection.selectedIndex = i;
                        break;
                    }
                }
            }
        }

        document.addEventListener('DOMContentLoaded', function() {
        const themeSelect = document.getElementById('theme-select');
        const themeStyle = document.getElementById('theme-style');

        // On theme selection change, set the theme CSS
        themeSelect.addEventListener('change', function() {
            const themeFile = this.value;
            themeStyle.href = themeFile
                ? "{{ url_for('static', filename='css/module_template/themes/') }}" + themeFile
                : "";
            localStorage.setItem('selectedTheme', themeFile);
        });

        // On load, restore the theme if previously selected
        const savedTheme = localStorage.getItem('selectedTheme');
        if (savedTheme) {
            themeStyle.href = "{{ url_for('static', filename='css/module_template/themes/') }}" + savedTheme;
            themeSelect.value = savedTheme;
        }
    });


        // Initial population
        populateVoiceList();

        // Update when voices change (happens asynchronously in some browsers)
        if (window.speechSynthesis.onvoiceschanged !== undefined) {
            window.speechSynthesis.onvoiceschanged = populateVoiceList;
        }

        // Handle voice selection change
        voiceSelection.addEventListener('change', function() {
            if (this.selectedIndex > 0) {
                const selectedOption = this.options[this.selectedIndex];
                const voiceName = selectedOption.getAttribute('data-name');
                console.log(`Selected voice: ${voiceName}`);
                // Store the selected voice for later use
                localStorage.setItem('selectedVoice', voiceName);
            }
        });
    }
}
