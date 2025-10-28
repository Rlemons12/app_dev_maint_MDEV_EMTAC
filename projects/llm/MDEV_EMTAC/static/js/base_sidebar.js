document.addEventListener("DOMContentLoaded", function() {
    const sidebarToggleBtn = document.getElementById("sidebarCollapse");
    const closeSidebarBtn = document.getElementById("closeSidebar");
    const mainContainer = document.querySelector(".main-container");
    const toggleVoiceBtn = document.getElementById("toggle-voice");
    const toggleTextToSpeechBtn = document.getElementById("toggle-text-to-speech");

    // Toggle Sidebar functionality for both open and close
    const toggleSidebar = () => {
        if (mainContainer) {
            mainContainer.classList.toggle("collapsed");
        }
    };

    // Sidebar collapse button (visible when sidebar is closed)
    if (sidebarToggleBtn) {
        sidebarToggleBtn.onclick = toggleSidebar;
    }

    // Close sidebar button (visible when sidebar is open)
    if (closeSidebarBtn) {
        closeSidebarBtn.onclick = toggleSidebar;
    }

    // Toggle buttons for voice and text-to-speech
    if (toggleVoiceBtn) {
        toggleVoiceBtn.onclick = function() {
            this.classList.toggle("active");
        };
    }

    if (toggleTextToSpeechBtn) {
        toggleTextToSpeechBtn.onclick = function() {
            this.classList.toggle("active");
        };
    }
});
