// static/js/tool/tool_initializer.js

document.addEventListener('DOMContentLoaded', () => {
    // Elements for dropdowns

    // Add event listeners to the accordion
    setupAccordionEvents();
});

// Function to handle Bootstrap Accordion events
function setupAccordionEvents() {
    const accordion = document.getElementById('toolFormAccordion');

    if (!accordion) {
        console.warn('Accordion element not found.');
        return;
    }

    // Triggered when an accordion section expands
    accordion.addEventListener('shown.bs.collapse', (event) => {
        console.log(`Section Expanded: ${event.target.id}`);
    });

    // Triggered when an accordion section collapses
    accordion.addEventListener('hidden.bs.collapse', (event) => {
        console.log(`Section Collapsed: ${event.target.id}`);
    });
}
