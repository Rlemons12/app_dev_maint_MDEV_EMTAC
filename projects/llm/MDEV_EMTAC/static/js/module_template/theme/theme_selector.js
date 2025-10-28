document.getElementById('theme-select').addEventListener('change', function() {
    var themeFile = this.value;
    document.getElementById('theme-style').href = themeFile ? 'static/css/themes/' + themeFile : '';
    // Optional: Save selection in localStorage to remember on next visit
    localStorage.setItem('selectedTheme', themeFile);
});

// On page load, restore previous theme if set
window.addEventListener('DOMContentLoaded', function() {
    var savedTheme = localStorage.getItem('selectedTheme');
    if (savedTheme) {
        document.getElementById('theme-style').href = 'static/css/themes/' + savedTheme;
        document.getElementById('theme-select').value = savedTheme;
    }
});
