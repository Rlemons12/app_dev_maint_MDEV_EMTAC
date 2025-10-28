// chatbot.js
document.getElementById('submit-question').addEventListener('click', function() {
    var userId = document.getElementById('user_id').value;
    var area = document.getElementById('area').value;
    var userInput = document.getElementById('user_input').value;

    // AJAX request to Flask server
    fetch('/chatbot/ask', {  // Corrected URL
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ userId: userId, area: area, question: userInput })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('answer').innerText = data.answer;
    })
    .catch(error => console.error('Error:', error));
});
