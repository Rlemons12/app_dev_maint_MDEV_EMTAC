<div class="right-side">
    {% if answer %}
        <div class="answer-section">
            <!-- Button to toggle full response -->
            <button id="toggle-full-response">Show Full Response</button>

            <div id="answer-content">
                {{ answer|safe }}  <!-- This ensures that the HTML content is rendered safely -->
            </div>

            <h2>Answer:</h2>

            <!-- Rating -->
            <form method="POST">
                <input type="hidden" name="user_id" value="{{ session['last_user_id'] or '' }}">
                <input type="hidden" name="user_input" value="{{ session['last_question'] or '' }}">
                
                <label for="rating">Rate the Answer:</label>
                <select id="rating" name="rating">
                    <option value="1">1</option>
                    <option value="2">2</option>
                    <option value="3">3</option>
                    <option value="4">4</option>
                    <option value="5">5</option>
                </select>
                <input type="submit" value="Rate">
            </form>
        </div>
    {% endif %}

    <!-- Display Thumbnails -->
    <h3>Related Thumbnails:</h3>
    {% for path in thumbnail_paths %}
        <a href="{{ path }}" target="_blank">
            <img src="{{ path }}" alt="Thumbnail" width="100">
        </a>
    {% endfor %}
</div>
