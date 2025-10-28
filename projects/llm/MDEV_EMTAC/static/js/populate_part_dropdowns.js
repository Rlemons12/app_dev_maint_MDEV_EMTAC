function populatePartDropdowns() {
    $.ajax({
        url: '/get_part_form_data',
        type: 'GET',
        success: function(data) {
            // Populate the model dropdown
            var modelDropdown = $('#part_modelDropdown');
            modelDropdown.empty();
            $.each(data['models'], function(index, model) {
                modelDropdown.append('<option value="' + model.id + '">' + model.name + '</option>');
            });
        },
        error: function(xhr, status, error) {
            console.error('Error fetching part form data:', error);
        }
    });
}

$(document).ready(function() {
    populatePartDropdowns();
});
