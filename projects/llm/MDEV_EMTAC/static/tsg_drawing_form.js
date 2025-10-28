$(document).ready(function() {
    $('#tsg_drawingSearchDropdown').select2({
        ajax: {
            url: '/search_drawing_by_number',
            dataType: 'json',
            delay: 250,
            data: function(params) {
                return {
                    query: params.term
                };
            },
            processResults: function(data) {
                return {
                    results: data.map(function(drawing) {
                        return {
                            id: drawing.id,
                            text: drawing.number + " - " + drawing.name
                        };
                    })
                };
            },
            cache: true
        },
        minimumInputLength: 1,
        placeholder: 'Search for drawing number',
        allowClear: true
    });
});
