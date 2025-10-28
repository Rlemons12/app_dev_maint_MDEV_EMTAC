$(document).ready(function() {
    $('#tsg_partSearchDropdown').select2({
        ajax: {
            url: '/search_parts',
            dataType: 'json',
            delay: 250,
            data: function(params) {
                return {
                    query: params.term
                };
            },
            processResults: function(data) {
                return {
                    results: data.map(function(part) {
                        return {
                            id: part.id,
                            text: part.part_number + " - " + part.name
                        };
                    })
                };
            },
            cache: true
        },
        minimumInputLength: 1,
        placeholder: 'Search for part number',
        allowClear: true
    });
});
