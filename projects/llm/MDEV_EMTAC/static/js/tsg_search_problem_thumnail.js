function displayThumbnails(thumbnails) {
    const thumbnailsSection = document.getElementById("thumbnails-section");
    thumbnailsSection.innerHTML = "";

    thumbnails.forEach(thumbnail => {
        const anchor = document.createElement("a");
        anchor.href = thumbnail.src;
        anchor.setAttribute("data-full-src", thumbnail.src);

        const img = document.createElement("img");
        img.src = thumbnail.thumbnail_src;
        img.alt = thumbnail.title;
        img.title = thumbnail.title;
        img.classList.add("thumbnail");

        anchor.appendChild(img);
        thumbnailsSection.appendChild(anchor);
    });
}

$(document).on('submit', 'form[action="/tsg_search_images"]', function(event) {
    event.preventDefault();

    var formData = $(this).serialize();

    $.ajax({
        url: '/tsg_search_images',
        method: 'GET',
        data: formData,
        success: function(response) {
            var thumbnails = response.thumbnails;
            displayThumbnails(thumbnails);
        },
        error: function(xhr, status, error) {
            console.error('Error:', error);
        }
    });
});
