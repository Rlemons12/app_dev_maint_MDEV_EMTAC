<script>
    Dropzone.autoDiscover = false; // Disable auto-discover to manually initialize Dropzone

    var myDropzone = new Dropzone("#myDropzone", {
    url: "/documents/add_document", // Update this to your file upload route
    paramName: "files",
    acceptedFiles: ".pdf,.txt,.csv,.zip",
    addRemoveLinks: true,
    autoProcessQueue: false,
});

    // Handle the form submission manually
    myDropzone.on("sending", function (file, xhr, formData) {
    var title = document.querySelector("input[name='title']").value;
    var area = document.querySelector("input[name='area']").value;
    var equipment_group = document.querySelector("input[name='equipment_group']").value;
    var model = document.querySelector("input[name='model']").value;
    var asset_number = document.querySelector("input[name='asset_number']").value;
    var location = document.querySelector("input[name='location']").value; // Get location value

    formData.append("title", title);
    formData.append("area", area);
    formData.append("equipment_group", equipment_group);
    formData.append("model", model);
    formData.append("asset_number", asset_number);
    formData.append("location", location); // Append location to FormData
});

    document.querySelector("form").addEventListener("submit", function (e) {
    e.preventDefault(); // Prevent the default form submission

    if (myDropzone.getQueuedFiles().length > 0) {
    // If there are files in the Dropzone queue, process them
    myDropzone.processQueue();
} else {
    // If no files in Dropzone, proceed with the default form submission
    this.submit();
}
});
</script>