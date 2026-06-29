// upload_document.js
// Handles upload document dropdowns, AJAX upload, visible processing state,
// and upload result rendering.

function escapeUploadDocumentHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function ensureUploadDocumentProcessingStyles() {
    if (document.getElementById("emtac-upload-processing-styles")) {
        return;
    }

    var style = document.createElement("style");
    style.id = "emtac-upload-processing-styles";

    style.textContent = `
        .emtac-upload-processing-box {
            display: flex;
            gap: 14px;
            align-items: flex-start;
            padding: 12px;
            border: 1px solid rgba(57, 255, 20, 0.35);
            border-radius: 8px;
            background: rgba(0, 0, 0, 0.18);
            color: #eee;
        }

        .emtac-upload-spinner {
            width: 30px;
            height: 30px;
            min-width: 30px;
            border: 4px solid rgba(255, 255, 255, 0.25);
            border-top-color: #39FF14;
            border-radius: 50%;
            animation: emtacUploadSpin 0.9s linear infinite;
            margin-top: 4px;
        }

        .emtac-upload-processing-title {
            font-weight: 700;
            margin-bottom: 4px;
        }

        .emtac-upload-processing-note {
            color: #facc15;
            font-size: 13px;
            margin-top: 6px;
        }

        .emtac-upload-processing-step {
            color: #bbb;
            font-size: 13px;
            margin-top: 4px;
        }

        .emtac-upload-result-success {
            color: #39FF14;
            font-weight: 700;
        }

        .emtac-upload-result-error {
            color: #ff6b6b;
            font-weight: 700;
        }

        @keyframes emtacUploadSpin {
            to {
                transform: rotate(360deg);
            }
        }
    `;

    document.head.appendChild(style);
}

function showUploadDocumentResults() {
    var resultsPanel = $("#results");
    var documentsList = $("#documents-list");

    if (resultsPanel.length) {
        resultsPanel.show();
    }

    if (documentsList.length) {
        documentsList.show();
    }
}

function getSelectedUploadDocumentFileNames(formElement) {
    var selectedFileNames = [];
    var fileInputs = $(formElement).find("input[type='file']");

    fileInputs.each(function () {
        if (!this.files) {
            return;
        }

        for (var i = 0; i < this.files.length; i++) {
            selectedFileNames.push(this.files[i].name);
        }
    });

    return selectedFileNames;
}

function renderUploadDocumentProcessingState(fileNames) {
    ensureUploadDocumentProcessingStyles();

    var target = $("#documents-list");

    if (!target.length) {
        console.warn("documents-list results container not found.");
        return;
    }

    var filesText = Array.isArray(fileNames) && fileNames.length
        ? fileNames.join(", ")
        : "Selected file(s)";

    target.html(`
        <li class="emtac-search-result-item">
            <div class="emtac-search-result-title">Processing Upload</div>
            <div class="emtac-upload-processing-box">
                <div class="emtac-upload-spinner"></div>
                <div>
                    <div class="emtac-upload-processing-title">Working...</div>
                    <p>EMTAC is extracting text, creating chunks, generating embeddings, and saving the document.</p>
                    <p><strong>File:</strong> ${escapeUploadDocumentHtml(filesText)}</p>
                    <div class="emtac-upload-processing-step">
                        Current stage: upload submitted and server-side processing is running.
                    </div>
                    <p class="emtac-upload-processing-note">
                        Large scanned PDFs may take longer because image/VLM extraction may run.
                    </p>
                </div>
            </div>
        </li>
    `);

    showUploadDocumentResults();
}

function renderUploadDocumentResults(response, isError) {
    ensureUploadDocumentProcessingStyles();

    var target = $("#documents-list");

    if (!target.length) {
        console.warn("documents-list results container not found.");
        alert(response.message || response.error || "Upload completed.");
        return;
    }

    target.empty();

    var status = response.status || (isError ? "failed" : "success");
    var message = response.message || response.error || "Upload completed.";
    var documentIds = Array.isArray(response.document_ids)
        ? response.document_ids.join(", ")
        : "";

    var statusClass = isError || status === "failed" || status === "skipped"
        ? "emtac-upload-result-error"
        : "emtac-upload-result-success";

    var tableHtml = `
        <li class="emtac-search-result-item">
            <div class="emtac-search-result-title">Upload Results</div>
            <table class="table table-striped table-bordered" style="width:100%;">
                <thead>
                    <tr>
                        <th>Status</th>
                        <th>Message</th>
                        <th>Documents</th>
                        <th>Chunks</th>
                        <th>Embeddings</th>
                        <th>Images</th>
                        <th>Position</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td class="${statusClass}">${escapeUploadDocumentHtml(status)}</td>
                        <td>${escapeUploadDocumentHtml(message)}</td>
                        <td>${escapeUploadDocumentHtml(documentIds)}</td>
                        <td>${escapeUploadDocumentHtml(response.chunks_created ?? 0)}</td>
                        <td>${escapeUploadDocumentHtml(response.embeddings_created ?? 0)}</td>
                        <td>${escapeUploadDocumentHtml(response.images_extracted ?? 0)}</td>
                        <td>${escapeUploadDocumentHtml(response.position_id ?? "")}</td>
                    </tr>
                </tbody>
            </table>
        </li>
    `;

    target.append(tableHtml);
    showUploadDocumentResults();
}

function attachUploadDocumentSubmitHandler() {
    var form = $("#upload-document form").first();

    if (!form.length) {
        form = $("form[action*='add_document']").first();
    }

    if (!form.length) {
        console.warn("Upload document form not found.");
        return;
    }

    form.off("submit.uploadDocumentAjax").on("submit.uploadDocumentAjax", function (event) {
        event.preventDefault();

        var formElement = this;
        var formData = new FormData(formElement);
        var actionUrl = $(formElement).attr("action") || "/documents/add_document";
        var submitButton = $(formElement).find("button[type='submit'], input[type='submit']").first();
        var selectedFileNames = getSelectedUploadDocumentFileNames(formElement);

        submitButton.prop("disabled", true);
        renderUploadDocumentProcessingState(selectedFileNames);

        $.ajax({
            url: actionUrl,
            type: "POST",
            data: formData,
            processData: false,
            contentType: false,
            headers: {
                "X-Requested-With": "XMLHttpRequest"
            },
            success: function (response) {
                console.log("Upload document response:", response);
                renderUploadDocumentResults(response, false);
            },
            error: function (xhr, status, error) {
                console.error("Upload document failed:", error);
                console.error("Status:", status);
                console.error("Response:", xhr.responseText);

                var response = xhr.responseJSON || {
                    status: "failed",
                    message: xhr.responseText || error || "Upload failed."
                };

                renderUploadDocumentResults(response, true);
            },
            complete: function () {
                submitButton.prop("disabled", false);
            }
        });
    });
}

function populateUploadDocumentDropdowns() {
    console.log("Populating upload document dropdowns...");

    $.ajax({
        url: "/get_upload_document_list_data",
        type: "GET",
        success: function (data) {
            console.log("Upload document data received:", data);

            var areaDropdown = $("#document_areaDropdown");
            var equipmentGroupDropdown = $("#document_equipmentGroupDropdown");
            var modelDropdown = $("#document_modelDropdown");
            var assetNumberDropdown = $("#document_assetNumberDropdown");
            var locationDropdown = $("#document_locationDropdown");

            areaDropdown.empty().append('<option value="">Select Area...</option>');
            equipmentGroupDropdown.empty().append('<option value="">Select Equipment Group...</option>');
            modelDropdown.empty().append('<option value="">Select Model...</option>');
            assetNumberDropdown.empty().append('<option value="">Select Asset Number...</option>');
            locationDropdown.empty().append('<option value="">Select Location...</option>');

            $.each(data.areas || [], function (index, area) {
                areaDropdown.append('<option value="' + area.id + '">' + area.name + "</option>");
            });

            areaDropdown.off("change.uploadDocument").on("change.uploadDocument", function () {
                var selectedAreaId = $(this).val();

                equipmentGroupDropdown.empty().append('<option value="">Select Equipment Group...</option>');
                modelDropdown.empty().append('<option value="">Select Model...</option>');
                assetNumberDropdown.empty().append('<option value="">Select Asset Number...</option>');
                locationDropdown.empty().append('<option value="">Select Location...</option>');

                if (selectedAreaId) {
                    $.each(data.equipment_groups || [], function (index, group) {
                        if (String(group.area_id) === String(selectedAreaId)) {
                            equipmentGroupDropdown.append('<option value="' + group.id + '">' + group.name + "</option>");
                        }
                    });
                }

                equipmentGroupDropdown.trigger("change");
            });

            equipmentGroupDropdown.off("change.uploadDocument").on("change.uploadDocument", function () {
                var selectedGroupId = $(this).val();

                modelDropdown.empty().append('<option value="">Select Model...</option>');
                assetNumberDropdown.empty().append('<option value="">Select Asset Number...</option>');
                locationDropdown.empty().append('<option value="">Select Location...</option>');

                if (selectedGroupId) {
                    $.each(data.models || [], function (index, model) {
                        if (String(model.equipment_group_id) === String(selectedGroupId)) {
                            modelDropdown.append('<option value="' + model.id + '">' + model.name + "</option>");
                        }
                    });
                }

                modelDropdown.trigger("change");
            });

            modelDropdown.off("change.uploadDocument").on("change.uploadDocument", function () {
                var selectedModelId = $(this).val();

                assetNumberDropdown.empty().append('<option value="">Select Asset Number...</option>');
                locationDropdown.empty().append('<option value="">Select Location...</option>');

                if (selectedModelId) {
                    $.each(data.asset_numbers || [], function (index, assetNumber) {
                        if (String(assetNumber.model_id) === String(selectedModelId)) {
                            assetNumberDropdown.append('<option value="' + assetNumber.id + '">' + assetNumber.number + "</option>");
                        }
                    });

                    $.each(data.locations || [], function (index, location) {
                        if (String(location.model_id) === String(selectedModelId)) {
                            locationDropdown.append('<option value="' + location.id + '">' + location.name + "</option>");
                        }
                    });
                }
            });

            areaDropdown.select2({ placeholder: "Select Area", width: "100%" });
            equipmentGroupDropdown.select2({ placeholder: "Select Equipment Group", width: "100%" });
            modelDropdown.select2({ placeholder: "Select Model", width: "100%" });
            assetNumberDropdown.select2({ placeholder: "Select Asset Number", width: "100%" });
            locationDropdown.select2({ placeholder: "Select Location", width: "100%" });

            console.log("Upload document dropdowns populated successfully");
        },
        error: function (xhr, status, error) {
            console.error("Error fetching upload document data:", error);
            console.error("Status:", status);
            console.error("Response:", xhr.responseText);
        }
    });
}

$(document).ready(function () {
    console.log("Upload document JS ready");
    populateUploadDocumentDropdowns();
    attachUploadDocumentSubmitHandler();
});