// batch_upload.js
// Handles batch upload dropdowns, AJAX batch submit, visible processing state,
// Socket.IO live progress updates, and batch upload result rendering.

function escapeBatchUploadHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
}

function ensureBatchUploadProcessingStyles() {
    if (document.getElementById("emtac-batch-upload-processing-styles")) {
        return;
    }

    var style = document.createElement("style");
    style.id = "emtac-batch-upload-processing-styles";

    style.textContent = `
        .emtac-batch-upload-processing-box {
            display: flex;
            gap: 14px;
            align-items: flex-start;
            padding: 12px;
            border: 1px solid rgba(57, 255, 20, 0.35);
            border-radius: 8px;
            background: rgba(0, 0, 0, 0.18);
            color: #eee;
        }

        .emtac-batch-upload-spinner {
            width: 30px;
            height: 30px;
            min-width: 30px;
            border: 4px solid rgba(255, 255, 255, 0.25);
            border-top-color: #39FF14;
            border-radius: 50%;
            animation: emtacBatchUploadSpin 0.9s linear infinite;
            margin-top: 4px;
        }

        .emtac-batch-upload-processing-title {
            font-weight: 700;
            margin-bottom: 4px;
        }

        .emtac-batch-upload-processing-note {
            color: #facc15;
            font-size: 13px;
            margin-top: 6px;
        }

        .emtac-batch-upload-processing-step {
            color: #bbb;
            font-size: 13px;
            margin-top: 4px;
            overflow-wrap: anywhere;
        }

        .emtac-batch-upload-processing-request {
            color: #9ca3af;
            font-size: 12px;
            margin-top: 4px;
            overflow-wrap: anywhere;
        }

        .emtac-batch-upload-result-success {
            color: #39FF14;
            font-weight: 700;
        }

        .emtac-batch-upload-result-error {
            color: #ff6b6b;
            font-weight: 700;
        }

        .emtac-batch-upload-result-warning {
            color: #facc15;
            font-weight: 700;
        }

        .emtac-batch-upload-summary {
            margin-bottom: 10px;
            color: #eee;
        }

        .emtac-batch-upload-results-table-wrapper {
            width: 100%;
            overflow-x: auto;
        }

        @keyframes emtacBatchUploadSpin {
            to {
                transform: rotate(360deg);
            }
        }
    `;

    document.head.appendChild(style);
}

function showBatchUploadResults() {
    var resultsPanel = $("#results");
    var documentsList = $("#documents-list");

    if (resultsPanel.length) {
        resultsPanel.show();
    }

    if (documentsList.length) {
        documentsList.show();
    }
}

function updateBatchUploadProgressMessage(data) {
    if (!data) {
        return;
    }

    var message = data.message || "Processing batch upload...";
    var pageText = "";

    if (data.page && data.total_pages) {
        pageText = " Page " + data.page + " of " + data.total_pages + ".";
    }

    var modeText = data.mode ? " [" + data.mode + "]" : "";
    var finalText = message + pageText + modeText;

    $(".emtac-batch-upload-processing-step").text(finalText);

    if (data.request_id) {
        $(".emtac-batch-upload-processing-request").text("Request ID: " + data.request_id);
    }
}

function attachBatchUploadProgressSocketListener() {
    if (!window.io) {
        console.warn("Socket.IO client not available. Batch upload progress streaming disabled.");
        return;
    }

    if (window.EMTAC_BATCH_UPLOAD_PROGRESS_SOCKET_ATTACHED) {
        return;
    }

    window.EMTAC_BATCH_UPLOAD_PROGRESS_SOCKET_ATTACHED = true;

    var socket = window.io();

    socket.on("upload_processing_progress", function (data) {
        console.log("Batch upload progress:", data);
        updateBatchUploadProgressMessage(data);
    });
}

function getSelectedBatchUploadFileNames(formElement) {
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

function getBatchFolderPath(formElement) {
    var folderInput = $(formElement).find("input[name='batchFolderPath'], #batchFolderPath").first();
    return folderInput.length ? folderInput.val() : "";
}

function renderBatchUploadProcessingState(formElement) {
    ensureBatchUploadProcessingStyles();

    var target = $("#documents-list");

    if (!target.length) {
        console.warn("documents-list results container not found.");
        return;
    }

    var selectedFileNames = getSelectedBatchUploadFileNames(formElement);
    var folderPath = getBatchFolderPath(formElement);

    var filesText = selectedFileNames.length
        ? selectedFileNames.join(", ")
        : "No direct uploaded files selected";

    var folderText = folderPath
        ? folderPath
        : "No folder path provided";

    target.html(`
        <li class="emtac-search-result-item">
            <div class="emtac-search-result-title">Processing Batch Upload</div>
            <div class="emtac-batch-upload-processing-box">
                <div class="emtac-batch-upload-spinner"></div>
                <div>
                    <div class="emtac-batch-upload-processing-title">Working...</div>
                    <p>EMTAC is staging files, extracting document text, creating chunks, generating embeddings, saving images, and linking records.</p>
                    <p><strong>Folder:</strong> ${escapeBatchUploadHtml(folderText)}</p>
                    <p><strong>Uploaded files:</strong> ${escapeBatchUploadHtml(filesText)}</p>
                    <div class="emtac-batch-upload-processing-step">
                        Current stage: batch request submitted and server-side processing is running.
                    </div>
                    <div class="emtac-batch-upload-processing-request"></div>
                    <p class="emtac-batch-upload-processing-note">
                        Large folders, scanned PDFs, PowerPoints, and image extraction can take longer. The page is still working.
                    </p>
                </div>
            </div>
        </li>
    `);

    showBatchUploadResults();
}

function getBatchResultStatusClass(status) {
    var normalized = String(status || "").toLowerCase();

    if (normalized === "success" || normalized === "processed") {
        return "emtac-batch-upload-result-success";
    }

    if (normalized === "partial_success" || normalized === "skipped") {
        return "emtac-batch-upload-result-warning";
    }

    return "emtac-batch-upload-result-error";
}

function normalizeBatchUploadResults(response) {
    if (!response) {
        return [];
    }

    if (Array.isArray(response.results)) {
        return response.results;
    }

    if (Array.isArray(response)) {
        return response;
    }

    return [
        {
            file_name: response.file_name || "",
            file_type: response.file_type || "document",
            status: response.status || "",
            message: response.message || "",
            response: response,
            duration_ms: response.duration_ms || 0
        }
    ];
}

function renderBatchUploadResults(response, isError) {
    ensureBatchUploadProcessingStyles();

    var target = $("#documents-list");

    if (!target.length) {
        console.warn("documents-list results container not found.");
        alert(response.message || response.error || "Batch upload completed.");
        return;
    }

    target.empty();

    var overallStatus = response.status || (isError ? "failed" : "success");
    var overallMessage = response.message || response.error || "Batch upload completed.";

    var processed = response.processed ?? "";
    var failed = response.failed ?? "";
    var skipped = response.skipped ?? "";
    var totalFilesFound = response.total_files_found ?? "";
    var durationMs = response.duration_ms ?? "";

    var results = normalizeBatchUploadResults(response);

    var rowsHtml = "";

    results.forEach(function (item) {
        var itemResponse = item.response || {};
        var docIds = Array.isArray(itemResponse.document_ids)
            ? itemResponse.document_ids.join(", ")
            : "";

        var imageId = "";
        if (itemResponse.image_id) {
            imageId = itemResponse.image_id;
        } else if (
            Array.isArray(itemResponse.results) &&
            itemResponse.results.length > 0 &&
            itemResponse.results[0].image_id
        ) {
            imageId = itemResponse.results[0].image_id;
        }

        var status = item.status || itemResponse.status || "";
        var statusClass = getBatchResultStatusClass(status);

        rowsHtml += `
            <tr>
                <td>${escapeBatchUploadHtml(item.file_name || "")}</td>
                <td>${escapeBatchUploadHtml(item.file_type || "")}</td>
                <td class="${statusClass}">${escapeBatchUploadHtml(status)}</td>
                <td>${escapeBatchUploadHtml(item.message || itemResponse.message || "")}</td>
                <td>${escapeBatchUploadHtml(docIds)}</td>
                <td>${escapeBatchUploadHtml(imageId)}</td>
                <td>${escapeBatchUploadHtml(itemResponse.chunks_created ?? "")}</td>
                <td>${escapeBatchUploadHtml(itemResponse.embeddings_created ?? "")}</td>
                <td>${escapeBatchUploadHtml(itemResponse.images_extracted ?? "")}</td>
                <td>${escapeBatchUploadHtml(itemResponse.position_id ?? "")}</td>
                <td>${escapeBatchUploadHtml(item.duration_ms ?? "")}</td>
            </tr>
        `;
    });

    if (!rowsHtml) {
        rowsHtml = `
            <tr>
                <td colspan="11">No per-file results were returned.</td>
            </tr>
        `;
    }

    var overallStatusClass = getBatchResultStatusClass(overallStatus);

    var tableHtml = `
        <li class="emtac-search-result-item">
            <div class="emtac-search-result-title">Batch Upload Results</div>

            <div class="emtac-batch-upload-summary">
                <div><strong>Status:</strong> <span class="${overallStatusClass}">${escapeBatchUploadHtml(overallStatus)}</span></div>
                <div><strong>Message:</strong> ${escapeBatchUploadHtml(overallMessage)}</div>
                <div>
                    <strong>Total:</strong> ${escapeBatchUploadHtml(totalFilesFound)}
                    &nbsp; <strong>Processed:</strong> ${escapeBatchUploadHtml(processed)}
                    &nbsp; <strong>Failed:</strong> ${escapeBatchUploadHtml(failed)}
                    &nbsp; <strong>Skipped:</strong> ${escapeBatchUploadHtml(skipped)}
                    &nbsp; <strong>Duration ms:</strong> ${escapeBatchUploadHtml(durationMs)}
                </div>
            </div>

            <div class="emtac-batch-upload-results-table-wrapper">
                <table class="table table-striped table-bordered" style="width:100%;">
                    <thead>
                        <tr>
                            <th>File</th>
                            <th>Type</th>
                            <th>Status</th>
                            <th>Message</th>
                            <th>Document IDs</th>
                            <th>Image ID</th>
                            <th>Chunks</th>
                            <th>Embeddings</th>
                            <th>Images Extracted</th>
                            <th>Position</th>
                            <th>Duration ms</th>
                        </tr>
                    </thead>
                    <tbody>
                        ${rowsHtml}
                    </tbody>
                </table>
            </div>
        </li>
    `;

    target.append(tableHtml);
    showBatchUploadResults();
}

function attachBatchUploadSubmitHandler() {
    var form = $("#batch-upload form").first();

    if (!form.length) {
        form = $("form[action*='batch_processing']").first();
    }

    if (!form.length) {
        form = $("form[action*='add_batch_folder']").first();
    }

    if (!form.length) {
        console.warn("Batch upload form not found.");
        return;
    }

    form.off("submit.batchUploadAjax").on("submit.batchUploadAjax", function (event) {
        event.preventDefault();

        var formElement = this;
        var formData = new FormData(formElement);
        var actionUrl = $(formElement).attr("action") || "/batch_processing";
        var submitButton = $(formElement).find("button[type='submit'], input[type='submit']").first();

        submitButton.prop("disabled", true);
        renderBatchUploadProcessingState(formElement);

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
                console.log("Batch upload response:", response);
                renderBatchUploadResults(response, false);
            },
            error: function (xhr, status, error) {
                console.error("Batch upload failed:", error);
                console.error("Status:", status);
                console.error("Response:", xhr.responseText);

                var response = xhr.responseJSON || {
                    status: "failed",
                    message: xhr.responseText || error || "Batch upload failed."
                };

                renderBatchUploadResults(response, true);
            },
            complete: function () {
                submitButton.prop("disabled", false);
            }
        });
    });
}

function populateBatchDropdowns() {
    console.log("Populating batch upload dropdowns...");

    $.ajax({
        url: "/batch/get_batch_list_data",
        type: "GET",
        success: function (data) {
            var areaDropdown = $("#batchAreaDropdown");
            var equipmentGroupDropdown = $("#batchEquipmentGroupDropdown");
            var modelDropdown = $("#batchModelDropdown");
            var assetNumberDropdown = $("#batchAssetNumberDropdown");
            var locationDropdown = $("#batchLocationDropdown");

            areaDropdown.empty().append('<option value="">Select Area...</option>');
            equipmentGroupDropdown.empty().append('<option value="">Select Equipment Group...</option>');
            modelDropdown.empty().append('<option value="">Select Model...</option>');
            assetNumberDropdown.empty().append('<option value="">Select Asset Number...</option>');
            locationDropdown.empty().append('<option value="">Select Location...</option>');

            $.each(data.areas || [], function (index, area) {
                areaDropdown.append('<option value="' + area.id + '">' + area.name + "</option>");
            });

            areaDropdown.off("change.batchUpload").on("change.batchUpload", function () {
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

            equipmentGroupDropdown.off("change.batchUpload").on("change.batchUpload", function () {
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

            modelDropdown.off("change.batchUpload").on("change.batchUpload", function () {
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

            areaDropdown.trigger("change");

            console.log("Batch upload dropdowns populated successfully");
        },
        error: function (xhr, status, error) {
            console.error("Error fetching batch dropdown data:", error);
            console.error("Status:", status);
            console.error("Response:", xhr.responseText);
        }
    });
}

$(document).ready(function () {
    console.log("Batch upload JS ready");
    ensureBatchUploadProcessingStyles();
    attachBatchUploadProgressSocketListener();
    populateBatchDropdowns();
    attachBatchUploadSubmitHandler();
});