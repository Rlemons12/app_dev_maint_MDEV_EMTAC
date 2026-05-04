/* static/js/bill_of_materials/search_edit_part.js */

$(document).ready(function () {
    console.log("[BOM UI] search_edit_part.js loaded");
    console.log("[BOM UI] Document ready");

    /**
     * ---------------------------------------------------------
     * HELPERS
     * ---------------------------------------------------------
     */

    function ensureEditPartSearchContainer() {
        const $editPart = $("#edit-part");
        const $existing = $("#edit-part #search-results-container");

        console.log("[BOM UI] ensureEditPartSearchContainer()");
        console.log("[BOM UI] #edit-part exists:", $editPart.length > 0);
        console.log("[BOM UI] existing #search-results-container:", $existing.length);

        if ($existing.length === 0 && $editPart.length > 0) {
            $editPart.append(
                '<div id="search-results-container" class="edit-part-search-results"></div>'
            );
            console.log("[BOM UI] Added #search-results-container inside #edit-part");
        }
    }

    function showResultsContainer(loadingMessage) {
        console.log("[BOM UI] showResultsContainer()", loadingMessage);

        const $resultsContainer = $("#results-container");
        const $searchResults = $("#searchResults");
        const $editPartForm = $("#edit-part-form");

        console.log("[BOM UI] #results-container exists:", $resultsContainer.length > 0);
        console.log("[BOM UI] #searchResults exists:", $searchResults.length > 0);
        console.log("[BOM UI] #edit-part-form exists:", $editPartForm.length > 0);

        $searchResults.html(
            `<p style="color:white;">${loadingMessage || "Loading..."}</p>`
        );
        $resultsContainer.show();
        $editPartForm.hide();

        console.log("[BOM UI] Results container shown with loading message");
    }

    function showResultsHtml(html) {
        const htmlLength = html ? html.length : 0;
        console.log("[BOM UI] showResultsHtml() html length:", htmlLength);

        $("#searchResults").html(html);
        $("#results-container").show();
        $("#edit-part-form").hide();

        console.log("[BOM UI] Results HTML injected into #searchResults");
    }

    function showResultsError(message) {
        console.warn("[BOM UI] showResultsError()", message);

        $("#searchResults").html(
            `<p style="color:red;">${message || "An error occurred."}</p>`
        );
        $("#results-container").show();
        $("#edit-part-form").hide();
    }

    function hideAllFormContainers() {
        const count = $(".form-container").length;
        console.log("[BOM UI] hideAllFormContainers() count:", count);
        $(".form-container").hide();
    }

    function showRequestedContainer(containerId) {
        console.log("[BOM UI] showRequestedContainer()", containerId);

        if (!containerId) {
            console.warn("[BOM UI] No containerId provided");
            return;
        }

        const $target = $("#" + containerId);
        console.log("[BOM UI] target exists:", $target.length > 0);

        if ($target.length) {
            $target.show();
            console.log("[BOM UI] Shown container:", containerId);
        } else {
            console.warn("[BOM UI] Container not found:", containerId);
        }
    }

    function responseLooksLikeFullPage(html) {
        if (!html || typeof html !== "string") {
            return false;
        }

        const sample = html.substring(0, 500).toLowerCase();
        return (
            sample.includes("<!doctype html") ||
            sample.includes("<html") ||
            sample.includes("<head") ||
            sample.includes("<body")
        );
    }

    /**
     * ---------------------------------------------------------
     * INITIAL SETUP
     * ---------------------------------------------------------
     */

    ensureEditPartSearchContainer();

    console.log("[BOM UI] Initial DOM presence check", {
        resultsContainer: $("#results-container").length,
        searchResults: $("#searchResults").length,
        editPartForm: $("#edit-part-form").length,
        advancedSearchForm: $("#advancedSearchForm").length,
        advancedSearchFilterForm: $("#advancedSearchFilterForm").length,
        generalSearch: $("#generalSearch").length,
        editPart: $("#edit-part").length
    });

    /**
     * ---------------------------------------------------------
     * TOGGLE ADVANCED SEARCH
     * ---------------------------------------------------------
     */

    $("#toggleFormBtn").on("click", function () {
        console.log("[BOM UI] #toggleFormBtn clicked");
        $("#advancedSearchForm").toggle();
        console.log(
            "[BOM UI] #advancedSearchForm visible after toggle:",
            $("#advancedSearchForm").is(":visible")
        );
    });

    /**
     * ---------------------------------------------------------
     * PART SEARCH (EDIT PART WORKFLOW)
     * ---------------------------------------------------------
     */

    $(document).on("submit", "#search-part-form", function (e) {
        e.preventDefault();

        const query = $("#search_query").val();
        const inEditPart =
            $("#edit-part").is(":visible") || window.location.hash === "#edit-part";

        console.log("[BOM UI] #search-part-form submitted");
        console.log("[BOM UI] search query:", query);
        console.log("[BOM UI] inEditPart:", inEditPart);

        if (inEditPart) {
            console.log("[BOM UI] Running edit-part search flow");

            $("#edit-part-form").hide();
            $("#edit-part #search-results-container")
                .html('<p style="color:white;">Loading...</p>')
                .show();

            $("#edit-part").show();
            $("#results-container").hide();

            $.ajax({
                url: "/update_part/search_part_ajax",
                type: "GET",
                data: { search_query: query },
                success: function (html) {
                    console.log("[BOM UI] /update_part/search_part_ajax success");
                    console.log("[BOM UI] returned html length:", html ? html.length : 0);

                    $("#edit-part #search-results-container").html(html).show();
                    $("#edit-part").show();
                },
                error: function (xhr, status, error) {
                    console.error("[BOM UI] /update_part/search_part_ajax error", {
                        status: status,
                        error: error,
                        responseText: xhr.responseText
                    });

                    $("#edit-part #search-results-container").html(
                        '<p style="color:red;">An error occurred.</p>'
                    );
                }
            });
        } else {
            console.log("[BOM UI] Running standard search-results flow");

            showResultsContainer("Loading...");

            $.ajax({
                url: "/update_part/search_part_ajax",
                type: "GET",
                data: { search_query: query },
                success: function (html) {
                    console.log("[BOM UI] /update_part/search_part_ajax success");
                    console.log("[BOM UI] returned html length:", html ? html.length : 0);

                    showResultsHtml(html);
                    hideAllFormContainers();
                },
                error: function (xhr, status, error) {
                    console.error("[BOM UI] /update_part/search_part_ajax error", {
                        status: status,
                        error: error,
                        responseText: xhr.responseText
                    });

                    showResultsError("An error occurred.");
                }
            });
        }
    });

    /**
     * ---------------------------------------------------------
     * BOM ADVANCED SEARCH
     * ---------------------------------------------------------
     */

    $(document).on("submit", "#advancedSearchFilterForm", function (e) {
        e.preventDefault();
        e.stopPropagation();

        const $form = $(this);
        const formAction = $form.attr("action");
        const formData = $form.serialize();

        console.log("[BOM UI] #advancedSearchFilterForm submitted");
        console.log("[BOM UI] action:", formAction);
        console.log("[BOM UI] serialized data:", formData);

        showResultsContainer("Loading BOM results...");

        $.ajax({
            url: formAction,
            type: "POST",
            data: formData,
            headers: {
                "X-Requested-With": "XMLHttpRequest"
            },
            success: function (html, status, xhr) {
                console.log("[BOM UI] Advanced BOM search AJAX success");
                console.log("[BOM UI] status:", status);
                console.log("[BOM UI] response status code:", xhr.status);
                console.log("[BOM UI] html length:", html ? html.length : 0);
                console.log("[BOM UI] response preview:", html ? html.substring(0, 300) : "(empty)");

                if (responseLooksLikeFullPage(html)) {
                    console.error("[BOM UI] ERROR: Advanced BOM search returned a full HTML page, not a partial.");
                    console.error("[BOM UI] The backend route must return a partial for AJAX requests.");
                    showResultsError("Search returned a full page instead of a results partial.");
                    return;
                }

                showResultsHtml(html);
            },
            error: function (xhr, status, error) {
                console.error("[BOM UI] Advanced BOM search AJAX error", {
                    status: status,
                    error: error,
                    responseStatus: xhr.status,
                    responseText: xhr.responseText
                });

                showResultsError("An error occurred during BOM search.");
            }
        });

        return false;
    });

    /**
     * ---------------------------------------------------------
     * BOM GENERAL SEARCH
     * ---------------------------------------------------------
     */

    $(document).on("submit", "#generalSearch form", function (e) {
        e.preventDefault();
        e.stopPropagation();

        const $form = $(this);
        const formAction = $form.attr("action");
        const formData = $form.serialize();

        console.log("[BOM UI] General BOM search submitted");
        console.log("[BOM UI] action:", formAction);
        console.log("[BOM UI] serialized data:", formData);

        showResultsContainer("Loading BOM results...");

        $.ajax({
            url: formAction,
            type: "POST",
            data: formData,
            headers: {
                "X-Requested-With": "XMLHttpRequest"
            },
            success: function (html, status, xhr) {
                console.log("[BOM UI] General BOM search AJAX success");
                console.log("[BOM UI] status:", status);
                console.log("[BOM UI] response status code:", xhr.status);
                console.log("[BOM UI] html length:", html ? html.length : 0);
                console.log("[BOM UI] response preview:", html ? html.substring(0, 300) : "(empty)");

                if (responseLooksLikeFullPage(html)) {
                    console.error("[BOM UI] ERROR: General BOM search returned a full HTML page, not a partial.");
                    console.error("[BOM UI] The backend route must return a partial for AJAX requests.");
                    showResultsError("Search returned a full page instead of a results partial.");
                    return;
                }

                showResultsHtml(html);
            },
            error: function (xhr, status, error) {
                console.error("[BOM UI] General BOM search AJAX error", {
                    status: status,
                    error: error,
                    responseStatus: xhr.status,
                    responseText: xhr.responseText
                });

                showResultsError("An error occurred during BOM search.");
            }
        });

        return false;
    });

    /**
     * ---------------------------------------------------------
     * EDIT PART BUTTON FROM SEARCH RESULTS
     * ---------------------------------------------------------
     */

    $(document).on("click", ".edit-part-btn", function () {
        const partId = $(this).data("part-id");

        console.log("[BOM UI] .edit-part-btn clicked");
        console.log("[BOM UI] partId:", partId);

        if (!partId) {
            console.error("[BOM UI] No part ID found on .edit-part-btn");
            return;
        }

        if (history.pushState) {
            history.pushState(null, null, "#edit-part");
            console.log("[BOM UI] Updated hash with pushState to #edit-part");
        } else {
            location.hash = "#edit-part";
            console.log("[BOM UI] Updated hash with location.hash to #edit-part");
        }

        $("#results-container").hide();
        $("#advancedSearchForm").hide();
        $("#edit-part").show();

        $("#edit-part-form")
            .html('<p style="color:white;">Loading edit form...</p>')
            .show();

        $.ajax({
            url: "/update_part/edit_part_ajax/" + partId,
            type: "GET",
            cache: false,
            success: function (data, status, xhr) {
                console.log("[BOM UI] /edit_part_ajax success");
                console.log("[BOM UI] status:", status);
                console.log("[BOM UI] response status code:", xhr.status);
                console.log("[BOM UI] html length:", data ? data.length : 0);

                $("#edit-part-form").html(data);

                const form = $("#edit-part-form form");
                console.log("[BOM UI] Edit form found inside response:", form.length > 0);

                if (form.length > 0) {
                    form.append('<input type="hidden" name="ajax" value="true">');
                    form.attr("id", "ajax-edit-form");

                    if ($("#cancel-edit-btn").length === 0) {
                        form.find('button[type="submit"], input[type="submit"]').after(
                            '<button type="button" class="btn btn-secondary ml-2" id="cancel-edit-btn">Cancel</button>'
                        );
                        console.log("[BOM UI] Cancel button added to ajax edit form");
                    }
                }

                if ($("#edit-part-form").length && $("#edit-part-form").offset()) {
                    $("html, body").animate(
                        {
                            scrollTop: $("#edit-part-form").offset().top - 50
                        },
                        500
                    );
                    console.log("[BOM UI] Scrolled to #edit-part-form");
                }
            },
            error: function (xhr, status, error) {
                console.error("[BOM UI] /edit_part_ajax error", {
                    status: status,
                    error: error,
                    responseText: xhr.responseText
                });

                $("#edit-part-form").html(
                    '<div class="alert alert-danger">Error loading part details: ' +
                        error +
                        "</div>"
                );
            }
        });
    });

    /**
     * ---------------------------------------------------------
     * EDIT PART FORM SUBMISSION
     * ---------------------------------------------------------
     */

    $(document).on("submit", "#ajax-edit-form", function (e) {
        e.preventDefault();

        const form = $(this);
        const action = form.attr("action");
        const formData = new FormData(this);

        console.log("[BOM UI] #ajax-edit-form submitted");
        console.log("[BOM UI] action:", action);

        $.ajax({
            url: action,
            type: "POST",
            data: formData,
            processData: false,
            contentType: false,
            success: function (response, status, xhr) {
                console.log("[BOM UI] ajax edit form success");
                console.log("[BOM UI] status:", status);
                console.log("[BOM UI] response status code:", xhr.status);
                console.log("[BOM UI] response:", response);

                $("#edit-part-form").prepend(
                    '<div class="alert alert-success">Part updated successfully!</div>'
                );

                const query = $("#search_query").val();
                console.log("[BOM UI] current #search_query after save:", query);

                if (query) {
                    $.ajax({
                        url: "/update_part/search_part_ajax",
                        type: "GET",
                        data: { search_query: query },
                        success: function (data, searchStatus, searchXhr) {
                            console.log("[BOM UI] refreshed part search after save");
                            console.log("[BOM UI] status:", searchStatus);
                            console.log("[BOM UI] response status code:", searchXhr.status);
                            console.log("[BOM UI] html length:", data ? data.length : 0);

                            $("#edit-part #search-results-container").html(data);
                        },
                        error: function (xhr, status, error) {
                            console.error("[BOM UI] error refreshing part search after save", {
                                status: status,
                                error: error,
                                responseText: xhr.responseText
                            });
                        }
                    });
                }
            },
            error: function (xhr, status, error) {
                console.error("[BOM UI] ajax edit form error", {
                    status: status,
                    error: error,
                    responseStatus: xhr.status,
                    responseText: xhr.responseText
                });

                const errorMessage =
                    xhr.responseJSON?.message || "Error updating part";

                $("#edit-part-form").prepend(
                    '<div class="alert alert-danger">' + errorMessage + "</div>"
                );
            }
        });
    });

    /**
     * ---------------------------------------------------------
     * CANCEL EDIT
     * ---------------------------------------------------------
     */

    $(document).on("click", "#cancel-edit-btn", function () {
        console.log("[BOM UI] #cancel-edit-btn clicked");

        $("#edit-part-form").html("").hide();
        $("#edit-part").show();

        if ($("#edit-part #search-results-container").children().length > 0) {
            $("#edit-part #search-results-container").show();
            console.log("[BOM UI] Restored edit-part search-results-container");
        }
    });

    /**
     * ---------------------------------------------------------
     * HASH-BASED VIEW SWITCHING
     * ---------------------------------------------------------
     */

    function handleHashChange() {
        const hash = window.location.hash.substring(1);

        console.log("[BOM UI] handleHashChange()");
        console.log("[BOM UI] current hash:", hash || "(empty)");

        if (hash) {
            hideAllFormContainers();
            $("#results-container").hide();
            $("#edit-part-form").hide();
            $("#advancedSearchForm").hide();

            showRequestedContainer(hash);

            if (hash === "search-bill-of-materials") {
                console.log("[BOM UI] search-bill-of-materials hash detected");
                if (window.populateDropdownsForPartsPosition) {
                    console.log("[BOM UI] Calling populateDropdownsForPartsPosition()");
                    window.populateDropdownsForPartsPosition();
                } else {
                    console.warn("[BOM UI] populateDropdownsForPartsPosition is missing");
                }
            }

            if (
                hash === "edit-part" &&
                $("#edit-part #search-results-container").children().length > 0
            ) {
                $("#edit-part #search-results-container").show();
                console.log("[BOM UI] Showing #edit-part #search-results-container");
            }
        } else {
            console.log("[BOM UI] No hash found, defaulting to search-bill-of-materials");

            hideAllFormContainers();
            $("#results-container").hide();
            $("#edit-part-form").hide();
            $("#search-bill-of-materials").show();

            if (window.populateDropdownsForPartsPosition) {
                console.log("[BOM UI] Calling populateDropdownsForPartsPosition() for default view");
                window.populateDropdownsForPartsPosition();
            } else {
                console.warn("[BOM UI] populateDropdownsForPartsPosition is missing");
            }
        }
    }

    console.log("[BOM UI] Running initial handleHashChange()");
    handleHashChange();

    $(window).on("hashchange", function () {
        console.log("[BOM UI] window hashchange event fired");
        handleHashChange();
    });

    console.log("[BOM UI] search_edit_part.js initialization complete");
});