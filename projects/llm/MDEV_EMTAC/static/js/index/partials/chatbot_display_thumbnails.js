// ============================================================
// MDEV_EMTAC\static\js\index\partials\chatbot_display_thumbnails.js
// Render Images as DOCUMENT-STYLE LINKS
//
// WebView-safe:
// - Does NOT use window.open()
// - Calls in-page image viewer when available
// - Falls back to window.openImageViewer(), which should also be WebView-safe
//
// Page-based image browsing:
// - Shows one page of images at a time
// - Default page size is 25
// - Adds top pager tabs/range buttons
// - Does NOT append pages into the DOM
// - Supports full local array paging
// - Supports backend page endpoint when available
//
// Rules:
// - Prefer web-safe image routes: /serve_image/<id>
// - Never prefer raw DB_IMAGES/file_path paths
// - Support array input OR { images: [...] }
// ============================================================

console.log("[EMTAC] chatbot_display_thumbnails.js loaded - WebView-safe page-tab image renderer");

// ------------------------------------------------------------
// Config
// ------------------------------------------------------------

const EMTAC_THUMBNAIL_DEFAULT_PAGE_SIZE = 25;
const EMTAC_THUMBNAIL_MAX_PAGE_SIZE = 100;
const EMTAC_THUMBNAIL_PAGE_WINDOW = 5;

// Backend endpoint:
//
// GET /chatbot/ask/payload/images?request_id=<id>&page=2&page_size=25
//
// Optional filters are also supported when the backend accepts them:
// - position_id
// - complete_document_id
//
const EMTAC_DEFAULT_IMAGE_PAGE_ENDPOINT = "/chatbot/ask/payload/images";

let thumbnailPaginationState = null;


// ------------------------------------------------------------
// Main Renderer
// ------------------------------------------------------------

function displayThumbnails(imagePanel, options = {}) {
    const section = document.getElementById("thumbnails-section");

    if (!section) {
        console.warn("[displayThumbnails] #thumbnails-section not found");
        return;
    }

    section.replaceChildren();
    section.classList.add("images-container", "image-results");

    const normalized = normalizeThumbnailInput(imagePanel, options);

    const totalPages = calculateTotalPages(
        normalized.totalCount,
        normalized.pageSize,
        normalized.images.length
    );

    const currentPage = clampPage(normalized.currentPage, totalPages);

    const pageCache = new Map();

    pageCache.set(currentPage, {
        images: getInitialPageImages(normalized.images, currentPage, normalized.pageSize, normalized.totalCount),
        page: currentPage,
    });

    thumbnailPaginationState = {
        sectionId: "thumbnails-section",

        // Initial/full array from payload.
        // If this array is larger than one page, local paging is used.
        // If this array is only one page but totalCount is larger, server paging is used.
        localImages: normalized.images,

        pageCache,
        currentPage,
        currentImages: pageCache.get(currentPage).images,

        pageSize: normalized.pageSize,
        totalCount: normalized.totalCount,
        totalPages,

        requestId: normalized.requestId,
        positionId: normalized.positionId,
        completeDocumentId: normalized.completeDocumentId,

        endpoint: normalized.endpoint,

        isLoading: false,
        lastError: "",
    };

    window.EMTAC_THUMBNAIL_PAGINATION_STATE = thumbnailPaginationState;

    renderThumbnailPanel();
}


// ------------------------------------------------------------
// Normalize input
// ------------------------------------------------------------

function normalizeThumbnailInput(imagePanel, options = {}) {
    const sourceObject =
        imagePanel &&
        typeof imagePanel === "object" &&
        !Array.isArray(imagePanel)
            ? imagePanel
            : {};

    const images = Array.isArray(sourceObject.images)
        ? sourceObject.images
        : Array.isArray(imagePanel)
            ? imagePanel
            : [];

    const pageSize = normalizePageSize(
        options.pageSize ??
        sourceObject.images_page_size ??
        sourceObject.image_page_size ??
        sourceObject.page_size ??
        sourceObject.pageSize ??
        EMTAC_THUMBNAIL_DEFAULT_PAGE_SIZE
    );

    const totalCount = normalizeNonNegativeInteger(
        sourceObject.images_total ??
        sourceObject.image_count_total ??
        sourceObject.total_images ??
        sourceObject.total_count ??
        sourceObject.total ??
        images.length,
        images.length
    );

    const requestId =
        firstNonBlank(
            sourceObject.request_id,
            sourceObject.requestId,
            sourceObject.original_request_id,
            sourceObject.originalRequestId,
            window.EMTAC_LAST_PAYLOAD?.raw?.request_id,
            window.EMTAC_LAST_PAYLOAD?.raw?.original_request_id
        );

    const firstImage = images.find(item => item && typeof item === "object") || {};

    const positionId =
        firstNonBlank(
            sourceObject.position_id,
            sourceObject.positionId,
            sourceObject.active_position_id,
            sourceObject.activePositionId,
            firstImage.position_id,
            firstImage.positionId
        );

    const completeDocumentId =
        firstNonBlank(
            sourceObject.complete_document_id,
            sourceObject.completeDocumentId,
            firstImage.complete_document_id,
            firstImage.completeDocumentId
        );

    const endpoint =
        firstNonBlank(
            options.endpoint,
            sourceObject.images_endpoint,
            sourceObject.image_endpoint,
            sourceObject.image_pagination_endpoint,
            sourceObject.next_page_endpoint,
            sourceObject.nextPageEndpoint
        ) ||
        (
            requestId || positionId || completeDocumentId
                ? EMTAC_DEFAULT_IMAGE_PAGE_ENDPOINT
                : ""
        );

    const currentPage = normalizePositiveInteger(
        sourceObject.images_page ??
        sourceObject.image_page ??
        sourceObject.page ??
        1,
        1
    );

    return {
        images,
        pageSize,
        totalCount,
        requestId,
        positionId,
        completeDocumentId,
        endpoint,
        currentPage,
    };
}

function normalizePageSize(value) {
    const parsed = Number(value);

    if (!Number.isFinite(parsed) || parsed <= 0) {
        return EMTAC_THUMBNAIL_DEFAULT_PAGE_SIZE;
    }

    return Math.min(Math.floor(parsed), EMTAC_THUMBNAIL_MAX_PAGE_SIZE);
}

function normalizePositiveInteger(value, fallback) {
    const parsed = Number(value);

    if (!Number.isFinite(parsed) || parsed <= 0) {
        return fallback;
    }

    return Math.floor(parsed);
}

function normalizeNonNegativeInteger(value, fallback) {
    const parsed = Number(value);

    if (!Number.isFinite(parsed) || parsed < 0) {
        return fallback;
    }

    return Math.floor(parsed);
}

function firstNonBlank(...values) {
    for (const value of values) {
        if (value === null || value === undefined) {
            continue;
        }

        const normalized = String(value).trim();

        if (normalized) {
            return normalized;
        }
    }

    return "";
}

function calculateTotalPages(totalCount, pageSize, fallbackLength) {
    const count = Math.max(
        normalizeNonNegativeInteger(totalCount, 0),
        normalizeNonNegativeInteger(fallbackLength, 0)
    );

    if (count <= 0) {
        return 1;
    }

    return Math.max(1, Math.ceil(count / pageSize));
}

function clampPage(page, totalPages) {
    const safePage = normalizePositiveInteger(page, 1);
    const safeTotalPages = normalizePositiveInteger(totalPages, 1);

    return Math.min(Math.max(safePage, 1), safeTotalPages);
}

function getInitialPageImages(images, page, pageSize, totalCount) {
    const safeImages = Array.isArray(images) ? images : [];

    if (safeImages.length <= pageSize) {
        return safeImages;
    }

    const totalPages = calculateTotalPages(totalCount, pageSize, safeImages.length);
    const safePage = clampPage(page, totalPages);

    const startIndex = (safePage - 1) * pageSize;
    const endIndex = startIndex + pageSize;

    return safeImages.slice(startIndex, endIndex);
}

function canPageLocally(state) {
    return Boolean(
        state &&
        Array.isArray(state.localImages) &&
        state.localImages.length > state.pageSize
    );
}

function getLocalPageImages(state, page) {
    if (!canPageLocally(state)) {
        return [];
    }

    const safePage = clampPage(page, state.totalPages);
    const startIndex = (safePage - 1) * state.pageSize;
    const endIndex = startIndex + state.pageSize;

    return state.localImages.slice(startIndex, endIndex);
}


// ------------------------------------------------------------
// Render page-based panel
// ------------------------------------------------------------

function renderThumbnailPanel() {
    const state = thumbnailPaginationState;
    const section = document.getElementById(state?.sectionId || "thumbnails-section");

    if (!section) {
        console.warn("[renderThumbnailPanel] #thumbnails-section not found");
        return;
    }

    section.replaceChildren();
    section.classList.add("images-container", "image-results");

    if (!state || !Array.isArray(state.currentImages)) {
        const empty = document.createElement("p");
        empty.textContent = "No images found.";
        empty.style.color = "#aaa";
        section.appendChild(empty);
        return;
    }

    const totalCount = Math.max(state.totalCount || 0, state.currentImages.length);
    const pageRange = getPageRange(state.currentPage, state.pageSize, totalCount);
    const shownCount = state.currentImages.length;

    section.appendChild(buildThumbnailSummary({
        start: shownCount > 0 ? pageRange.start : 0,
        end: shownCount > 0 ? pageRange.start + shownCount - 1 : 0,
        shownCount,
        totalCount,
        currentPage: state.currentPage,
        totalPages: state.totalPages,
        isLoading: state.isLoading,
        lastError: state.lastError,
    }));

    const topPager = buildThumbnailPager("top");

    if (topPager) {
        section.appendChild(topPager);
    }

    if (state.currentImages.length === 0) {
        const emptyPage = document.createElement("p");
        emptyPage.textContent = "No images found for this page.";
        emptyPage.style.color = "#aaa";
        section.appendChild(emptyPage);
    } else {
        const list = document.createElement("div");
        list.className = "emtac-thumbnail-list";

        const fragment = document.createDocumentFragment();
        const startIndex = (state.currentPage - 1) * state.pageSize;

        state.currentImages.forEach((imgData, index) => {
            const item = buildThumbnailItem(imgData, startIndex + index);

            if (item) {
                fragment.appendChild(item);
            }
        });

        list.appendChild(fragment);
        section.appendChild(list);
    }

    const bottomPager = buildThumbnailPager("bottom");

    if (bottomPager) {
        section.appendChild(bottomPager);
    }
}

function getPageRange(page, pageSize, totalCount) {
    const safeTotal = normalizeNonNegativeInteger(totalCount, 0);
    const safePage = normalizePositiveInteger(page, 1);
    const start = safeTotal <= 0
        ? 0
        : ((safePage - 1) * pageSize) + 1;
    const end = safeTotal <= 0
        ? 0
        : Math.min(safeTotal, safePage * pageSize);

    return { start, end };
}

function buildThumbnailSummary({
    start,
    end,
    shownCount,
    totalCount,
    currentPage,
    totalPages,
    isLoading,
    lastError,
}) {
    const summary = document.createElement("div");
    summary.className = "emtac-thumbnail-summary";

    const safeTotal = Number.isFinite(totalCount) && totalCount > 0
        ? totalCount
        : shownCount;

    if (shownCount > 0) {
        summary.textContent = `Showing ${start}-${end} of ${safeTotal} images`;
    } else {
        summary.textContent = `Showing 0 of ${safeTotal} images`;
    }

    summary.textContent += ` (Page ${currentPage} of ${totalPages})`;

    if (isLoading) {
        const loading = document.createElement("span");
        loading.className = "emtac-thumbnail-loading";
        loading.textContent = " Loading...";
        summary.appendChild(loading);
    }

    if (lastError) {
        const error = document.createElement("div");
        error.className = "emtac-thumbnail-error";
        error.textContent = lastError;
        summary.appendChild(error);
    }

    return summary;
}

function buildThumbnailItem(imgData, absoluteIndex) {
    if (!imgData || typeof imgData !== "object") {
        console.warn("[displayThumbnails] Invalid image item:", imgData);
        return null;
    }

    const src = resolveImageUrl(imgData);

    if (!src) {
        console.warn("[displayThumbnails] Image missing usable src:", imgData);
        return null;
    }

    const title =
        imgData.title ||
        imgData.name ||
        imgData.description ||
        imgData.filename ||
        imgData.file_name ||
        `View Image ${absoluteIndex + 1}`;

    const item = document.createElement("div");
    item.className = "document-item image-document-item";

    const imageLink = document.createElement("button");
    imageLink.type = "button";
    imageLink.className = "document-chunk-link image-document-link";
    imageLink.textContent = title;
    imageLink.title = title;

    imageLink.addEventListener("click", event => {
        event.preventDefault();
        event.stopPropagation();

        openThumbnailImageViewer(title, src);
    });

    item.appendChild(imageLink);
    return item;
}

function buildThumbnailPager(position = "top") {
    const state = thumbnailPaginationState;

    if (!state || state.totalPages <= 1) {
        return null;
    }

    const controls = document.createElement("div");
    controls.className = `emtac-thumbnail-controls emtac-thumbnail-pagebar emtac-thumbnail-pagebar-${position}`;

    const firstButton = buildPagerButton({
        label: "First",
        title: "Go to first image set",
        disabled: state.currentPage <= 1 || state.isLoading,
        onClick: () => loadThumbnailImagePage(1),
    });

    const prevButton = buildPagerButton({
        label: "Prev",
        title: "Go to previous image set",
        disabled: state.currentPage <= 1 || state.isLoading,
        onClick: () => loadThumbnailImagePage(state.currentPage - 1),
    });

    controls.appendChild(firstButton);
    controls.appendChild(prevButton);

    const pageNumbers = getPagerWindow(
        state.currentPage,
        state.totalPages,
        EMTAC_THUMBNAIL_PAGE_WINDOW
    );

    let previousPageNumber = 0;

    pageNumbers.forEach(pageNumber => {
        if (previousPageNumber && pageNumber > previousPageNumber + 1) {
            controls.appendChild(buildPagerEllipsis());
        }

        controls.appendChild(buildPagerButton({
            label: buildPageRangeLabel(pageNumber, state.pageSize, state.totalCount),
            title: `Show images ${buildPageRangeLabel(pageNumber, state.pageSize, state.totalCount)}`,
            active: pageNumber === state.currentPage,
            disabled: state.isLoading,
            onClick: () => loadThumbnailImagePage(pageNumber),
        }));

        previousPageNumber = pageNumber;
    });

    const nextButton = buildPagerButton({
        label: "Next",
        title: "Go to next image set",
        disabled: state.currentPage >= state.totalPages || state.isLoading,
        onClick: () => loadThumbnailImagePage(state.currentPage + 1),
    });

    const lastButton = buildPagerButton({
        label: "Last",
        title: "Go to last image set",
        disabled: state.currentPage >= state.totalPages || state.isLoading,
        onClick: () => loadThumbnailImagePage(state.totalPages),
    });

    controls.appendChild(nextButton);
    controls.appendChild(lastButton);

    return controls;
}

function buildPagerButton({
    label,
    title,
    disabled = false,
    active = false,
    onClick,
}) {
    const button = document.createElement("button");

    button.type = "button";
    button.className = active
        ? "emtac-thumbnail-load-more emtac-thumbnail-page-button is-active"
        : "emtac-thumbnail-load-more emtac-thumbnail-page-button";

    button.textContent = label;
    button.title = title || label;
    button.disabled = Boolean(disabled || active);

    if (active) {
        button.setAttribute("aria-current", "page");
    }

    button.addEventListener("click", event => {
        event.preventDefault();
        event.stopPropagation();

        if (button.disabled || typeof onClick !== "function") {
            return;
        }

        onClick();
    });

    return button;
}

function buildPagerEllipsis() {
    const ellipsis = document.createElement("span");

    ellipsis.className = "emtac-thumbnail-page-ellipsis";
    ellipsis.textContent = "…";
    ellipsis.setAttribute("aria-hidden", "true");

    return ellipsis;
}

function getPagerWindow(currentPage, totalPages, windowSize) {
    const total = Math.max(1, normalizePositiveInteger(totalPages, 1));
    const current = clampPage(currentPage, total);
    const size = Math.max(3, normalizePositiveInteger(windowSize, EMTAC_THUMBNAIL_PAGE_WINDOW));

    if (total <= size + 2) {
        return Array.from({ length: total }, (_, index) => index + 1);
    }

    const middleSlots = Math.max(1, size - 2);
    const half = Math.floor(middleSlots / 2);

    let start = Math.max(2, current - half);
    let end = Math.min(total - 1, start + middleSlots - 1);

    if (end - start + 1 < middleSlots) {
        start = Math.max(2, end - middleSlots + 1);
    }

    const pages = [1];

    for (let page = start; page <= end; page += 1) {
        pages.push(page);
    }

    pages.push(total);

    return pages.filter((page, index, array) => array.indexOf(page) === index);
}

function buildPageRangeLabel(page, pageSize, totalCount) {
    const range = getPageRange(page, pageSize, totalCount);

    if (range.start <= 0 || range.end <= 0) {
        return "0";
    }

    return `${range.start}-${range.end}`;
}


// ------------------------------------------------------------
// Page navigation
// ------------------------------------------------------------

async function loadThumbnailImagePage(page) {
    const state = thumbnailPaginationState;

    if (!state || state.isLoading) {
        return;
    }

    const requestedPage = clampPage(page, state.totalPages);

    if (requestedPage === state.currentPage) {
        renderThumbnailPanel();
        return;
    }

    state.lastError = "";

    if (state.pageCache.has(requestedPage)) {
        const cachedPage = state.pageCache.get(requestedPage);

        state.currentPage = requestedPage;
        state.currentImages = Array.isArray(cachedPage.images)
            ? cachedPage.images
            : [];

        window.EMTAC_THUMBNAIL_PAGINATION_STATE = state;
        renderThumbnailPanel();
        return;
    }

    if (canPageLocally(state)) {
        const localImages = getLocalPageImages(state, requestedPage);

        state.pageCache.set(requestedPage, {
            images: localImages,
            page: requestedPage,
        });

        state.currentPage = requestedPage;
        state.currentImages = localImages;

        window.EMTAC_THUMBNAIL_PAGINATION_STATE = state;
        renderThumbnailPanel();
        return;
    }

    if (!state.endpoint || !(state.requestId || state.positionId || state.completeDocumentId)) {
        state.lastError = "Unable to load that image set.";
        renderThumbnailPanel();
        return;
    }

    state.isLoading = true;
    renderThumbnailPanel();

    try {
        const pageData = await fetchThumbnailImagePage(state, requestedPage);

        const normalized = normalizeThumbnailInput(pageData, {
            pageSize: state.pageSize,
            endpoint: state.endpoint,
        });

        const pageImages = Array.isArray(normalized.images)
            ? normalized.images
            : [];

        state.totalCount = Math.max(
            normalizeNonNegativeInteger(normalized.totalCount, 0),
            state.totalCount || 0,
            pageImages.length
        );

        state.totalPages = calculateTotalPages(
            state.totalCount,
            state.pageSize,
            pageImages.length
        );

        const safeRequestedPage = clampPage(requestedPage, state.totalPages);

        state.pageCache.set(safeRequestedPage, {
            images: pageImages,
            page: safeRequestedPage,
        });

        state.currentPage = safeRequestedPage;
        state.currentImages = pageImages;

    } catch (error) {
        console.error("[loadThumbnailImagePage] Failed to load image page:", error);

        state.lastError = "Unable to load that image set.";
    } finally {
        state.isLoading = false;
        window.EMTAC_THUMBNAIL_PAGINATION_STATE = state;
        renderThumbnailPanel();
    }
}

async function loadNextThumbnailImagePage() {
    const state = thumbnailPaginationState;

    if (!state) {
        return;
    }

    return loadThumbnailImagePage(state.currentPage + 1);
}

// Backward-compatible alias for older button handlers.
async function loadMoreThumbnailImages() {
    return loadNextThumbnailImagePage();
}

async function fetchThumbnailImagePage(state, page) {
    const url = new URL(state.endpoint, window.location.origin);

    if (state.requestId) {
        url.searchParams.set("request_id", state.requestId);
    }

    if (state.positionId) {
        url.searchParams.set("position_id", state.positionId);
    }

    if (state.completeDocumentId) {
        url.searchParams.set("complete_document_id", state.completeDocumentId);
    }

    url.searchParams.set("page", String(page));
    url.searchParams.set("page_size", String(state.pageSize));

    console.log("[EMTAC] Loading image page:", {
        url: url.pathname + url.search,
        request_id: state.requestId,
        position_id: state.positionId,
        complete_document_id: state.completeDocumentId,
        page,
        page_size: state.pageSize,
    });

    const response = await fetch(url.pathname + url.search, {
        method: "GET",
        headers: {
            "Accept": "application/json",
        },
        credentials: "same-origin",
    });

    const data = await safeThumbnailJsonResponse(response);

    if (!response.ok || data.status === "error" || data.payload_status === "error") {
        throw new Error(data.message || "Image page request failed.");
    }

    return data;
}

async function safeThumbnailJsonResponse(response) {
    try {
        return await response.json();
    } catch (error) {
        console.error("[safeThumbnailJsonResponse] Invalid JSON response:", error);

        return {
            status: "error",
            message: "Invalid image pagination response.",
            images: [],
        };
    }
}


// ------------------------------------------------------------
// Open image viewer
// ------------------------------------------------------------

function openThumbnailImageViewer(title, src) {
    if (typeof window.openImageViewerInPage === "function") {
        window.openImageViewerInPage(title, src);
        return;
    }

    if (typeof window.openImageViewer === "function") {
        window.openImageViewer(title, src);
        return;
    }

    console.error("[openThumbnailImageViewer] No image viewer function available.", {
        title,
        src
    });
}


// ------------------------------------------------------------
// Image URL Resolver
// ------------------------------------------------------------

function resolveImageUrl(img) {
    if (!img || typeof img !== "object") {
        return "";
    }

    // Best route: build from image ID.
    if (img.id !== undefined && img.id !== null && String(img.id).trim() !== "") {
        const numericId = Number(img.id);

        if (Number.isInteger(numericId) && numericId > 0) {
            return `/serve_image/${numericId}`;
        }
    }

    const candidates = [
        img.src,
        img.url,
        img.href,
        img.image_url,
        img.file_url,
        img.file_path
    ];

    for (const candidate of candidates) {
        if (typeof candidate !== "string") {
            continue;
        }

        const value = candidate.trim();

        if (!value) {
            continue;
        }

        // Already correct.
        if (value.startsWith("/serve_image/")) {
            return value;
        }

        // Convert older route convention.
        if (value.startsWith("/images/")) {
            const maybeId = value.split("/").pop();

            if (/^\d+$/.test(maybeId)) {
                return `/serve_image/${maybeId}`;
            }
        }

        // Absolute same-host route support.
        try {
            const parsed = new URL(value, window.location.origin);

            if (parsed.pathname.startsWith("/serve_image/")) {
                return parsed.pathname + parsed.search;
            }

            if (parsed.pathname.startsWith("/images/")) {
                const maybeId = parsed.pathname.split("/").pop();

                if (/^\d+$/.test(maybeId)) {
                    return `/serve_image/${maybeId}`;
                }
            }
        } catch (err) {
            // Ignore invalid URL values.
        }
    }

    return "";
}


// ------------------------------------------------------------
// Global exports
// ------------------------------------------------------------

window.displayThumbnails = displayThumbnails;
window.resolveImageUrl = resolveImageUrl;
window.openThumbnailImageViewer = openThumbnailImageViewer;
window.loadThumbnailImagePage = loadThumbnailImagePage;
window.loadNextThumbnailImagePage = loadNextThumbnailImagePage;
window.loadMoreThumbnailImages = loadMoreThumbnailImages;
window.EMTAC_THUMBNAIL_DEFAULT_PAGE_SIZE = EMTAC_THUMBNAIL_DEFAULT_PAGE_SIZE;
window.EMTAC_THUMBNAIL_MAX_PAGE_SIZE = EMTAC_THUMBNAIL_MAX_PAGE_SIZE;