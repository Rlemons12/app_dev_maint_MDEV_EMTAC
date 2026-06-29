/* static/js/module_template/banner.js
==================================
EMTAC BANNER
Standalone banner behavior.
Handles:
- syncing body padding with banner height
- keeping banner height responsive
- safe initialization
==================================
*/

console.log("[BANNER] banner.js loaded");

(function () {
    function initBanner() {
        const chrome = document.querySelector(".app-top-chrome");
        const banner = chrome ? chrome.querySelector(".banner") : null;
        const root = document.documentElement;

        if (!chrome || !banner) {
            console.warn("[BANNER] Banner elements not found");
            return;
        }

        if (chrome.dataset.bannerInitialized === "true") {
            console.log("[BANNER] Already initialized, skipping duplicate bind.");
            return;
        }

        chrome.dataset.bannerInitialized = "true";

        let resizeTimeout = null;

        function syncBannerHeight() {
            const bannerHeight = banner.offsetHeight;

            if (!bannerHeight) {
                return;
            }

            root.style.setProperty("--banner-height", `${bannerHeight}px`);
            root.style.setProperty("--app-banner-height", `${bannerHeight}px`);
            root.style.setProperty("--app-top-banner-offset", `${bannerHeight}px`);

            document.body.style.paddingTop = `${bannerHeight}px`;
        }

        function queueBannerSync() {
            window.clearTimeout(resizeTimeout);
            resizeTimeout = window.setTimeout(syncBannerHeight, 10);
        }

        syncBannerHeight();

        window.addEventListener("resize", queueBannerSync);
        window.addEventListener("load", syncBannerHeight);

        if ("ResizeObserver" in window) {
            const resizeObserver = new ResizeObserver(function () {
                queueBannerSync();
            });
            resizeObserver.observe(banner);
        }

        console.log("[BANNER] initialized");
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initBanner, { once: true });
    } else {
        initBanner();
    }
})();