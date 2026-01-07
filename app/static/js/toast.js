/**
 * Toast Notification System
 * Provides easy-to-use toast notifications with 5-second default duration
 */

const ToastManager = {
    /**
     * Show a success toast
     * @param {string} message - The message to display
     * @param {number} duration - Duration in milliseconds (default: 5000ms)
     */
    success: function(message, duration = 5000) {
        showToast(message, 'success', duration);
    },

    /**
     * Show an error toast
     * @param {string} message - The message to display
     * @param {number} duration - Duration in milliseconds (default: 5000ms)
     */
    error: function(message, duration = 5000) {
        showToast(message, 'error', duration);
    },

    /**
     * Show a warning toast
     * @param {string} message - The message to display
     * @param {number} duration - Duration in milliseconds (default: 5000ms)
     */
    warning: function(message, duration = 5000) {
        showToast(message, 'warning', duration);
    },

    /**
     * Show an info toast
     * @param {string} message - The message to display
     * @param {number} duration - Duration in milliseconds (default: 5000ms)
     */
    info: function(message, duration = 5000) {
        showToast(message, 'info', duration);
    },

    /**
     * Show a custom toast
     * @param {string} message - The message to display
     * @param {string} type - The toast type: 'success', 'error', 'warning', 'info'
     * @param {number} duration - Duration in milliseconds (default: 5000ms)
     */
    show: function(message, type = 'info', duration = 5000) {
        showToast(message, type, duration);
    }
};

// Listen for HTMX events to automatically show toasts from response headers
document.addEventListener('htmx:afterSwap', function(event) {
    const xhr = event.detail.xhr;

    // Check for toast headers in response
    const toastMessage = xhr.getResponseHeader('X-Toast-Message');
    const toastType = xhr.getResponseHeader('X-Toast-Type') || 'info';
    const toastDuration = xhr.getResponseHeader('X-Toast-Duration') ? parseInt(xhr.getResponseHeader('X-Toast-Duration')) : 5000;

    if (toastMessage) {
        showToast(toastMessage, toastType, toastDuration);
    }
});

// Alternative: Listen for custom events
document.addEventListener('showToast', function(event) {
    const { message, type = 'info', duration = 5000 } = event.detail;
    showToast(message, type, duration);
});

// Expose globally for inline HTML onclick handlers
window.Toast = ToastManager;
