// ASAM Browser Extension - Background Service Worker

class ASAMBackgroundService {
    constructor() {
        this.config = {
            statusEndpoint: 'http://localhost:8888/api/status',
            checkInterval: 30000 // 30 seconds
        };

        this.connectionStatus = 'unknown';
        this.lastActivity = null;

        this.init();
    }

    init() {
        console.log('ASAM Background Service initialized');

        // Set up message listeners
        chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
            this.handleMessage(message, sender, sendResponse);
            return true; // Keep message channel open for async response
        });

        // Check ASAM service status periodically
        this.startStatusChecking();

        // Set up extension lifecycle handlers
        chrome.runtime.onStartup.addListener(() => {
            console.log('ASAM: Extension started');
        });

        chrome.runtime.onInstalled.addListener((details) => {
            console.log('ASAM: Extension installed/updated', details.reason);
            this.checkASAMConnection();
        });
    }

    handleMessage(message, sender, sendResponse) {
        switch (message.type) {
            case 'content_sent':
                this.handleContentSent(message.data, sender);
                sendResponse({ status: 'acknowledged' });
                break;

            case 'get_status':
                sendResponse({
                    connectionStatus: this.connectionStatus,
                    lastActivity: this.lastActivity
                });
                break;

            case 'check_connection':
                this.checkASAMConnection().then(status => {
                    sendResponse({ connectionStatus: status });
                });
                break;

            default:
                sendResponse({ error: 'Unknown message type' });
        }
    }

    handleContentSent(data, sender) {
        this.lastActivity = {
            timestamp: new Date().toISOString(),
            url: data.url,
            title: data.title,
            tabId: sender.tab?.id
        };

        console.log('ASAM: Content activity recorded', data);

        // Update badge to show activity
        this.updateExtensionBadge('✓', '#4CAF50');

        // Clear badge after a few seconds
        setTimeout(() => {
            this.updateExtensionBadge('', '#757575');
        }, 2000);
    }

    async checkASAMConnection() {
        try {
            const response = await fetch(this.config.statusEndpoint, {
                method: 'GET',
                headers: {
                    'Content-Type': 'application/json'
                }
            });

            if (response.ok) {
                const data = await response.json();
                this.connectionStatus = 'connected';
                console.log('ASAM: Service connected', data);

                this.updateExtensionBadge('', '#4CAF50');
                return 'connected';
            } else {
                this.connectionStatus = 'error';
                this.updateExtensionBadge('!', '#FF5722');
                return 'error';
            }
        } catch (error) {
            console.error('ASAM: Connection check failed', error);
            this.connectionStatus = 'disconnected';
            this.updateExtensionBadge('×', '#757575');
            return 'disconnected';
        }
    }

    startStatusChecking() {
        // Initial check
        this.checkASAMConnection();

        // Periodic checks
        setInterval(() => {
            this.checkASAMConnection();
        }, this.config.checkInterval);
    }

    updateExtensionBadge(text, color) {
        if (chrome.action) {
            chrome.action.setBadgeText({ text: text });
            chrome.action.setBadgeBackgroundColor({ color: color });
        }
    }

    // Handle extension lifecycle
    onExtensionSuspend() {
        console.log('ASAM: Extension suspended');
    }

    onExtensionResume() {
        console.log('ASAM: Extension resumed');
        this.checkASAMConnection();
    }
}

// Initialize background service
new ASAMBackgroundService();
