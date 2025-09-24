// ASAM Browser Extension - Popup UI

class ASAMPopup {
    constructor() {
        this.elements = {
            statusSection: document.getElementById('status-section'),
            statusText: document.getElementById('status-text'),
            statusDetail: document.getElementById('status-detail'),
            connectionStatus: document.getElementById('connection-status'),
            serviceUrl: document.getElementById('service-url'),
            lastActivity: document.getElementById('last-activity'),
            pageTitle: document.getElementById('page-title'),
            pageUrl: document.getElementById('page-url'),
            monitoringStatus: document.getElementById('monitoring-status'),
            refreshBtn: document.getElementById('refresh-btn'),
            testBtn: document.getElementById('test-btn'),
            loading: document.getElementById('loading')
        };

        this.init();
    }

    async init() {
        // Set up event listeners
        this.elements.refreshBtn.addEventListener('click', () => this.refreshConnection());
        this.elements.testBtn.addEventListener('click', () => this.sendTestData());

        // Load initial data
        await this.loadCurrentPageInfo();
        await this.loadConnectionStatus();

        console.log('ASAM Popup initialized');
    }

    async loadCurrentPageInfo() {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

            if (tab) {
                this.elements.pageTitle.textContent = tab.title || 'Untitled';
                this.elements.pageUrl.textContent = this.truncateUrl(tab.url || '');
            }
        } catch (error) {
            console.error('Error loading page info:', error);
            this.elements.pageTitle.textContent = 'Error';
            this.elements.pageUrl.textContent = 'Could not load page info';
        }
    }

    async loadConnectionStatus() {
        try {
            // Get status from background script
            const response = await chrome.runtime.sendMessage({ type: 'get_status' });

            this.updateConnectionStatus(response.connectionStatus);

            if (response.lastActivity) {
                this.updateLastActivity(response.lastActivity);
            }
        } catch (error) {
            console.error('Error loading status:', error);
            this.updateConnectionStatus('error');
        }
    }

    updateConnectionStatus(status) {
        // Remove all status classes
        this.elements.statusSection.classList.remove('connected', 'disconnected', 'error');

        let statusText, statusDetail;

        switch (status) {
            case 'connected':
                this.elements.statusSection.classList.add('connected');
                statusText = 'Connected to ASAM';
                statusDetail = 'Content monitoring active';
                this.elements.connectionStatus.textContent = 'Connected';
                break;

            case 'disconnected':
                this.elements.statusSection.classList.add('disconnected');
                statusText = 'ASAM Service Offline';
                statusDetail = 'Start ASAM service to begin monitoring';
                this.elements.connectionStatus.textContent = 'Disconnected';
                break;

            case 'error':
                this.elements.statusSection.classList.add('error');
                statusText = 'Connection Error';
                statusDetail = 'Check ASAM service configuration';
                this.elements.connectionStatus.textContent = 'Error';
                break;

            default:
                this.elements.statusSection.classList.add('disconnected');
                statusText = 'Unknown Status';
                statusDetail = 'Checking connection...';
                this.elements.connectionStatus.textContent = 'Unknown';
        }

        this.elements.statusText.textContent = statusText;
        this.elements.statusDetail.textContent = statusDetail;
    }

    updateLastActivity(activity) {
        if (activity && activity.timestamp) {
            const time = new Date(activity.timestamp);
            const timeStr = time.toLocaleTimeString([], {
                hour: '2-digit',
                minute: '2-digit'
            });

            this.elements.lastActivity.textContent = timeStr;
        } else {
            this.elements.lastActivity.textContent = 'None';
        }
    }

    async refreshConnection() {
        this.showLoading(true);
        this.elements.refreshBtn.disabled = true;
        this.elements.refreshBtn.textContent = 'Checking...';

        try {
            // Request fresh status check from background script
            const response = await chrome.runtime.sendMessage({ type: 'check_connection' });

            this.updateConnectionStatus(response.connectionStatus);

            // Also refresh page info
            await this.loadCurrentPageInfo();

        } catch (error) {
            console.error('Error refreshing connection:', error);
            this.updateConnectionStatus('error');
        }

        this.showLoading(false);
        this.elements.refreshBtn.disabled = false;
        this.elements.refreshBtn.textContent = 'Refresh Connection';
    }

    async sendTestData() {
        this.elements.testBtn.disabled = true;
        this.elements.testBtn.textContent = 'Sending...';

        try {
            const testData = {
                url: window.location.href || 'chrome://newtab/',
                title: 'ASAM Extension Test',
                content: 'This is a test message from the ASAM browser extension to verify connectivity and functionality.',
                tabId: 'test_tab',
                browserType: 'chrome',
                metadata: {
                    source: 'extension_test',
                    timestamp: new Date().toISOString(),
                    contentType: 'test'
                }
            };

            const response = await fetch('http://localhost:8888/api/content', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': 'asam-browser-integration'
                },
                body: JSON.stringify(testData)
            });

            if (response.ok) {
                const result = await response.json();
                console.log('Test data sent successfully:', result);

                // Show success feedback
                this.elements.testBtn.textContent = '✓ Test Sent';
                this.elements.testBtn.style.backgroundColor = '#4CAF50';

                // Update last activity
                this.updateLastActivity({ timestamp: result.timestamp });

                setTimeout(() => {
                    this.elements.testBtn.textContent = 'Send Test Data';
                    this.elements.testBtn.style.backgroundColor = '';
                }, 2000);

            } else {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
        } catch (error) {
            console.error('Error sending test data:', error);

            // Show error feedback
            this.elements.testBtn.textContent = '✗ Test Failed';
            this.elements.testBtn.style.backgroundColor = '#F44336';

            setTimeout(() => {
                this.elements.testBtn.textContent = 'Send Test Data';
                this.elements.testBtn.style.backgroundColor = '';
            }, 2000);
        }

        this.elements.testBtn.disabled = false;
    }

    showLoading(show) {
        if (show) {
            this.elements.loading.style.display = 'block';
        } else {
            this.elements.loading.style.display = 'none';
        }
    }

    truncateUrl(url) {
        if (url.length > 40) {
            return url.substring(0, 37) + '...';
        }
        return url;
    }
}

// Initialize popup when DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    new ASAMPopup();
});