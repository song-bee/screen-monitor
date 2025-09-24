// ASAM Browser Extension - Content Script
// Extracts page content and sends to ASAM service

class ASAMContentExtractor {
    constructor() {
        this.config = {
            apiEndpoint: 'http://localhost:8888/api/content',
            apiKey: 'asam-browser-integration',
            updateInterval: 5000,
            maxContentLength: 10000
        };

        this.lastContent = '';
        this.isActive = false;
        this.tabId = null;

        this.init();
    }

    async init() {
        // Get tab ID
        this.tabId = await this.getTabId();

        // Start monitoring
        this.startMonitoring();

        console.log('ASAM Content Extractor initialized for tab:', this.tabId);
    }

    async getTabId() {
        return new Promise((resolve) => {
            if (chrome?.tabs) {
                chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
                    resolve(tabs[0]?.id?.toString() || 'unknown');
                });
            } else {
                resolve('content_script_' + Date.now());
            }
        });
    }

    startMonitoring() {
        // Initial extraction
        this.extractAndSend();

        // Set up periodic extraction
        setInterval(() => {
            this.extractAndSend();
        }, this.config.updateInterval);

        // Monitor for dynamic content changes
        this.setupDynamicMonitoring();
    }

    setupDynamicMonitoring() {
        // Monitor DOM changes
        const observer = new MutationObserver((mutations) => {
            let significantChange = false;

            mutations.forEach((mutation) => {
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                    significantChange = true;
                }
            });

            if (significantChange) {
                // Debounce rapid changes
                clearTimeout(this.dynamicUpdateTimeout);
                this.dynamicUpdateTimeout = setTimeout(() => {
                    this.extractAndSend();
                }, 1000);
            }
        });

        observer.observe(document.body, {
            childList: true,
            subtree: true
        });

        // Monitor URL changes (SPA navigation)
        let lastUrl = window.location.href;
        setInterval(() => {
            if (window.location.href !== lastUrl) {
                lastUrl = window.location.href;
                setTimeout(() => this.extractAndSend(), 500); // Wait for content to load
            }
        }, 1000);
    }

    extractContent() {
        const content = {
            url: window.location.href,
            title: document.title || 'Untitled',
            content: '',
            metadata: {
                domain: window.location.hostname,
                pathname: window.location.pathname,
                timestamp: new Date().toISOString(),
                contentType: this.detectContentType()
            }
        };

        // Extract main content
        const textContent = this.extractTextContent();
        content.content = textContent.substring(0, this.config.maxContentLength);

        // Add metadata based on content analysis
        content.metadata = {
            ...content.metadata,
            ...this.analyzePageMetadata()
        };

        return content;
    }

    extractTextContent() {
        // Remove scripts, styles, and hidden elements
        const elementsToRemove = document.querySelectorAll('script, style, noscript, [style*="display: none"], [style*="display:none"]');
        const clonedDoc = document.cloneNode(true);

        elementsToRemove.forEach(el => {
            const clonedEl = clonedDoc.querySelector(el.tagName.toLowerCase());
            if (clonedEl) clonedEl.remove();
        });

        // Priority content selectors
        const contentSelectors = [
            'main',
            'article',
            '.content',
            '#content',
            '.post-content',
            '.entry-content',
            '[role="main"]'
        ];

        let mainContent = '';

        // Try to find main content area
        for (const selector of contentSelectors) {
            const element = document.querySelector(selector);
            if (element) {
                mainContent = element.innerText || element.textContent || '';
                if (mainContent.trim().length > 100) {
                    break;
                }
            }
        }

        // Fallback to body content if no main content found
        if (!mainContent.trim()) {
            mainContent = document.body.innerText || document.body.textContent || '';
        }

        // Clean up text
        return mainContent
            .replace(/\s+/g, ' ')
            .replace(/\n\s*\n/g, '\n')
            .trim();
    }

    detectContentType() {
        const url = window.location.href.toLowerCase();
        const title = document.title.toLowerCase();
        const domain = window.location.hostname.toLowerCase();

        // Social media detection
        if (domain.includes('facebook') || domain.includes('twitter') ||
            domain.includes('instagram') || domain.includes('tiktok') ||
            domain.includes('reddit') || domain.includes('linkedin')) {
            return 'social_media';
        }

        // Video streaming detection
        if (domain.includes('youtube') || domain.includes('netflix') ||
            domain.includes('hulu') || domain.includes('twitch') ||
            document.querySelector('video')) {
            return 'video_streaming';
        }

        // Gaming detection
        if (domain.includes('steam') || domain.includes('epic') ||
            title.includes('game') || title.includes('gaming') ||
            document.querySelector('canvas[width][height]')) {
            return 'gaming';
        }

        // News detection
        if (domain.includes('news') || domain.includes('bbc') ||
            domain.includes('cnn') || domain.includes('reuters') ||
            document.querySelector('article')) {
            return 'news';
        }

        // Shopping detection
        if (domain.includes('amazon') || domain.includes('shop') ||
            domain.includes('store') || url.includes('cart') ||
            document.querySelector('.price, .cart, .buy-now')) {
            return 'shopping';
        }

        return 'general';
    }

    analyzePageMetadata() {
        const metadata = {};

        // Check for video elements
        const videos = document.querySelectorAll('video');
        if (videos.length > 0) {
            metadata.hasVideo = true;
            metadata.videoCount = videos.length;
        }

        // Check for game-like elements
        const canvases = document.querySelectorAll('canvas');
        if (canvases.length > 0) {
            metadata.hasCanvas = true;
            metadata.canvasCount = canvases.length;
        }

        // Check for social media indicators
        const socialButtons = document.querySelectorAll('[class*="share"], [class*="like"], [class*="follow"]');
        if (socialButtons.length > 0) {
            metadata.hasSocialElements = true;
        }

        // Check page structure
        metadata.linkCount = document.querySelectorAll('a').length;
        metadata.imageCount = document.querySelectorAll('img').length;
        metadata.wordCount = this.extractTextContent().split(/\s+/).length;

        return metadata;
    }

    async extractAndSend() {
        try {
            const content = this.extractContent();

            // Skip if content hasn't changed significantly
            if (this.isSimilarContent(content.content)) {
                return;
            }

            this.lastContent = content.content;

            // Send to ASAM service
            await this.sendToASAM(content);

        } catch (error) {
            console.error('ASAM: Error extracting/sending content:', error);
        }
    }

    isSimilarContent(newContent) {
        if (!this.lastContent) return false;

        // Simple similarity check - could be improved with more sophisticated algorithms
        const similarity = this.calculateSimilarity(this.lastContent, newContent);
        return similarity > 0.8; // 80% similar
    }

    calculateSimilarity(str1, str2) {
        const maxLength = Math.max(str1.length, str2.length);
        if (maxLength === 0) return 1;

        const distance = this.levenshteinDistance(str1, str2);
        return 1 - distance / maxLength;
    }

    levenshteinDistance(str1, str2) {
        const matrix = Array(str2.length + 1).fill(null).map(() => Array(str1.length + 1).fill(null));

        for (let i = 0; i <= str1.length; i++) matrix[0][i] = i;
        for (let j = 0; j <= str2.length; j++) matrix[j][0] = j;

        for (let j = 1; j <= str2.length; j++) {
            for (let i = 1; i <= str1.length; i++) {
                const indicator = str1[i - 1] === str2[j - 1] ? 0 : 1;
                matrix[j][i] = Math.min(
                    matrix[j][i - 1] + 1,
                    matrix[j - 1][i] + 1,
                    matrix[j - 1][i - 1] + indicator
                );
            }
        }

        return matrix[str2.length][str1.length];
    }

    async sendToASAM(content) {
        const payload = {
            url: content.url,
            title: content.title,
            content: content.content,
            tabId: this.tabId,
            browserType: 'chrome',
            metadata: content.metadata
        };

        try {
            const response = await fetch(this.config.apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'X-API-Key': this.config.apiKey
                },
                body: JSON.stringify(payload)
            });

            if (response.ok) {
                const result = await response.json();
                console.log('ASAM: Content sent successfully', result);

                // Send message to popup about successful transmission
                this.notifyExtension('content_sent', {
                    url: content.url,
                    title: content.title,
                    timestamp: result.timestamp
                });
            } else {
                console.error('ASAM: Failed to send content', response.status, response.statusText);
            }
        } catch (error) {
            console.error('ASAM: Network error sending content:', error);
        }
    }

    notifyExtension(type, data) {
        if (chrome?.runtime?.sendMessage) {
            chrome.runtime.sendMessage({
                type: type,
                data: data,
                tabId: this.tabId
            }).catch(err => {
                // Extension context might be invalidated, ignore
            });
        }
    }
}

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
        new ASAMContentExtractor();
    });
} else {
    new ASAMContentExtractor();
}
