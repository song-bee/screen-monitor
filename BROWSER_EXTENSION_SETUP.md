# ASAM Browser Extension Setup Guide

This guide will help you test the ASAM browser extension integration.

## Quick Start

### 1. Start ASAM Browser Integration Server

```bash
# Activate virtual environment
source venv/bin/activate

# Run the browser integration test
python test_browser_integration.py
```

Choose option **1** (Browser integration only) for lightweight testing.

The server will start at `http://localhost:8888`

### 2. Install Browser Extension

1. **Open Chrome** and navigate to `chrome://extensions/`

2. **Enable Developer Mode** (toggle switch in top-right corner)

3. **Click "Load unpacked"** and select the `browser-extension` folder:
   ```
   /Users/song/workspace/projects/screen-monitor/browser-extension/
   ```

4. The **ASAM Monitor** extension should appear in your extensions list

### 3. Test the Integration

1. **Click the ASAM extension icon** in Chrome's toolbar (🔍)

2. The popup should show:
   - Connection status
   - Current page info
   - Test buttons

3. **Click "Send Test Data"** to verify connectivity

4. **Browse different websites** and watch the console output in your terminal

## What You'll See

### In the Terminal
```
📨 Received content from chrome:
   📄 Title: Example Website
   🔗 URL: https://example.com
   📝 Content length: 1250 characters
   🏷️  Tab ID: 12345
   📊 Metadata: {
     "domain": "example.com",
     "contentType": "general"
   }
```

### In the Extension Popup
- **Green status**: Connected to ASAM service
- **Red status**: Service offline or unreachable
- **Last activity**: Timestamp of recent content transmission

## Testing Different Content Types

The extension automatically detects and categorizes:

- **🎮 Gaming**: Steam, Epic Games, Canvas games
- **📺 Video Streaming**: YouTube, Netflix, Twitch
- **📱 Social Media**: Facebook, Twitter, Instagram, Reddit
- **📰 News**: News websites with articles
- **🛒 Shopping**: Amazon, online stores

## Advanced Testing

### Full ASAM Integration
Run the test with option **2** to integrate with the full ASAM detection pipeline:

```bash
python test_browser_integration.py
# Choose option 2
```

This will:
- Start the complete ASAM service
- Process browser content through LLM analysis
- Apply detection rules and confidence scoring
- Show full analysis results

### API Testing
Test the API directly with curl:

```bash
# Test status endpoint
curl http://localhost:8888/api/status

# Send test content
curl -X POST http://localhost:8888/api/content \
  -H "Content-Type: application/json" \
  -H "X-API-Key: asam-browser-integration" \
  -d '{
    "url": "https://example.com",
    "title": "Test Page",
    "content": "This is test content for analysis",
    "tabId": "test123",
    "browserType": "chrome"
  }'
```

## Troubleshooting

### Extension Not Loading
- Ensure Developer Mode is enabled
- Check that all extension files are present
- Look for errors in Chrome's Extensions page

### Connection Issues
- Verify ASAM service is running on port 8888
- Check firewall settings
- Ensure API key matches: `asam-browser-integration`

### No Content Being Sent
- Check browser console for JavaScript errors
- Verify the extension has permissions for the current site
- Try refreshing the page after installing

## Browser Extension Files

```
browser-extension/
├── manifest.json       # Extension configuration
├── content.js          # Content extraction script
├── background.js       # Background service worker
├── popup.html          # Extension popup UI
├── popup.js            # Popup functionality
└── icon*.png           # Extension icons
```

## API Endpoints

- **POST** `/api/content` - Receive browser content
- **GET** `/api/status` - Check service status
- **OPTIONS** `/api/content` - CORS preflight

## Next Steps

1. **Customize Content Extraction**: Modify `content.js` to extract specific data
2. **Add More Content Types**: Extend detection logic in `detectContentType()`
3. **Improve UI**: Enhance the popup interface in `popup.html`
4. **Integration**: Connect browser data with ASAM's analysis pipeline

## Security Notes

- Extension only sends data to `localhost:8888`
- API key required for all requests
- Content is processed locally (no cloud services)
- Respects browser permissions and security policies
