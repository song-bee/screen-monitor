"""
Browser Extension Integration

Framework for receiving content analysis data from browser extensions.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

from aiohttp import web

from ..core.detection.types import TextContent


@dataclass
class BrowserContent:
    """Content extracted from browser extension"""

    url: str
    title: str
    text_content: str
    timestamp: datetime
    tab_id: str
    browser_type: str
    metadata: Optional[Dict[str, Any]] = None


class BrowserIntegrationServer:
    """HTTP server for receiving browser extension data"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Server configuration
        self.host = self.config.get("host", "localhost")
        self.port = self.config.get("port", 8888)
        self.api_key = self.config.get("api_key", "asam-browser-integration")

        # State management
        self.app: Optional[web.Application] = None
        self.server: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None
        self.running = False

        # Content storage
        self.recent_content: List[BrowserContent] = []
        self.max_content_history = self.config.get("max_content_history", 50)

        # Content callbacks
        self.content_callbacks: List[callable] = []

    async def start(self) -> bool:
        """Start the browser integration server"""
        try:
            if self.running:
                self.logger.warning("Browser integration server already running")
                return True

            # Create web application
            self.app = web.Application()
            self._setup_routes()

            # Start server
            self.server = web.AppRunner(self.app)
            await self.server.setup()

            self.site = web.TCPSite(self.server, self.host, self.port)
            await self.site.start()

            self.running = True
            self.logger.info(
                f"Browser integration server started on {self.host}:{self.port}"
            )
            return True

        except Exception as e:
            self.logger.error(f"Failed to start browser integration server: {e}")
            return False

    async def stop(self) -> None:
        """Stop the browser integration server"""
        try:
            if not self.running:
                return

            self.running = False

            if self.site:
                await self.site.stop()
                self.site = None

            if self.server:
                await self.server.cleanup()
                self.server = None

            self.app = None
            self.logger.info("Browser integration server stopped")

        except Exception as e:
            self.logger.error(f"Error stopping browser integration server: {e}")

    def _setup_routes(self) -> None:
        """Setup HTTP routes for browser extension communication"""
        self.app.router.add_post("/api/content", self._handle_content_submission)
        self.app.router.add_get("/api/status", self._handle_status_check)
        self.app.router.add_options("/api/content", self._handle_preflight)

        # Add CORS middleware
        self.app.middlewares.append(self._cors_middleware)

    @web.middleware
    async def _cors_middleware(self, request: web.Request, handler) -> web.Response:
        """Handle CORS for browser extensions"""
        # Handle preflight requests
        if request.method == "OPTIONS":
            response = web.Response()
        else:
            try:
                response = await handler(request)
            except Exception as e:
                self.logger.error(f"Request handler error: {e}")
                response = web.json_response(
                    {"error": "Internal server error"}, status=500
                )

        # Add CORS headers
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = (
            "Content-Type, Authorization, X-API-Key"
        )

        return response

    async def _handle_content_submission(self, request: web.Request) -> web.Response:
        """Handle content submission from browser extension"""
        try:
            # Verify API key
            api_key = request.headers.get("X-API-Key")
            if api_key != self.api_key:
                return web.json_response({"error": "Invalid API key"}, status=401)

            # Parse request data
            data = await request.json()

            # Validate required fields
            required_fields = ["url", "title", "content", "tabId", "browserType"]
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return web.json_response(
                    {"error": f"Missing required fields: {missing_fields}"}, status=400
                )

            # Create browser content object
            browser_content = BrowserContent(
                url=data["url"],
                title=data["title"],
                text_content=data["content"],
                timestamp=datetime.now(),
                tab_id=data["tabId"],
                browser_type=data["browserType"],
                metadata=data.get("metadata", {}),
            )

            # Store content
            await self._store_content(browser_content)

            # Notify callbacks
            await self._notify_content_callbacks(browser_content)

            return web.json_response(
                {
                    "status": "success",
                    "message": "Content received",
                    "timestamp": browser_content.timestamp.isoformat(),
                }
            )

        except json.JSONDecodeError:
            return web.json_response({"error": "Invalid JSON data"}, status=400)
        except Exception as e:
            self.logger.error(f"Error handling content submission: {e}")
            return web.json_response({"error": "Internal server error"}, status=500)

    async def _handle_status_check(self, request: web.Request) -> web.Response:
        """Handle status check requests"""
        return web.json_response(
            {
                "status": "running",
                "version": "1.0.0",
                "content_count": len(self.recent_content),
                "callbacks_registered": len(self.content_callbacks),
            }
        )

    async def _handle_preflight(self, request: web.Request) -> web.Response:
        """Handle CORS preflight requests"""
        return web.Response(status=204)

    async def _store_content(self, content: BrowserContent) -> None:
        """Store browser content in recent history"""
        self.recent_content.append(content)

        # Keep only recent content
        if len(self.recent_content) > self.max_content_history:
            self.recent_content = self.recent_content[-self.max_content_history :]

        self.logger.debug(f"Stored content from {content.browser_type}: {content.url}")

    async def _notify_content_callbacks(self, content: BrowserContent) -> None:
        """Notify registered callbacks about new content"""
        for callback in self.content_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(content)
                else:
                    callback(content)
            except Exception as e:
                self.logger.error(f"Error in content callback: {e}")

    def register_content_callback(self, callback: callable) -> None:
        """Register a callback to be notified of new browser content"""
        self.content_callbacks.append(callback)
        self.logger.info(f"Registered content callback: {callback.__name__}")

    def unregister_content_callback(self, callback: callable) -> None:
        """Unregister a content callback"""
        if callback in self.content_callbacks:
            self.content_callbacks.remove(callback)
            self.logger.info(f"Unregistered content callback: {callback.__name__}")

    def get_recent_content(self, limit: Optional[int] = None) -> List[BrowserContent]:
        """Get recent browser content"""
        if limit:
            return self.recent_content[-limit:]
        return self.recent_content.copy()

    def get_content_for_url(self, url: str) -> List[BrowserContent]:
        """Get content for a specific URL"""
        return [content for content in self.recent_content if content.url == url]

    def clear_content_history(self) -> None:
        """Clear stored content history"""
        self.recent_content.clear()
        self.logger.info("Cleared browser content history")

    def convert_to_text_content(self, browser_content: BrowserContent) -> TextContent:
        """Convert browser content to TextContent for analysis"""
        # Combine title and content
        combined_text = f"Page Title: {browser_content.title}\nURL: {browser_content.url}\nContent: {browser_content.text_content}"

        metadata = {
            "url": browser_content.url,
            "title": browser_content.title,
            "tab_id": browser_content.tab_id,
            "browser_type": browser_content.browser_type,
            **(browser_content.metadata or {}),
        }

        return TextContent(
            content=combined_text,
            source=f"browser_{browser_content.browser_type}",
            timestamp=browser_content.timestamp,
            metadata=metadata,
        )


class BrowserExtensionManager:
    """Manager for browser extension integrations"""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(__name__)

        # Integration server
        self.server = BrowserIntegrationServer(self.config.get("server", {}))

        # Analysis integration
        self.text_analysis_callback: Optional[callable] = None

    async def start(self) -> bool:
        """Start browser extension integration"""
        try:
            # Start integration server
            if not await self.server.start():
                return False

            # Register content callback
            self.server.register_content_callback(self._handle_browser_content)

            self.logger.info("Browser extension integration started")
            return True

        except Exception as e:
            self.logger.error(f"Failed to start browser extension integration: {e}")
            return False

    async def stop(self) -> None:
        """Stop browser extension integration"""
        try:
            await self.server.stop()
            self.logger.info("Browser extension integration stopped")
        except Exception as e:
            self.logger.error(f"Error stopping browser extension integration: {e}")

    async def _handle_browser_content(self, browser_content: BrowserContent) -> None:
        """Handle new browser content"""
        self.logger.info(
            f"Received content from {browser_content.browser_type}: {browser_content.title}"
        )

        # Convert to TextContent for analysis
        if self.text_analysis_callback:
            try:
                text_content = self.server.convert_to_text_content(browser_content)
                if asyncio.iscoroutinefunction(self.text_analysis_callback):
                    await self.text_analysis_callback(text_content)
                else:
                    self.text_analysis_callback(text_content)
            except Exception as e:
                self.logger.error(f"Error in text analysis callback: {e}")

    def set_text_analysis_callback(self, callback: callable) -> None:
        """Set callback for text analysis"""
        self.text_analysis_callback = callback
        self.logger.info("Text analysis callback registered")

    def get_server_info(self) -> Dict[str, Any]:
        """Get browser integration server information"""
        return {
            "running": self.server.running,
            "host": self.server.host,
            "port": self.server.port,
            "endpoint": f"http://{self.server.host}:{self.server.port}/api/content",
            "status_endpoint": f"http://{self.server.host}:{self.server.port}/api/status",
        }

    def get_extension_config(self) -> Dict[str, Any]:
        """Get configuration for browser extensions"""
        return {
            "api_endpoint": f"http://{self.server.host}:{self.server.port}/api/content",
            "api_key": self.server.api_key,
            "content_types": ["text", "title", "url"],
            "update_interval": 5000,  # milliseconds
            "max_content_length": 10000,
        }
