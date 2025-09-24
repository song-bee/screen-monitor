#!/usr/bin/env python3
"""
ASAM Browser Integration Test Script

Tests the browser extension integration by starting the ASAM service
with browser integration enabled and providing testing utilities.
"""

import asyncio
import json
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from asam.core.service import ASAMService
from asam.integrations.browser import BrowserExtensionManager


async def test_browser_integration():
    """Test the browser extension integration"""

    print("🚀 Starting ASAM Browser Integration Test...")
    print("=" * 60)

    # Configuration for testing
    config = {
        "integrations": {
            "browser": {
                "enabled": True,
                "server": {
                    "host": "localhost",
                    "port": 8888,
                    "api_key": "asam-browser-integration"
                }
            }
        },
        "detection": {
            "confidence_threshold": 0.6,
            "text_detection": {
                "llm_model": "llama3.2:3b"
            }
        }
    }

    try:
        # Initialize browser extension manager
        print("📱 Initializing Browser Extension Manager...")
        browser_manager = BrowserExtensionManager(config.get("integrations", {}).get("browser", {}))

        # Start browser integration
        if await browser_manager.start():
            print("✅ Browser integration server started successfully!")

            # Display connection info
            server_info = browser_manager.get_server_info()
            print(f"🌐 Server running at: {server_info['endpoint']}")
            print(f"🔑 API Key: {browser_manager.server.api_key}")
            print(f"📊 Status endpoint: {server_info['status_endpoint']}")

            # Display extension configuration
            print("\n📋 Browser Extension Configuration:")
            extension_config = browser_manager.get_extension_config()
            for key, value in extension_config.items():
                print(f"  {key}: {value}")

            print("\n" + "=" * 60)
            print("🎯 TESTING INSTRUCTIONS:")
            print("=" * 60)
            print("1. Open Chrome and go to chrome://extensions/")
            print("2. Enable 'Developer mode' (toggle in top right)")
            print("3. Click 'Load unpacked' and select:")
            print(f"   📁 {Path(__file__).parent / 'browser-extension'}")
            print("4. The ASAM Monitor extension should appear")
            print("5. Click the extension icon to see the popup")
            print("6. Click 'Send Test Data' to test connectivity")
            print("7. Browse to different websites and watch the console")
            print("\n💡 Monitor this console for incoming browser data...")
            print("🛑 Press Ctrl+C to stop the test")
            print("=" * 60)

            # Set up content callback to display received data
            def handle_browser_content(browser_content):
                print(f"\n📨 Received content from {browser_content.browser_type}:")
                print(f"   📄 Title: {browser_content.title}")
                print(f"   🔗 URL: {browser_content.url}")
                print(f"   📝 Content length: {len(browser_content.text_content)} characters")
                print(f"   🏷️  Tab ID: {browser_content.tab_id}")

                # Show content preview (first 200 characters)
                if browser_content.text_content:
                    content_preview = browser_content.text_content.strip()[:200]
                    if len(browser_content.text_content) > 200:
                        content_preview += "..."
                    print(f"   📖 Content preview:")
                    print(f"      {repr(content_preview)}")

                if browser_content.metadata:
                    print(f"   📊 Metadata:")
                    for key, value in browser_content.metadata.items():
                        if isinstance(value, str) and len(value) > 50:
                            value_display = value[:47] + "..."
                        else:
                            value_display = value
                        print(f"      {key}: {value_display}")

                print("   " + "-" * 70)

            browser_manager.server.register_content_callback(handle_browser_content)

            # Keep running until interrupted
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Test interrupted by user")

        else:
            print("❌ Failed to start browser integration server")
            return False

    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

    finally:
        # Cleanup
        print("\n🧹 Cleaning up...")
        if browser_manager:
            await browser_manager.stop()
        print("✅ Browser integration test completed")

    return True


async def test_with_full_asam():
    """Test browser integration with full ASAM service"""

    print("🚀 Starting ASAM Service with Browser Integration...")
    print("=" * 60)

    try:
        # Initialize ASAM service
        service = ASAMService(None)  # Use default config

        # Start ASAM service
        print("🔧 Starting ASAM service...")
        await service.start()

        # Initialize browser integration
        print("📱 Starting browser integration...")
        browser_manager = BrowserExtensionManager()

        if await browser_manager.start():
            print("✅ Full ASAM + Browser integration running!")

            # Set up integration between browser and ASAM
            def handle_browser_content(browser_content):
                print(f"\n📨 Browser content received: {browser_content.title}")

                # Convert to TextContent and analyze
                text_content = browser_manager.server.convert_to_text_content(browser_content)
                print(f"   📝 Converted to TextContent: {len(text_content.content)} chars")

                # You could integrate this with ASAM's detection engine here
                # For now, just log it

            browser_manager.server.register_content_callback(handle_browser_content)

            print("\n🎯 Full ASAM system ready for browser extension testing!")
            print("Follow the same extension installation steps as above.")
            print("Press Ctrl+C to stop...")

            # Keep running
            try:
                while True:
                    await asyncio.sleep(1)
            except KeyboardInterrupt:
                print("\n🛑 Shutting down...")

        else:
            print("❌ Failed to start browser integration")

    except Exception as e:
        print(f"❌ Full ASAM test failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        # Cleanup
        if browser_manager:
            await browser_manager.stop()
        if 'service' in locals():
            await service.stop()
        print("✅ Full ASAM test completed")


def main():
    """Main test function"""
    print("ASAM Browser Integration Test Suite")
    print("Choose a test mode:")
    print("1. Browser integration only (lightweight)")
    print("2. Full ASAM + Browser integration")

    choice = input("\nEnter choice (1 or 2): ").strip()

    if choice == "1":
        asyncio.run(test_browser_integration())
    elif choice == "2":
        asyncio.run(test_with_full_asam())
    else:
        print("Invalid choice. Exiting.")
        return False

    return True


if __name__ == "__main__":
    main()