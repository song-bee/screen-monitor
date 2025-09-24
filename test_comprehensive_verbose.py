#!/usr/bin/env python3
"""
Comprehensive Ultra-Verbose ASAM Test Suite

Maximum verbosity test combining:
- Live browser integration
- Ultra-detailed LLM analysis
- Complete pipeline visibility
- Performance profiling
- Real-time content streaming
- Interactive debugging mode
"""

import asyncio
import json
import sys
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import aiohttp

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from asam.integrations.browser import BrowserExtensionManager
from asam.core.detection.analyzers.text import TextAnalyzer
from asam.core.detection.types import TextContent


class ComprehensiveVerboseLogger:
    """Ultra-detailed logger with multiple output modes"""

    def __init__(self):
        self.session_start = datetime.now()
        self.analysis_count = 0
        self.setup_enhanced_logging()

    def setup_enhanced_logging(self):
        """Setup enhanced logging with detailed formatting"""
        # Create custom formatter with even more detail
        class MaxVerboseFormatter(logging.Formatter):
            def format(self, record):
                timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

                # Enhanced color coding
                colors = {
                    'DEBUG': '\033[96m',     # Bright Cyan
                    'INFO': '\033[92m',      # Bright Green
                    'WARNING': '\033[93m',   # Bright Yellow
                    'ERROR': '\033[91m',     # Bright Red
                    'CRITICAL': '\033[95m'   # Bright Magenta
                }
                reset = '\033[0m'
                bold = '\033[1m'

                color = colors.get(record.levelname, '')
                level_display = f"{color}{bold}[{record.levelname:<8}]{reset}"

                # Enhanced module display
                module_parts = record.name.split('.')
                if len(module_parts) > 2:
                    module_display = f"{module_parts[-2]}.{module_parts[-1]}"
                else:
                    module_display = module_parts[-1] if module_parts else record.name
                module_display = f"{module_display:<20}"

                # Thread/task info
                thread_info = f"[T:{record.thread:x}]" if hasattr(record, 'thread') else ""

                return f"{timestamp} | {level_display} | {module_display} | {thread_info} {record.getMessage()}"

        # Configure root logger with maximum verbosity
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)

        # Remove existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Add enhanced console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        console_handler.setFormatter(MaxVerboseFormatter())
        root_logger.addHandler(console_handler)

        # Set all ASAM loggers to maximum verbosity
        asam_loggers = [
            'asam.core.detection.analyzers.text',
            'asam.integrations.browser',
            'asam.core.detection.engine',
            'asam.core.service'
        ]

        for logger_name in asam_loggers:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.DEBUG)

    def print_mega_header(self, title, width=150):
        """Print mega-sized header"""
        border = "█" * width
        padding = " " * ((width - len(title)) // 2)
        print(f"\n{border}")
        print(f"{padding}{title}")
        print(f"{border}\n")

    def print_section_divider(self, title, char="═", width=120):
        """Print section divider"""
        print(f"\n{char * width}")
        print(f"  🔹 {title}")
        print(f"{char * width}")

    def print_analysis_header(self, count, content_info):
        """Print detailed analysis header"""
        self.analysis_count += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        session_time = (datetime.now() - self.session_start).total_seconds()

        print(f"\n{'▓' * 140}")
        print(f"🚀 ANALYSIS #{self.analysis_count:03d} | {timestamp} | Session: {session_time:.1f}s")
        print(f"{'▓' * 140}")

        print(f"📊 CONTENT METADATA:")
        print(f"   📄 Title: {content_info.get('title', 'Unknown')}")
        print(f"   🔗 URL: {content_info.get('url', 'Unknown')}")
        print(f"   🏷️  Tab: {content_info.get('tab_id', 'Unknown')}")
        print(f"   🌐 Browser: {content_info.get('browser_type', 'Unknown')}")
        print(f"   📏 Content Length: {content_info.get('content_length', 0):,} characters")
        print(f"   ⏱️  Processing Started: {timestamp}")


def format_json_with_colors(data, indent=2):
    """Format JSON with color coding for better readability"""
    formatted = json.dumps(data, indent=indent, ensure_ascii=False)

    # Color coding (basic)
    colored = formatted
    colored = colored.replace('"category"', '\033[96m"category"\033[0m')
    colored = colored.replace('"confidence"', '\033[93m"confidence"\033[0m')
    colored = colored.replace('"reasoning"', '\033[92m"reasoning"\033[0m')
    colored = colored.replace('"keywords"', '\033[95m"keywords"\033[0m')

    return colored


def display_content_analysis(content, max_preview=600, max_lines=30):
    """Display content with intelligent truncation and analysis"""
    lines = content.split('\n')
    char_count = len(content)
    line_count = len(lines)

    print(f"📝 CONTENT ANALYSIS:")
    print(f"   📊 Statistics: {char_count:,} chars, {line_count} lines")
    print(f"   📈 Avg line length: {char_count/line_count:.1f} chars/line")

    # Content type heuristics
    content_lower = content.lower()
    indicators = {
        'Gaming': ['game', 'play', 'win', 'level', 'score', 'gaming'],
        'Social': ['like', 'share', 'comment', 'follow', 'social'],
        'Educational': ['learn', 'tutorial', 'course', 'study', 'education'],
        'Entertainment': ['video', 'watch', 'funny', 'entertainment', 'movie'],
        'News': ['breaking', 'news', 'report', 'today', 'update'],
        'Technical': ['code', 'programming', 'development', 'api', 'technical']
    }

    detected_types = []
    for content_type, keywords in indicators.items():
        matches = sum(1 for keyword in keywords if keyword in content_lower)
        if matches >= 2:
            detected_types.append(f"{content_type}({matches})")

    print(f"   🎯 Content Type Hints: {', '.join(detected_types) if detected_types else 'Mixed/Unclear'}")

    print(f"\n📖 CONTENT PREVIEW (showing {min(max_preview, char_count)} of {char_count} chars):")
    print("┌" + "─" * 118 + "┐")

    displayed_chars = 0
    for i, line in enumerate(lines[:max_lines], 1):
        if displayed_chars >= max_preview:
            remaining_lines = len(lines) - i + 1
            print(f"│ ... [{remaining_lines} more lines, {char_count - displayed_chars:,} more chars] ...{' ' * 40}│")
            break

        line_display = line[:116] if len(line) > 116 else line
        padding = " " * (116 - len(line_display))
        print(f"│ {line_display}{padding} │")
        displayed_chars += len(line) + 1  # +1 for newline

    print("└" + "─" * 118 + "┘")


async def comprehensive_verbose_test():
    """Run comprehensive ultra-verbose test"""

    verbose_logger = ComprehensiveVerboseLogger()
    verbose_logger.print_mega_header("🔬 COMPREHENSIVE ULTRA-VERBOSE ASAM ANALYSIS SUITE 🔬")

    print(f"🚀 Session Started: {verbose_logger.session_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🖥️  Platform: {sys.platform}")
    print(f"🐍 Python: {sys.version.split()[0]}")
    print(f"📂 Working Dir: {Path.cwd()}")

    verbose_logger.print_section_divider("COMPONENT INITIALIZATION", "═")

    # Initialize components with detailed tracking
    init_start = time.perf_counter()

    print("🔧 Initializing TextAnalyzer with ultra-verbose settings...")
    text_analyzer = TextAnalyzer()

    analyzer_init_start = time.perf_counter()
    if not await text_analyzer.initialize():
        print("❌ CRITICAL: TextAnalyzer initialization failed!")
        print("💡 Solution: Ensure Ollama is running (brew services start ollama)")
        return False
    analyzer_init_time = time.perf_counter() - analyzer_init_start

    print(f"✅ TextAnalyzer initialized in {analyzer_init_time:.3f}s")

    print("🔧 Initializing BrowserExtensionManager...")
    browser_manager = BrowserExtensionManager()

    browser_init_start = time.perf_counter()
    if not await browser_manager.start():
        print("❌ CRITICAL: BrowserExtensionManager initialization failed!")
        return False
    browser_init_time = time.perf_counter() - browser_init_start

    total_init_time = time.perf_counter() - init_start
    print(f"✅ BrowserExtensionManager initialized in {browser_init_time:.3f}s")
    print(f"🎯 Total initialization time: {total_init_time:.3f}s")

    # Display server information
    server_info = browser_manager.get_server_info()
    print(f"\n🌐 Server Details:")
    print(f"   🔗 API Endpoint: {server_info['endpoint']}")
    print(f"   📊 Status Endpoint: {server_info['status_endpoint']}")
    print(f"   🔑 API Key: {browser_manager.server.api_key}")
    print(f"   🏃 Running: {server_info['running']}")

    verbose_logger.print_section_divider("ULTRA-VERBOSE CONTENT ANALYSIS PIPELINE", "═")

    try:
        # Enhanced content callback with maximum verbosity
        async def ultra_verbose_content_handler(browser_content):
            content_info = {
                'title': browser_content.title,
                'url': browser_content.url,
                'tab_id': browser_content.tab_id,
                'browser_type': browser_content.browser_type,
                'content_length': len(browser_content.text_content),
                'timestamp': browser_content.timestamp
            }

            verbose_logger.print_analysis_header(verbose_logger.analysis_count, content_info)

            # Display ultra-detailed content analysis
            display_content_analysis(browser_content.text_content)

            # Metadata deep dive
            if browser_content.metadata:
                print(f"\n🔍 METADATA DEEP DIVE:")
                for key, value in browser_content.metadata.items():
                    value_display = str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                    print(f"   📊 {key}: {value_display}")

            verbose_logger.print_section_divider("🧠 LLM ANALYSIS PIPELINE", "─", 100)

            # Convert to TextContent with timing
            conversion_start = time.perf_counter()
            text_content = browser_manager.server.convert_to_text_content(browser_content)
            conversion_time = time.perf_counter() - conversion_start

            print(f"📦 TextContent Conversion: {conversion_time*1000:.2f}ms")
            print(f"   🏷️  Source: {text_content.source}")
            print(f"   📏 Combined Length: {len(text_content.content):,} chars")
            print(f"   🕐 Timestamp: {text_content.timestamp}")

            # Perform LLM analysis with ultra-detailed tracking
            analysis_start = time.perf_counter()
            print(f"\n⚡ Starting LLM Analysis Pipeline...")
            print(f"   ⏰ Analysis Start: {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")

            try:
                detection_result = await text_analyzer.analyze(text_content)
                analysis_time = time.perf_counter() - analysis_start

                if detection_result:
                    verbose_logger.print_section_divider("📋 ANALYSIS RESULTS", "─", 80)

                    print(f"✅ Analysis SUCCESS in {analysis_time*1000:.1f}ms")
                    print(f"🎯 Category: {detection_result.category.value.upper()}")
                    print(f"🔢 Confidence: {detection_result.confidence:.4f} ({detection_result.confidence*100:.1f}%)")
                    print(f"🔧 Analyzer: {detection_result.analyzer_type.value}")

                    # Confidence level analysis
                    if detection_result.confidence >= 0.9:
                        confidence_level = "EXTREMELY HIGH 🔥"
                        action_color = "🔴"
                    elif detection_result.confidence >= 0.8:
                        confidence_level = "HIGH 🟠"
                        action_color = "🟡"
                    elif detection_result.confidence >= 0.6:
                        confidence_level = "MEDIUM 🟡"
                        action_color = "🟢"
                    else:
                        confidence_level = "LOW 🔵"
                        action_color = "⚪"

                    print(f"📊 Confidence Level: {confidence_level}")

                    # Evidence deep dive
                    if detection_result.evidence:
                        evidence = detection_result.evidence

                        verbose_logger.print_section_divider("🔍 EVIDENCE ANALYSIS", "·", 60)

                        print(f"🤖 LLM Details:")
                        print(f"   📦 Model Used: {evidence.get('model_used', 'unknown')}")
                        print(f"   📝 LLM Category: {evidence.get('llm_category', 'unknown')}")
                        print(f"   🏷️  Subcategory: {evidence.get('subcategory', 'unclear')}")
                        print(f"   📏 Input Length: {evidence.get('text_length', 'unknown')} chars")
                        print(f"   📡 Text Source: {evidence.get('text_source', 'unknown')}")

                        # Keywords analysis
                        keywords = evidence.get('keywords', [])
                        print(f"\n🔑 KEYWORDS ANALYSIS ({len(keywords)} found):")
                        if keywords:
                            # Group keywords by length for better display
                            short_keywords = [k for k in keywords if len(k) <= 8]
                            long_keywords = [k for k in keywords if len(k) > 8]

                            if short_keywords:
                                print(f"   📎 Short: {', '.join(short_keywords[:10])}")
                            if long_keywords:
                                print(f"   📝 Long: {', '.join(long_keywords[:5])}")
                        else:
                            print("   ❌ No keywords identified")

                        # Reasoning deep dive
                        reasoning = evidence.get('reasoning', 'No reasoning provided')
                        print(f"\n💭 LLM REASONING ANALYSIS:")
                        print(f"   📏 Length: {len(reasoning)} characters")

                        # Split reasoning into sentences for better display
                        sentences = [s.strip() for s in reasoning.split('.') if s.strip()]
                        print(f"   📄 Sentences: {len(sentences)}")
                        print(f"   📖 Full Reasoning:")

                        for i, sentence in enumerate(sentences[:8], 1):  # Show max 8 sentences
                            print(f"      {i:2d}. {sentence}")

                        if len(sentences) > 8:
                            print(f"      ... [{len(sentences) - 8} more sentences]")

                    # Decision analysis
                    verbose_logger.print_section_divider("⚖️  DECISION ANALYSIS", "·", 60)

                    category = detection_result.category.value
                    confidence = detection_result.confidence

                    if category == "entertainment":
                        if confidence >= 0.8:
                            action = f"{action_color} BLOCK - High confidence entertainment detection"
                            risk_level = "HIGH RISK"
                        elif confidence >= 0.6:
                            action = f"{action_color} WARN - Medium confidence entertainment detection"
                            risk_level = "MEDIUM RISK"
                        else:
                            action = f"{action_color} MONITOR - Low confidence, continue observation"
                            risk_level = "LOW RISK"
                    elif category == "productive":
                        action = f"✅ ALLOW - Productive content detected"
                        risk_level = "PRODUCTIVE"
                    else:
                        action = f"🤔 REVIEW - Unclear content type"
                        risk_level = "UNKNOWN"

                    print(f"🎯 Risk Level: {risk_level}")
                    print(f"⚡ Recommended Action: {action}")
                    print(f"🔒 Would Lock Screen: {'YES' if 'BLOCK' in action else 'NO'}")

                    # Performance metrics
                    verbose_logger.print_section_divider("📊 PERFORMANCE METRICS", "·", 60)
                    print(f"⏱️  Timing Breakdown:")
                    print(f"   🔄 Content Conversion: {conversion_time*1000:.2f}ms")
                    print(f"   🧠 LLM Analysis: {analysis_time*1000:.1f}ms")
                    print(f"   📊 Total Processing: {(conversion_time + analysis_time)*1000:.1f}ms")

                else:
                    analysis_time = time.perf_counter() - analysis_start
                    print(f"❌ Analysis FAILED after {analysis_time*1000:.1f}ms")
                    print("🔍 Check logs above for detailed error information")

            except Exception as e:
                analysis_time = time.perf_counter() - analysis_start
                print(f"💥 Analysis CRASHED after {analysis_time*1000:.1f}ms")
                print(f"❌ Error: {str(e)}")
                print(f"🔧 Error Type: {type(e).__name__}")

                import traceback
                verbose_logger.print_section_divider("🐛 ERROR TRACEBACK", "!", 80)
                traceback.print_exc()

            # Analysis completion
            print(f"\n{'▓' * 140}")
            print(f"✅ Analysis #{verbose_logger.analysis_count} Complete | {datetime.now().strftime('%H:%M:%S')}")
            print(f"{'▓' * 140}")

        # Register ultra-verbose handler
        browser_manager.server.register_content_callback(ultra_verbose_content_handler)

        verbose_logger.print_section_divider("🎯 ULTRA-VERBOSE MONITORING ACTIVE", "═")
        print("📡 Monitoring browser content with MAXIMUM verbosity...")
        print("🌐 Browse any website to see ultra-detailed analysis")
        print("🛑 Press Ctrl+C to stop monitoring")
        print("=" * 120)

        # Keep monitoring
        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            print(f"\n🛑 Monitoring stopped by user")

    except Exception as e:
        print(f"💥 Test execution failed: {e}")
        import traceback
        traceback.print_exc()

    finally:
        verbose_logger.print_section_divider("🧹 CLEANUP PROCESS", "═")
        cleanup_start = time.perf_counter()

        print("🔄 Cleaning up TextAnalyzer...")
        await text_analyzer.cleanup()

        print("🔄 Cleaning up BrowserExtensionManager...")
        await browser_manager.stop()

        cleanup_time = time.perf_counter() - cleanup_start
        print(f"✅ Cleanup completed in {cleanup_time:.3f}s")

        # Session summary
        session_time = (datetime.now() - verbose_logger.session_start).total_seconds()
        verbose_logger.print_mega_header(f"🏁 SESSION COMPLETE - {verbose_logger.analysis_count} ANALYSES IN {session_time:.1f}s 🏁")


if __name__ == "__main__":
    try:
        asyncio.run(comprehensive_verbose_test())
    except KeyboardInterrupt:
        print("\n🛑 Test interrupted by user")
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()