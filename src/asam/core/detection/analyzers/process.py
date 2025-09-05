"""
Process Monitor Analyzer

Monitors running processes and applications for entertainment detection.
"""

from datetime import datetime
from typing import Any, Optional

import psutil

from ..types import AnalysisType, ContentCategory, DetectionResult, ProcessInfo
from .base import AnalyzerBase


class ProcessAnalyzer(AnalyzerBase):
    """Analyzes running processes for entertainment applications"""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)

        # Load entertainment application patterns
        self.gaming_processes = set(
            self.config.get(
                "gaming_processes",
                [
                    "steam.exe",
                    "steamwebhelper.exe",
                    "gameoverlayui.exe",
                    "epic games launcher",
                    "epicgameslauncher.exe",
                    "origin.exe",
                    "originwebhelper.exe",
                    "battle.net.exe",
                    "agent.exe",
                    "discord.exe",
                    "discordptb.exe",
                    "discordcanary.exe",
                    "minecraft.exe",
                    "javaw.exe",  # Minecraft
                    "league of legends.exe",
                    "riotclientux.exe",
                    "csgo.exe",
                    "dota2.exe",
                    "among us.exe",
                    "genshin impact game",
                    "valorant.exe",
                    "world of warcraft.exe",
                    "wow.exe",
                    "fortnite",
                    "fortniteClient-win64-shipping.exe",
                ],
            )
        )

        self.streaming_processes = set(
            self.config.get(
                "streaming_processes",
                [
                    "spotify.exe",
                    "spotify",
                    "vlc.exe",
                    "vlc",
                    "mpv.exe",
                    "mpv",
                    "netflix",
                    "prime video",
                    "disney+",
                    "youtube",
                    "twitch",
                    "obs64.exe",
                    "obs32.exe",
                    "obs",
                    "streamlabs obs",
                    "xsplit.gamecaster.exe",
                    "xsplit.broadcaster.exe",
                ],
            )
        )

        self.social_media_processes = set(
            self.config.get(
                "social_media_processes",
                [
                    "whatsapp.exe",
                    "whatsapp",
                    "telegram.exe",
                    "telegram",
                    "slack.exe",
                    "slack",
                    "teams.exe",
                    "microsoft teams",
                    "zoom.exe",
                    "zoom",
                    "skype.exe",
                    "skype",
                    "tiktok",
                    "instagram",
                    "facebook messenger",
                ],
            )
        )

        self.browser_processes = set(
            self.config.get(
                "browser_processes",
                [
                    "chrome.exe",
                    "google chrome",
                    "chromium",
                    "firefox.exe",
                    "firefox",
                    "safari",
                    "webkit2webprocess",
                    "msedge.exe",
                    "microsoft edge",
                    "opera.exe",
                    "opera",
                ],
            )
        )

        # Productive application patterns
        self.productive_processes = set(
            self.config.get(
                "productive_processes",
                [
                    "code.exe",
                    "visual studio code",
                    "sublime_text.exe",
                    "notepad++.exe",
                    "atom.exe",
                    "vim",
                    "emacs",
                    "idea64.exe",
                    "intellij",
                    "pycharm64.exe",
                    "devenv.exe",
                    "visual studio",
                    "excel.exe",
                    "winword.exe",
                    "powerpnt.exe",
                    "acrobat.exe",
                    "adobe reader",
                    "terminal",
                    "cmd.exe",
                    "powershell.exe",
                    "git.exe",
                    "python.exe",
                    "node.exe",
                ],
            )
        )

        # CPU/Memory thresholds for gaming detection
        self.high_cpu_threshold = self.config.get("high_cpu_threshold", 15.0)
        self.high_memory_threshold = self.config.get("high_memory_threshold", 10.0)

        # Cache for process information
        self._process_cache: dict[int, ProcessInfo] = {}
        self._last_scan_time: Optional[datetime] = None

    @property
    def analyzer_type(self) -> AnalysisType:
        return AnalysisType.PROCESS

    async def analyze(self, data: Optional[Any] = None) -> Optional[DetectionResult]:
        """
        Analyze currently running processes

        Args:
            data: Unused for process analysis

        Returns:
            DetectionResult with confidence and category
        """
        if not self.should_analyze(data):
            return None

        try:
            # Get current process information
            current_processes = await self._get_current_processes()

            # Analyze processes for entertainment indicators
            analysis_results = self._analyze_processes(current_processes)

            if not analysis_results["detected_categories"]:
                return self._create_low_confidence_result(analysis_results["evidence"])

            # Calculate overall confidence and determine primary category
            confidence = self._calculate_confidence(analysis_results)
            primary_category = self._determine_primary_category(analysis_results)

            return DetectionResult(
                analyzer_type=self.analyzer_type,
                confidence=confidence,
                category=primary_category,
                evidence=analysis_results["evidence"],
                timestamp=datetime.now(),
                metadata={
                    "process_count": len(current_processes),
                    "entertainment_processes": len(
                        analysis_results["entertainment_processes"]
                    ),
                },
            )

        except Exception as e:
            self.logger.error(f"Error in process analysis: {e}")
            return None

    async def _get_current_processes(self) -> list[ProcessInfo]:
        """Get information about currently running processes"""
        processes = []
        current_time = datetime.now()

        try:
            for proc in psutil.process_iter(
                ["pid", "name", "exe", "cpu_percent", "memory_percent"]
            ):
                try:
                    pinfo = proc.info
                    if pinfo["name"] and pinfo["pid"]:
                        # Determine if process is in foreground (simplified)
                        is_foreground = self._is_likely_foreground_process(proc)

                        process_info = ProcessInfo(
                            pid=pinfo["pid"],
                            name=pinfo["name"].lower(),
                            executable_path=pinfo["exe"] or "",
                            cpu_percent=pinfo["cpu_percent"] or 0.0,
                            memory_percent=pinfo["memory_percent"] or 0.0,
                            is_foreground=is_foreground,
                            timestamp=current_time,
                        )
                        processes.append(process_info)

                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

        except Exception as e:
            self.logger.error(f"Error getting process list: {e}")

        return processes

    def _is_likely_foreground_process(self, proc: psutil.Process) -> bool:
        """Determine if a process is likely in the foreground"""
        try:
            # Simple heuristic: processes using more CPU are more likely to be active
            cpu_percent = proc.cpu_percent()
            return cpu_percent > 1.0  # Basic threshold
        except:
            return False

    def _analyze_processes(self, processes: list[ProcessInfo]) -> dict[str, Any]:
        """Analyze processes for entertainment indicators"""
        results = {
            "detected_categories": [],
            "entertainment_processes": [],
            "evidence": {},
            "confidence_scores": [],
        }

        gaming_matches = []
        streaming_matches = []
        social_matches = []
        browser_matches = []
        productive_matches = []
        high_resource_processes = []

        for proc in processes:
            proc_name = proc.name.lower()
            proc_path = proc.executable_path.lower()

            # Check against known entertainment processes
            if self._matches_process_set(proc_name, proc_path, self.gaming_processes):
                gaming_matches.append(
                    {
                        "name": proc.name,
                        "cpu": proc.cpu_percent,
                        "memory": proc.memory_percent,
                        "foreground": proc.is_foreground,
                    }
                )

            elif self._matches_process_set(
                proc_name, proc_path, self.streaming_processes
            ):
                streaming_matches.append(
                    {
                        "name": proc.name,
                        "cpu": proc.cpu_percent,
                        "memory": proc.memory_percent,
                        "foreground": proc.is_foreground,
                    }
                )

            elif self._matches_process_set(
                proc_name, proc_path, self.social_media_processes
            ):
                social_matches.append(
                    {
                        "name": proc.name,
                        "cpu": proc.cpu_percent,
                        "memory": proc.memory_percent,
                        "foreground": proc.is_foreground,
                    }
                )

            elif self._matches_process_set(
                proc_name, proc_path, self.browser_processes
            ):
                browser_matches.append(
                    {
                        "name": proc.name,
                        "cpu": proc.cpu_percent,
                        "memory": proc.memory_percent,
                        "foreground": proc.is_foreground,
                    }
                )

            elif self._matches_process_set(
                proc_name, proc_path, self.productive_processes
            ):
                productive_matches.append(
                    {
                        "name": proc.name,
                        "cpu": proc.cpu_percent,
                        "memory": proc.memory_percent,
                        "foreground": proc.is_foreground,
                    }
                )

            # Check for high resource usage (potential gaming)
            if (
                proc.cpu_percent > self.high_cpu_threshold
                or proc.memory_percent > self.high_memory_threshold
            ):
                high_resource_processes.append(
                    {
                        "name": proc.name,
                        "cpu": proc.cpu_percent,
                        "memory": proc.memory_percent,
                        "foreground": proc.is_foreground,
                    }
                )

        # Evaluate gaming indicators
        if gaming_matches:
            gaming_confidence = self._calculate_gaming_confidence(gaming_matches)
            results["confidence_scores"].append(gaming_confidence)
            results["detected_categories"].append(ContentCategory.GAMING)
            results["entertainment_processes"].extend(gaming_matches)
            results["evidence"]["gaming_processes"] = gaming_matches

        # Evaluate streaming indicators
        if streaming_matches:
            streaming_confidence = self._calculate_streaming_confidence(
                streaming_matches
            )
            results["confidence_scores"].append(streaming_confidence)
            results["detected_categories"].append(ContentCategory.VIDEO_STREAMING)
            results["entertainment_processes"].extend(streaming_matches)
            results["evidence"]["streaming_processes"] = streaming_matches

        # Evaluate social media indicators
        if social_matches:
            social_confidence = self._calculate_social_confidence(social_matches)
            results["confidence_scores"].append(social_confidence * 0.7)  # Lower weight
            results["detected_categories"].append(ContentCategory.SOCIAL_MEDIA)
            results["entertainment_processes"].extend(social_matches)
            results["evidence"]["social_media_processes"] = social_matches

        # Evaluate browser activity (needs additional context)
        if browser_matches:
            browser_confidence = self._calculate_browser_confidence(browser_matches)
            if browser_confidence > 0.3:
                results["confidence_scores"].append(
                    browser_confidence * 0.4
                )  # Low weight
                results["detected_categories"].append(ContentCategory.ENTERTAINMENT)
            results["evidence"]["browser_processes"] = browser_matches

        # High resource processes (potential unknown games)
        if high_resource_processes:
            resource_confidence = self._calculate_resource_confidence(
                high_resource_processes
            )
            if resource_confidence > 0.5:
                results["confidence_scores"].append(resource_confidence * 0.6)
                if ContentCategory.GAMING not in results["detected_categories"]:
                    results["detected_categories"].append(ContentCategory.GAMING)
            results["evidence"]["high_resource_processes"] = high_resource_processes

        # Add productive process context
        if productive_matches:
            results["evidence"]["productive_processes"] = productive_matches

        return results

    def _matches_process_set(
        self, proc_name: str, proc_path: str, process_set: set[str]
    ) -> bool:
        """Check if process matches any pattern in the given set"""
        for pattern in process_set:
            pattern_lower = pattern.lower()
            if (
                pattern_lower in proc_name
                or pattern_lower in proc_path
                or proc_name.startswith(pattern_lower.split(".")[0])
            ):
                return True
        return False

    def _calculate_gaming_confidence(self, gaming_matches: list[dict]) -> float:
        """Calculate confidence for gaming activity"""
        if not gaming_matches:
            return 0.0

        base_confidence = 0.8  # High confidence for known gaming processes

        # Boost confidence for foreground gaming processes
        foreground_boost = (
            0.2 if any(proc["foreground"] for proc in gaming_matches) else 0.0
        )

        # Boost confidence for high resource usage
        resource_boost = 0.0
        for proc in gaming_matches:
            if proc["cpu"] > 20 or proc["memory"] > 15:
                resource_boost += 0.1

        return min(base_confidence + foreground_boost + resource_boost, 1.0)

    def _calculate_streaming_confidence(self, streaming_matches: list[dict]) -> float:
        """Calculate confidence for streaming/media activity"""
        if not streaming_matches:
            return 0.0

        base_confidence = 0.7

        # Higher confidence for active streaming processes
        active_boost = (
            0.2 if any(proc["cpu"] > 5 for proc in streaming_matches) else 0.0
        )

        return min(base_confidence + active_boost, 1.0)

    def _calculate_social_confidence(self, social_matches: list[dict]) -> float:
        """Calculate confidence for social media activity"""
        if not social_matches:
            return 0.0

        # Lower base confidence as social apps might be work-related
        base_confidence = 0.5

        # Multiple social apps increase entertainment likelihood
        multi_app_boost = min(len(social_matches) * 0.1, 0.3)

        return min(base_confidence + multi_app_boost, 1.0)

    def _calculate_browser_confidence(self, browser_matches: list[dict]) -> float:
        """Calculate confidence for browser-based entertainment"""
        if not browser_matches:
            return 0.0

        # Very low base confidence - browsers are used for everything
        base_confidence = 0.2

        # Boost for high CPU usage (video streaming, games)
        resource_boost = 0.0
        for proc in browser_matches:
            if proc["cpu"] > 10:  # High CPU suggests video/games
                resource_boost += 0.3

        return min(base_confidence + resource_boost, 1.0)

    def _calculate_resource_confidence(
        self, high_resource_processes: list[dict]
    ) -> float:
        """Calculate confidence for high-resource processes (potential games)"""
        if not high_resource_processes:
            return 0.0

        # Filter out known productive processes
        suspicious_processes = [
            proc
            for proc in high_resource_processes
            if not self._matches_process_set(
                proc["name"].lower(), "", self.productive_processes
            )
        ]

        if not suspicious_processes:
            return 0.0

        # Calculate confidence based on resource usage
        max_cpu = max(proc["cpu"] for proc in suspicious_processes)
        max_memory = max(proc["memory"] for proc in suspicious_processes)

        cpu_confidence = min(max_cpu / 50, 1.0)  # Scale CPU usage
        memory_confidence = min(max_memory / 25, 1.0)  # Scale memory usage

        return max(cpu_confidence, memory_confidence) * 0.6  # Conservative multiplier

    def _calculate_confidence(self, analysis_results: dict[str, Any]) -> float:
        """Calculate overall confidence from analysis results"""
        confidence_scores = analysis_results["confidence_scores"]
        if not confidence_scores:
            return 0.1

        # Use maximum confidence but consider multiple indicators
        max_confidence = max(confidence_scores)
        multi_indicator_boost = min(len(confidence_scores) * 0.1, 0.2)

        return min(max_confidence + multi_indicator_boost, 1.0)

    def _determine_primary_category(
        self, analysis_results: dict[str, Any]
    ) -> ContentCategory:
        """Determine primary content category"""
        categories = analysis_results["detected_categories"]
        if not categories:
            return ContentCategory.UNKNOWN

        # Priority order for categories
        category_priority = {
            ContentCategory.GAMING: 3,
            ContentCategory.VIDEO_STREAMING: 2,
            ContentCategory.SOCIAL_MEDIA: 1,
            ContentCategory.ENTERTAINMENT: 0,
        }

        return max(categories, key=lambda x: category_priority.get(x, 0))

    def _create_low_confidence_result(
        self, evidence: dict[str, Any]
    ) -> DetectionResult:
        """Create a low-confidence result when no clear entertainment is detected"""
        return DetectionResult(
            analyzer_type=self.analyzer_type,
            confidence=0.1,
            category=ContentCategory.PRODUCTIVE,  # Default to productive
            evidence=evidence,
            timestamp=datetime.now(),
            metadata={"no_entertainment_detected": True},
        )
