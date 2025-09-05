"""
Network Activity Analyzer

Monitors network connections and traffic for entertainment detection.
"""

import asyncio
import socket
from datetime import datetime
from typing import Any, Optional

import psutil

from ..types import AnalysisType, ContentCategory, DetectionResult, NetworkActivity
from .base import AnalyzerBase


class NetworkAnalyzer(AnalyzerBase):
    """Analyzes network activity for entertainment-related traffic"""

    def __init__(self, config: Optional[dict[str, Any]] = None):
        super().__init__(config)

        # Entertainment-related domains and patterns
        self.gaming_domains = set(
            self.config.get(
                "gaming_domains",
                [
                    "steampowered.com",
                    "steamcommunity.com",
                    "steamusercontent.com",
                    "epicgames.com",
                    "unrealengine.com",
                    "origin.com",
                    "ea.com",
                    "eaplay.com",
                    "battle.net",
                    "blizzard.com",
                    "activision.com",
                    "riotgames.com",
                    "leagueoflegends.com",
                    "valorant.com",
                    "minecraft.net",
                    "mojang.com",
                    "ubisoft.com",
                    "ubisoftconnect.com",
                    "rockstargames.com",
                    "socialclub.rockstargames.com",
                    "twitch.tv",
                    "twitchcdn.net",
                    "discord.com",
                    "discordapp.com",
                    "discord.gg",
                ],
            )
        )

        self.streaming_domains = set(
            self.config.get(
                "streaming_domains",
                [
                    "youtube.com",
                    "youtu.be",
                    "ytimg.com",
                    "googlevideo.com",
                    "netflix.com",
                    "nflximg.net",
                    "nflxext.com",
                    "nflxvideo.net",
                    "spotify.com",
                    "scdn.co",
                    "spotifycdn.com",
                    "twitch.tv",
                    "twitchcdn.net",
                    "jtvnw.net",
                    "hulu.com",
                    "hulustream.com",
                    "primevideo.com",
                    "amazon.com",
                    "amazonvideo.com",
                    "disneyplus.com",
                    "disney.com",
                    "hbomax.com",
                    "hbo.com",
                    "paramount.com",
                    "paramountplus.com",
                    "crunchyroll.com",
                    "funimation.com",
                ],
            )
        )

        self.social_media_domains = set(
            self.config.get(
                "social_media_domains",
                [
                    "facebook.com",
                    "fbcdn.net",
                    "fb.com",
                    "instagram.com",
                    "cdninstagram.com",
                    "twitter.com",
                    "twimg.com",
                    "x.com",
                    "tiktok.com",
                    "tiktokcdn.com",
                    "byteoversea.com",
                    "linkedin.com",
                    "licdn.com",
                    "reddit.com",
                    "redd.it",
                    "redditstatic.com",
                    "redditmedia.com",
                    "snapchat.com",
                    "snap.com",
                    "pinterest.com",
                    "pinimg.com",
                    "whatsapp.com",
                    "whatsapp.net",
                    "telegram.org",
                    "telegram.me",
                ],
            )
        )

        # CDN and content delivery patterns
        self.streaming_patterns = [
            "googlevideo.com",  # YouTube
            "nflxvideo.net",  # Netflix
            "cloudfront.net",  # Amazon CloudFront
            "fastly.com",  # Fastly CDN
            "akamai.net",  # Akamai CDN
            "cdn",  # Generic CDN pattern
        ]

        # Port patterns for entertainment
        self.gaming_ports = set(
            self.config.get(
                "gaming_ports",
                [
                    27015,
                    27016,
                    27017,
                    27018,
                    27019,  # Steam
                    3724,  # Battle.net
                    1119,  # Origin
                    5222,  # League of Legends
                    8393,
                    8394,
                    8395,
                    8396,
                    8397,  # Riot Games
                    25565,  # Minecraft
                    7777,
                    7778,
                    7779,  # Common game ports
                ],
            )
        )

        # Traffic thresholds
        self.high_bandwidth_threshold = self.config.get(
            "high_bandwidth_threshold", 1024 * 1024
        )  # 1MB
        self.streaming_port_range = self.config.get(
            "streaming_port_range", (8000, 65535)
        )

        # Connection tracking
        self._previous_connections: dict[str, NetworkActivity] = {}
        self._connection_history: list[NetworkActivity] = []

    @property
    def analyzer_type(self) -> AnalysisType:
        return AnalysisType.NETWORK

    async def analyze(self, data: Optional[Any] = None) -> Optional[DetectionResult]:
        """
        Analyze current network connections and traffic

        Args:
            data: Unused for network analysis

        Returns:
            DetectionResult with confidence and category
        """
        if not self.should_analyze(data):
            return None

        try:
            # Get current network connections
            current_connections = await self._get_network_connections()

            # Analyze connections for entertainment indicators
            analysis_results = self._analyze_connections(current_connections)

            # Update connection history
            self._update_connection_history(current_connections)

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
                    "connection_count": len(current_connections),
                    "entertainment_connections": len(
                        analysis_results["entertainment_connections"]
                    ),
                },
            )

        except Exception as e:
            self.logger.error(f"Error in network analysis: {e}")
            return None

    async def _get_network_connections(self) -> list[NetworkActivity]:
        """Get current network connections with process information"""
        connections = []
        current_time = datetime.now()

        try:
            # Get network connections
            for conn in psutil.net_connections(kind="inet"):
                if conn.status != psutil.CONN_ESTABLISHED:
                    continue

                # Get process info if available
                process_name = "unknown"
                if conn.pid:
                    try:
                        process = psutil.Process(conn.pid)
                        process_name = process.name()
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass

                # Extract remote address info
                if not conn.raddr:
                    continue

                remote_host = conn.raddr.ip
                remote_port = conn.raddr.port

                # Try to resolve hostname (with timeout)
                hostname = await self._resolve_hostname(remote_host)

                # Create network activity record
                # Note: psutil doesn't provide real-time byte counts per connection
                # This would need to be tracked separately or estimated
                network_activity = NetworkActivity(
                    process_name=process_name,
                    destination_host=hostname or remote_host,
                    destination_port=remote_port,
                    bytes_sent=0,  # Would need separate tracking
                    bytes_received=0,  # Would need separate tracking
                    protocol="TCP" if conn.type == socket.SOCK_STREAM else "UDP",
                    timestamp=current_time,
                )

                connections.append(network_activity)

        except Exception as e:
            self.logger.error(f"Error getting network connections: {e}")

        return connections

    async def _resolve_hostname(
        self, ip_address: str, timeout: float = 1.0
    ) -> Optional[str]:
        """Resolve IP address to hostname with timeout"""
        try:
            # Use asyncio timeout for hostname resolution
            future = asyncio.get_event_loop().run_in_executor(
                None, socket.gethostbyaddr, ip_address
            )
            result = await asyncio.wait_for(future, timeout=timeout)
            return result[0] if result else None
        except (asyncio.TimeoutError, socket.herror, socket.gaierror):
            return None

    def _analyze_connections(
        self, connections: list[NetworkActivity]
    ) -> dict[str, Any]:
        """Analyze network connections for entertainment indicators"""
        results = {
            "detected_categories": [],
            "entertainment_connections": [],
            "evidence": {},
            "confidence_scores": [],
        }

        gaming_connections = []
        streaming_connections = []
        social_connections = []
        suspicious_ports = []

        for conn in connections:
            hostname = conn.destination_host.lower()
            port = conn.destination_port

            # Check against known entertainment domains
            if self._matches_domain_set(hostname, self.gaming_domains):
                gaming_connections.append(
                    {
                        "process": conn.process_name,
                        "host": hostname,
                        "port": port,
                        "protocol": conn.protocol,
                    }
                )

            elif self._matches_domain_set(hostname, self.streaming_domains):
                streaming_connections.append(
                    {
                        "process": conn.process_name,
                        "host": hostname,
                        "port": port,
                        "protocol": conn.protocol,
                    }
                )

            elif self._matches_domain_set(hostname, self.social_media_domains):
                social_connections.append(
                    {
                        "process": conn.process_name,
                        "host": hostname,
                        "port": port,
                        "protocol": conn.protocol,
                    }
                )

            # Check for streaming patterns in hostname
            elif any(pattern in hostname for pattern in self.streaming_patterns):
                streaming_connections.append(
                    {
                        "process": conn.process_name,
                        "host": hostname,
                        "port": port,
                        "protocol": conn.protocol,
                        "pattern_match": True,
                    }
                )

            # Check for gaming ports
            if port in self.gaming_ports:
                gaming_connections.append(
                    {
                        "process": conn.process_name,
                        "host": hostname,
                        "port": port,
                        "protocol": conn.protocol,
                        "gaming_port": True,
                    }
                )

            # Check for suspicious port patterns
            if self._is_suspicious_port(port, conn.process_name):
                suspicious_ports.append(
                    {
                        "process": conn.process_name,
                        "host": hostname,
                        "port": port,
                        "protocol": conn.protocol,
                    }
                )

        # Evaluate gaming indicators
        if gaming_connections:
            gaming_confidence = self._calculate_gaming_confidence(gaming_connections)
            results["confidence_scores"].append(gaming_confidence)
            results["detected_categories"].append(ContentCategory.GAMING)
            results["entertainment_connections"].extend(gaming_connections)
            results["evidence"]["gaming_connections"] = gaming_connections

        # Evaluate streaming indicators
        if streaming_connections:
            streaming_confidence = self._calculate_streaming_confidence(
                streaming_connections
            )
            results["confidence_scores"].append(streaming_confidence)
            results["detected_categories"].append(ContentCategory.VIDEO_STREAMING)
            results["entertainment_connections"].extend(streaming_connections)
            results["evidence"]["streaming_connections"] = streaming_connections

        # Evaluate social media indicators
        if social_connections:
            social_confidence = self._calculate_social_confidence(social_connections)
            results["confidence_scores"].append(social_confidence * 0.6)  # Lower weight
            results["detected_categories"].append(ContentCategory.SOCIAL_MEDIA)
            results["entertainment_connections"].extend(social_connections)
            results["evidence"]["social_media_connections"] = social_connections

        # Evaluate suspicious port activity
        if suspicious_ports:
            port_confidence = self._calculate_port_confidence(suspicious_ports)
            if port_confidence > 0.4:
                results["confidence_scores"].append(port_confidence)
                results["detected_categories"].append(ContentCategory.ENTERTAINMENT)
            results["evidence"]["suspicious_ports"] = suspicious_ports

        return results

    def _matches_domain_set(self, hostname: str, domain_set: set[str]) -> bool:
        """Check if hostname matches any domain in the set"""
        for domain in domain_set:
            if domain in hostname or hostname.endswith(f".{domain}"):
                return True
        return False

    def _is_suspicious_port(self, port: int, process_name: str) -> bool:
        """Check if port usage is suspicious for entertainment"""
        # High ports often used for P2P gaming or streaming
        if port > 49152:  # Dynamic/private ports
            # Check if it's a browser or known media process
            media_processes = ["chrome", "firefox", "vlc", "obs", "steam"]
            return any(proc in process_name.lower() for proc in media_processes)

        # Common streaming/game server ports
        streaming_ports = {80, 443, 8080, 8443, 1935, 8000, 8001}  # HTTP/RTMP/Media
        return port in streaming_ports

    def _calculate_gaming_confidence(self, gaming_connections: list[dict]) -> float:
        """Calculate confidence for gaming network activity"""
        if not gaming_connections:
            return 0.0

        base_confidence = 0.8  # High confidence for known gaming domains

        # Boost for multiple gaming connections
        multi_connection_boost = min(len(gaming_connections) * 0.05, 0.2)

        # Boost for dedicated gaming ports
        port_boost = (
            0.1 if any(conn.get("gaming_port") for conn in gaming_connections) else 0.0
        )

        return min(base_confidence + multi_connection_boost + port_boost, 1.0)

    def _calculate_streaming_confidence(
        self, streaming_connections: list[dict]
    ) -> float:
        """Calculate confidence for streaming network activity"""
        if not streaming_connections:
            return 0.0

        base_confidence = 0.7

        # Boost for CDN pattern matches (likely video content)
        cdn_boost = (
            0.2
            if any(conn.get("pattern_match") for conn in streaming_connections)
            else 0.0
        )

        # Boost for multiple streaming connections (multiple tabs/apps)
        multi_stream_boost = min(len(streaming_connections) * 0.1, 0.3)

        return min(base_confidence + cdn_boost + multi_stream_boost, 1.0)

    def _calculate_social_confidence(self, social_connections: list[dict]) -> float:
        """Calculate confidence for social media network activity"""
        if not social_connections:
            return 0.0

        # Lower base confidence as social media might be work-related
        base_confidence = 0.5

        # Boost for multiple social platforms
        multi_platform_boost = min(
            len({conn["host"] for conn in social_connections}) * 0.1, 0.3
        )

        return min(base_confidence + multi_platform_boost, 1.0)

    def _calculate_port_confidence(self, suspicious_ports: list[dict]) -> float:
        """Calculate confidence for suspicious port activity"""
        if not suspicious_ports:
            return 0.0

        # Lower confidence for port-based detection
        base_confidence = 0.3

        # Boost for high port numbers (P2P, gaming)
        high_port_boost = 0.0
        for conn in suspicious_ports:
            if conn["port"] > 49152:
                high_port_boost += 0.1

        return min(base_confidence + high_port_boost, 1.0)

    def _calculate_confidence(self, analysis_results: dict[str, Any]) -> float:
        """Calculate overall confidence from analysis results"""
        confidence_scores = analysis_results["confidence_scores"]
        if not confidence_scores:
            return 0.1

        # Use maximum confidence but consider multiple indicators
        max_confidence = max(confidence_scores)
        multi_indicator_boost = min(len(confidence_scores) * 0.05, 0.15)

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

    def _update_connection_history(self, connections: list[NetworkActivity]) -> None:
        """Update connection history for trend analysis"""
        # Keep last 100 connection records
        self._connection_history.extend(connections)
        if len(self._connection_history) > 100:
            self._connection_history = self._connection_history[-100:]

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
