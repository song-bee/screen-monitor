# Security Guidelines
## Advanced Screen Activity Monitor (ASAM)

### 1. SECURITY ARCHITECTURE OVERVIEW

ASAM implements a multi-layered security architecture to protect against tampering, bypass attempts, and unauthorized access while maintaining user privacy and system integrity.

#### 1.1 Security Principles
- **Defense in Depth**: Multiple overlapping security measures
- **Least Privilege**: Minimal required permissions for each component
- **Privacy by Design**: Local processing, minimal data collection
- **Fail-Safe Defaults**: Secure defaults with explicit opt-in for features
- **Continuous Monitoring**: Real-time threat detection and response

#### 1.2 Threat Model

**Identified Threats:**
1. **Service Termination**: Attempts to stop/disable the monitoring service
2. **Extension Tampering**: Disabling/removing browser extensions
3. **Configuration Manipulation**: Unauthorized changes to settings
4. **Process Injection**: Code injection into running processes
5. **File System Attacks**: Modification of service files
6. **Network Interception**: Man-in-the-middle attacks (future remote features)
7. **Privilege Escalation**: Attempts to gain administrative access

### 2. SERVICE PROTECTION MECHANISMS

#### 2.1 Process Protection
```python
class ServiceProtector:
    def __init__(self):
        self.watchdog_processes = []
        self.parent_process_id = os.getppid()
        self.service_executable = sys.executable
        
    async def enable_protection(self):
        """Enable comprehensive service protection"""
        await self._create_watchdog_processes()
        await self._register_termination_handlers()
        await self._verify_parent_process()
        await self._enable_file_monitoring()
        
    async def _create_watchdog_processes(self):
        """Create multiple watchdog processes for redundancy"""
        watchdog_count = 2  # Primary and backup watchdog
        
        for i in range(watchdog_count):
            watchdog = await self._spawn_watchdog(f"watchdog_{i}")
            self.watchdog_processes.append(watchdog)
            
    async def _spawn_watchdog(self, name: str) -> subprocess.Popen:
        """Spawn an independent watchdog process"""
        watchdog_script = f"""
import os
import time
import psutil
import subprocess
import sys

SERVICE_PID = {os.getpid()}
SERVICE_PATH = '{self.service_executable}'
WATCHDOG_NAME = '{name}'

def restart_service():
    try:
        subprocess.Popen([
            SERVICE_PATH, '--restart', 
            '--watchdog-restart', WATCHDOG_NAME
        ])
    except Exception as e:
        # Log error and attempt alternative restart
        with open('/tmp/asam_watchdog_error.log', 'a') as f:
            f.write(f'{{time.time()}}: Restart failed: {{e}}\\n')

while True:
    try:
        if not psutil.pid_exists(SERVICE_PID):
            restart_service()
            break
            
        # Verify service integrity
        proc = psutil.Process(SERVICE_PID)
        if proc.exe() != SERVICE_PATH:
            # Process path changed - possible hijacking
            restart_service()
            break
            
    except Exception as e:
        # Service process issues - restart
        restart_service()
        break
        
    time.sleep(5)
"""
        
        return subprocess.Popen([
            sys.executable, '-c', watchdog_script
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
```

#### 2.2 File Integrity Monitoring
```python
class IntegrityChecker:
    def __init__(self):
        self.file_hashes = {}
        self.monitored_files = [
            sys.executable,
            'config.yaml',
            'detection_rules.json'
        ]
        
    async def initialize_baseline(self):
        """Create integrity baseline for monitored files"""
        for file_path in self.monitored_files:
            if os.path.exists(file_path):
                self.file_hashes[file_path] = self._calculate_hash(file_path)
                
    async def verify_integrity(self) -> List[IntegrityViolation]:
        """Check file integrity against baseline"""
        violations = []
        
        for file_path, expected_hash in self.file_hashes.items():
            if not os.path.exists(file_path):
                violations.append(IntegrityViolation(
                    file_path=file_path,
                    violation_type="FILE_MISSING",
                    expected_hash=expected_hash,
                    actual_hash=None
                ))
                continue
                
            actual_hash = self._calculate_hash(file_path)
            if actual_hash != expected_hash:
                violations.append(IntegrityViolation(
                    file_path=file_path,
                    violation_type="HASH_MISMATCH",
                    expected_hash=expected_hash,
                    actual_hash=actual_hash
                ))
                
        return violations
        
    def _calculate_hash(self, file_path: str) -> str:
        """Calculate SHA-256 hash of file"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hasher.update(chunk)
        return hasher.hexdigest()
```

### 3. BROWSER EXTENSION SECURITY

#### 3.1 Extension Monitoring
```python
class ExtensionMonitor:
    def __init__(self):
        self.extension_paths = {
            'chrome': self._get_chrome_extension_path(),
            'firefox': self._get_firefox_extension_path()
        }
        self.monitoring_active = False
        
    async def start_monitoring(self):
        """Begin monitoring browser extensions"""
        self.monitoring_active = True
        
        # Monitor extension files
        asyncio.create_task(self._monitor_extension_files())
        
        # Monitor browser processes
        asyncio.create_task(self._monitor_browser_processes())
        
        # Monitor native messaging
        asyncio.create_task(self._monitor_native_messaging())
        
    async def _monitor_extension_files(self):
        """Monitor extension file integrity"""
        while self.monitoring_active:
            for browser, path in self.extension_paths.items():
                if not os.path.exists(path):
                    await self._handle_extension_removal(browser, path)
                else:
                    # Verify manifest integrity
                    manifest_path = os.path.join(path, 'manifest.json')
                    if not self._verify_manifest(manifest_path):
                        await self._handle_extension_tampering(browser, manifest_path)
                        
            await asyncio.sleep(10)  # Check every 10 seconds
            
    async def _handle_extension_removal(self, browser: str, path: str):
        """Handle detected extension removal"""
        security_event = SecurityEvent(
            event_type="extension_removal",
            browser=browser,
            path=path,
            severity="HIGH",
            timestamp=datetime.now()
        )
        
        await self._log_security_event(security_event)
        await self._send_security_alert(security_event)
        
        # Attempt to reinstall extension (if possible)
        await self._attempt_extension_recovery(browser, path)
```

#### 3.2 Native Messaging Security
```javascript
// Extension security measures
class ExtensionSecurity {
    constructor() {
        this.heartbeatInterval = null;
        this.tamperDetected = false;
        this.securityToken = this.generateSecurityToken();
    }
    
    initialize() {
        this.startHeartbeat();
        this.enableTamperDetection();
        this.validateEnvironment();
    }
    
    startHeartbeat() {
        // Send regular heartbeat to native app
        this.heartbeatInterval = setInterval(() => {
            if (!this.tamperDetected) {
                this.sendNativeMessage({
                    type: 'heartbeat',
                    token: this.securityToken,
                    timestamp: Date.now(),
                    integrity: this.calculateIntegrityHash()
                });
            }
        }, 30000); // Every 30 seconds
    }
    
    enableTamperDetection() {
        // Monitor for devtools
        let devtools = {open: false, orientation: null};
        
        setInterval(() => {
            if (window.outerHeight - window.innerHeight > 160 ||
                window.outerWidth - window.innerWidth > 160) {
                if (!devtools.open) {
                    devtools.open = true;
                    this.handleTamperAttempt('devtools_opened');
                }
            } else {
                devtools.open = false;
            }
        }, 500);
        
        // Monitor for script injection
        const originalAppendChild = Node.prototype.appendChild;
        Node.prototype.appendChild = function(child) {
            if (child.tagName === 'SCRIPT' && 
                !child.src.startsWith('chrome-extension://')) {
                this.handleTamperAttempt('script_injection');
            }
            return originalAppendChild.call(this, child);
        }.bind(this);
    }
    
    handleTamperAttempt(type) {
        this.tamperDetected = true;
        this.sendNativeMessage({
            type: 'security_alert',
            alert_type: 'tamper_attempt',
            details: type,
            timestamp: Date.now()
        });
    }
    
    calculateIntegrityHash() {
        // Calculate hash of critical extension files
        const manifest = chrome.runtime.getManifest();
        return btoa(JSON.stringify(manifest)).slice(0, 32);
    }
}
```

### 4. DATA PROTECTION AND PRIVACY

#### 4.1 Local Data Encryption
```python
class DataProtection:
    def __init__(self):
        self.encryption_key = self._derive_encryption_key()
        
    def _derive_encryption_key(self) -> bytes:
        """Derive encryption key from system-specific data"""
        # Use system-specific information for key derivation
        system_info = f"{platform.node()}{platform.machine()}{os.getuid()}"
        
        # Derive key using PBKDF2
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'asam_salt_v1',  # Fixed salt for consistency
            iterations=100000,
            backend=default_backend()
        )
        
        return kdf.derive(system_info.encode())
        
    def encrypt_sensitive_data(self, data: dict) -> bytes:
        """Encrypt sensitive configuration data"""
        json_data = json.dumps(data, sort_keys=True)
        
        # Use Fernet for symmetric encryption
        fernet = Fernet(base64.urlsafe_b64encode(self.encryption_key))
        encrypted_data = fernet.encrypt(json_data.encode())
        
        return encrypted_data
        
    def decrypt_sensitive_data(self, encrypted_data: bytes) -> dict:
        """Decrypt sensitive configuration data"""
        fernet = Fernet(base64.urlsafe_b64encode(self.encryption_key))
        decrypted_data = fernet.decrypt(encrypted_data)
        
        return json.loads(decrypted_data.decode())
```

#### 4.2 Privacy Protection
```python
class PrivacyProtection:
    def __init__(self):
        self.pii_patterns = [
            r'\b\d{3}-\d{2}-\d{4}\b',  # SSN
            r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b',  # Credit card
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'  # Email
        ]
        
    def sanitize_content(self, content: str) -> str:
        """Remove PII from content before analysis"""
        sanitized = content
        
        for pattern in self.pii_patterns:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)
            
        return sanitized
        
    def sanitize_logs(self, log_entry: dict) -> dict:
        """Sanitize log entries before storage"""
        sanitized_entry = log_entry.copy()
        
        # Remove sensitive fields
        sensitive_fields = ['password', 'token', 'key', 'secret']
        for field in sensitive_fields:
            if field in sanitized_entry:
                sanitized_entry[field] = '[REDACTED]'
                
        # Sanitize text content
        if 'content' in sanitized_entry:
            sanitized_entry['content'] = self.sanitize_content(
                sanitized_entry['content']
            )
            
        return sanitized_entry
```

### 5. PRIVILEGE MANAGEMENT

#### 5.1 Least Privilege Implementation
```python
class PrivilegeManager:
    def __init__(self):
        self.required_permissions = {
            'macos': [
                'com.apple.security.automation.apple-events',
                'com.apple.security.device.audio-input',
                'com.apple.security.personal-information.location'
            ],
            'windows': [
                'SeServiceLogonRight',
                'SeIncreaseQuotaPrivilege'
            ]
        }
        
    async def request_minimal_permissions(self):
        """Request only required permissions"""
        platform_name = platform.system().lower()
        
        if platform_name == 'darwin':
            await self._request_macos_permissions()
        elif platform_name == 'windows':
            await self._request_windows_permissions()
            
    async def _request_macos_permissions(self):
        """Request macOS-specific permissions"""
        # Request screen recording permission
        if not self._has_screen_recording_permission():
            await self._prompt_screen_recording_permission()
            
        # Request accessibility permission
        if not self._has_accessibility_permission():
            await self._prompt_accessibility_permission()
            
    def _has_screen_recording_permission(self) -> bool:
        """Check if screen recording permission is granted"""
        try:
            # Attempt screen capture to test permission
            ImageGrab.grab()
            return True
        except Exception:
            return False
            
    def drop_unnecessary_privileges(self):
        """Drop privileges that are no longer needed"""
        if hasattr(os, 'setuid') and os.getuid() == 0:
            # Running as root - drop to normal user
            os.setuid(1000)  # Drop to first normal user UID
```

### 6. NETWORK SECURITY (FUTURE FEATURES)

#### 6.1 TLS Configuration
```python
class NetworkSecurity:
    def __init__(self):
        self.tls_context = self._create_secure_tls_context()
        
    def _create_secure_tls_context(self) -> ssl.SSLContext:
        """Create secure TLS context"""
        context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        
        # Require TLS 1.3 minimum
        context.minimum_version = ssl.TLSVersion.TLSv1_3
        context.maximum_version = ssl.TLSVersion.TLSv1_3
        
        # Disable weak ciphers
        context.set_ciphers('ECDHE+AESGCM:ECDHE+CHACHA20:DHE+AESGCM')
        
        # Enable certificate verification
        context.check_hostname = True
        context.verify_mode = ssl.CERT_REQUIRED
        
        return context
        
    def verify_server_certificate(self, hostname: str, cert_chain: list) -> bool:
        """Verify server certificate with certificate pinning"""
        # Implement certificate pinning for known servers
        pinned_certificates = {
            'api.asam.com': 'sha256:expected_cert_hash_here'
        }
        
        if hostname in pinned_certificates:
            cert_hash = hashlib.sha256(cert_chain[0]).hexdigest()
            expected_hash = pinned_certificates[hostname].replace('sha256:', '')
            return cert_hash == expected_hash
            
        return True  # Allow other certificates through normal validation
```

### 7. SECURITY MONITORING AND RESPONSE

#### 7.1 Security Event Handling
```python
class SecurityEventHandler:
    def __init__(self):
        self.threat_level_thresholds = {
            'LOW': 1,
            'MEDIUM': 3,
            'HIGH': 5,
            'CRITICAL': 10
        }
        self.recent_events = deque(maxlen=100)
        
    async def handle_security_event(self, event: SecurityEvent):
        """Process security events and determine response"""
        self.recent_events.append(event)
        
        # Calculate threat score
        threat_score = self._calculate_threat_score(event)
        
        # Determine response based on threat level
        if threat_score >= self.threat_level_thresholds['CRITICAL']:
            await self._handle_critical_threat(event)
        elif threat_score >= self.threat_level_thresholds['HIGH']:
            await self._handle_high_threat(event)
        elif threat_score >= self.threat_level_thresholds['MEDIUM']:
            await self._handle_medium_threat(event)
        else:
            await self._log_security_event(event)
            
    async def _handle_critical_threat(self, event: SecurityEvent):
        """Handle critical security threats"""
        # Immediately lock screen
        await self.platform_adapter.lock_screen()
        
        # Send emergency notification
        await self.notification_service.send_emergency_alert(
            "Critical security threat detected",
            f"Threat type: {event.threat_type}"
        )
        
        # Log security event with high priority
        await self._log_security_event(event, priority='CRITICAL')
        
        # Attempt to restart service with enhanced protection
        await self._restart_with_enhanced_protection()
```

### 8. SECURITY BEST PRACTICES

#### 8.1 Development Security Guidelines
- **Code Review**: All security-related code requires peer review
- **Static Analysis**: Use tools like bandit, semgrep for vulnerability scanning
- **Dependency Scanning**: Regular updates and vulnerability checks for dependencies
- **Secure Coding**: Follow OWASP guidelines for secure development

#### 8.2 Deployment Security
- **Code Signing**: All binaries must be signed with valid certificates
- **Integrity Verification**: Package checksums and digital signatures
- **Secure Distribution**: Official channels only (Mac App Store, etc.)
- **Update Security**: Secure update mechanism with signature verification

#### 8.3 Operational Security
- **Log Monitoring**: Regular analysis of security logs
- **Incident Response**: Documented procedures for security incidents
- **Backup Security**: Encrypted backups with secure storage
- **Access Control**: Minimal administrative access requirements

### 9. COMPLIANCE AND AUDITING

#### 9.1 Security Audit Logging
```python
class SecurityAuditor:
    def __init__(self):
        self.audit_log = SecureLogger('security_audit')
        
    async def log_security_event(self, event: SecurityEvent):
        """Log security events for audit purposes"""
        audit_entry = {
            'timestamp': event.timestamp.isoformat(),
            'event_type': event.event_type,
            'severity': event.severity,
            'source': event.source,
            'details': event.details,
            'response_actions': event.response_actions,
            'system_state': await self._capture_system_state()
        }
        
        await self.audit_log.log_entry(audit_entry)
        
    async def generate_security_report(self, period: str) -> SecurityReport:
        """Generate security report for specified period"""
        events = await self.audit_log.get_events(period)
        
        return SecurityReport(
            period=period,
            total_events=len(events),
            events_by_severity=self._group_by_severity(events),
            threat_trends=self._analyze_threat_trends(events),
            recommendations=self._generate_recommendations(events)
        )
```

This security framework provides comprehensive protection for the ASAM system while maintaining usability and performance. The multi-layered approach ensures that even if one security measure is bypassed, others remain in place to protect the system integrity.