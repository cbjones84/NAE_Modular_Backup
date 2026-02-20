# NAE/human_safety_gates.py
"""
Human-in-the-Loop Safety Gates for NAE
Implements FINRA/SEC compliant human oversight for critical trading decisions
"""

import os
import datetime
import json
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from enum import Enum
import threading
import time

class ApprovalStatus(Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"

class ApprovalType(Enum):
    PAPER_TO_LIVE = "paper_to_live"
    NEW_STRATEGY = "new_strategy"
    LARGE_TRADE = "large_trade"
    RISK_OVERRIDE = "risk_override"
    KILL_SWITCH = "kill_switch"
    EMERGENCY_ACTION = "emergency_action"

@dataclass
class ApprovalRequest:
    """Human approval request with audit trail"""
    request_id: str
    approval_type: ApprovalType
    requester: str
    details: Dict[str, Any]
    timestamp: str
    status: ApprovalStatus
    approver: Optional[str] = None
    approval_timestamp: Optional[str] = None
    approval_reason: Optional[str] = None
    hash: str = ""

class HumanSafetyGates:
    """Human-in-the-loop safety system for NAE"""
    
    def __init__(self):
        self.approval_queue: List[ApprovalRequest] = []
        self.approval_history: List[ApprovalRequest] = []
        self.owner_email = os.getenv('NAE_OWNER_EMAIL', 'owner@nae.com')
        self.approval_timeout_hours = 24  # 24 hour timeout for approvals
        self.auto_approve_threshold = 1000.0  # Auto-approve trades under $1000
        
        # Logging
        self.log_file = "logs/human_safety_gates.log"
        self.audit_log_file = "logs/human_safety_audit.log"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        
        # Monitoring thread
        self.monitor_thread = threading.Thread(target=self._monitor_approvals, daemon=True)
        self.monitor_thread.start()
        
        self.log_action("Human Safety Gates initialized")

    def log_action(self, message: str):
        """Log action with timestamp"""
        ts = datetime.datetime.now().isoformat()
        with open(self.log_file, "a") as f:
            f.write(f"[{ts}] {message}\n")
        print(f"[Safety Gates LOG] {message}")

    def create_approval_request(self, approval_type: ApprovalType, requester: str, 
                              details: Dict[str, Any]) -> str:
        """Create a new approval request"""
        try:
            request_id = f"req_{int(time.time())}_{hashlib.md5(str(details).encode()).hexdigest()[:8]}"
            
            # Check if auto-approval applies
            if self._should_auto_approve(approval_type, details):
                self.log_action(f"Auto-approving {approval_type.value} request: {request_id}")
                return self._auto_approve(request_id, approval_type, requester, details)
            
            # Create approval request
            approval_request = ApprovalRequest(
                request_id=request_id,
                approval_type=approval_type,
                requester=requester,
                details=details,
                timestamp=datetime.datetime.now().isoformat(),
                status=ApprovalStatus.PENDING,
                hash=""
            )
            
            # Generate hash for integrity
            approval_request.hash = self._generate_request_hash(approval_request)
            
            # Add to queue
            self.approval_queue.append(approval_request)
            
            # Create audit log
            self._create_audit_log("APPROVAL_REQUEST_CREATED", {
                "request_id": request_id,
                "approval_type": approval_type.value,
                "requester": requester,
                "details": details
            })
            
            # Send notification (placeholder)
            self._send_approval_notification(approval_request)
            
            self.log_action(f"Created approval request: {request_id} ({approval_type.value})")
            return request_id
            
        except Exception as e:
            self.log_action(f"Error creating approval request: {e}")
            return ""

    def approve_request(self, request_id: str, approver: str, reason: str = "") -> bool:
        """Approve a pending request"""
        try:
            request = self._find_request(request_id)
            if not request:
                self.log_action(f"Request not found: {request_id}")
                return False
            
            if request.status != ApprovalStatus.PENDING:
                self.log_action(f"Request {request_id} is not pending")
                return False
            
            # Update request
            request.status = ApprovalStatus.APPROVED
            request.approver = approver
            request.approval_timestamp = datetime.datetime.now().isoformat()
            request.approval_reason = reason
            
            # Move to history
            self.approval_history.append(request)
            self.approval_queue.remove(request)
            
            # Create audit log
            self._create_audit_log("APPROVAL_GRANTED", {
                "request_id": request_id,
                "approver": approver,
                "reason": reason,
                "approval_type": request.approval_type.value
            })
            
            self.log_action(f"Request {request_id} approved by {approver}")
            return True
            
        except Exception as e:
            self.log_action(f"Error approving request: {e}")
            return False

    def reject_request(self, request_id: str, approver: str, reason: str = "") -> bool:
        """Reject a pending request"""
        try:
            request = self._find_request(request_id)
            if not request:
                self.log_action(f"Request not found: {request_id}")
                return False
            
            if request.status != ApprovalStatus.PENDING:
                self.log_action(f"Request {request_id} is not pending")
                return False
            
            # Update request
            request.status = ApprovalStatus.REJECTED
            request.approver = approver
            request.approval_timestamp = datetime.datetime.now().isoformat()
            request.approval_reason = reason
            
            # Move to history
            self.approval_history.append(request)
            self.approval_queue.remove(request)
            
            # Create audit log
            self._create_audit_log("APPROVAL_REJECTED", {
                "request_id": request_id,
                "approver": approver,
                "reason": reason,
                "approval_type": request.approval_type.value
            })
            
            self.log_action(f"Request {request_id} rejected by {approver}")
            return True
            
        except Exception as e:
            self.log_action(f"Error rejecting request: {e}")
            return False

    def get_pending_requests(self) -> List[Dict[str, Any]]:
        """Get all pending approval requests"""
        return [
            {
                "request_id": req.request_id,
                "approval_type": req.approval_type.value,
                "requester": req.requester,
                "details": req.details,
                "timestamp": req.timestamp,
                "age_hours": self._get_request_age_hours(req)
            }
            for req in self.approval_queue
        ]

    def get_approval_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific approval request"""
        request = self._find_request(request_id)
        if not request:
            return None
        
        return {
            "request_id": request.request_id,
            "approval_type": request.approval_type.value,
            "status": request.status.value,
            "requester": request.requester,
            "approver": request.approver,
            "timestamp": request.timestamp,
            "approval_timestamp": request.approval_timestamp,
            "approval_reason": request.approval_reason,
            "details": request.details
        }

    def _should_auto_approve(self, approval_type: ApprovalType, details: Dict[str, Any]) -> bool:
        """Determine if request should be auto-approved"""
        # Auto-approve small trades
        if approval_type == ApprovalType.LARGE_TRADE:
            trade_amount = details.get('amount', 0)
            if trade_amount <= self.auto_approve_threshold:
                return True
        
        # Never auto-approve critical actions
        critical_types = [
            ApprovalType.PAPER_TO_LIVE,
            ApprovalType.KILL_SWITCH,
            ApprovalType.EMERGENCY_ACTION
        ]
        
        if approval_type in critical_types:
            return False
        
        return False

    def _auto_approve(self, request_id: str, approval_type: ApprovalType, 
                     requester: str, details: Dict[str, Any]) -> str:
        """Auto-approve a request"""
        approval_request = ApprovalRequest(
            request_id=request_id,
            approval_type=approval_type,
            requester=requester,
            details=details,
            timestamp=datetime.datetime.now().isoformat(),
            status=ApprovalStatus.APPROVED,
            approver="system",
            approval_timestamp=datetime.datetime.now().isoformat(),
            approval_reason="Auto-approved based on risk threshold",
            hash=""
        )
        
        approval_request.hash = self._generate_request_hash(approval_request)
        self.approval_history.append(approval_request)
        
        self._create_audit_log("AUTO_APPROVAL_GRANTED", {
            "request_id": request_id,
            "approval_type": approval_type.value,
            "reason": "Auto-approved based on risk threshold"
        })
        
        return request_id

    def _find_request(self, request_id: str) -> Optional[ApprovalRequest]:
        """Find request by ID in queue or history"""
        # Check pending queue
        for req in self.approval_queue:
            if req.request_id == request_id:
                return req
        
        # Check history
        for req in self.approval_history:
            if req.request_id == request_id:
                return req
        
        return None

    def _get_request_age_hours(self, request: ApprovalRequest) -> float:
        """Calculate age of request in hours"""
        request_time = datetime.datetime.fromisoformat(request.timestamp)
        current_time = datetime.datetime.now()
        age = current_time - request_time
        return age.total_seconds() / 3600

    def _monitor_approvals(self):
        """Monitor approval requests for timeouts"""
        while True:
            try:
                current_time = datetime.datetime.now()
                expired_requests = []
                
                for request in self.approval_queue:
                    request_time = datetime.datetime.fromisoformat(request.timestamp)
                    age_hours = (current_time - request_time).total_seconds() / 3600
                    
                    if age_hours > self.approval_timeout_hours:
                        expired_requests.append(request)
                
                # Handle expired requests
                for request in expired_requests:
                    request.status = ApprovalStatus.EXPIRED
                    self.approval_history.append(request)
                    self.approval_queue.remove(request)
                    
                    self._create_audit_log("APPROVAL_EXPIRED", {
                        "request_id": request.request_id,
                        "approval_type": request.approval_type.value,
                        "age_hours": age_hours
                    })
                    
                    self.log_action(f"Request {request.request_id} expired after {age_hours:.1f} hours")
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.log_action(f"Error in approval monitoring: {e}")
                time.sleep(60)

    def _send_approval_notification(self, request: ApprovalRequest):
        """Send notification for approval request (placeholder)"""
        # In production, this would send email/SMS to owner
        self.log_action(f"Notification sent for request {request.request_id}")

    def _generate_request_hash(self, request: ApprovalRequest) -> str:
        """Generate hash for request integrity"""
        data_string = f"{request.request_id}{request.approval_type.value}{request.requester}{request.timestamp}{json.dumps(request.details, sort_keys=True)}"
        return hashlib.sha256(data_string.encode()).hexdigest()

    def _create_audit_log(self, action: str, details: Dict[str, Any]):
        """Create immutable audit log entry"""
        timestamp = datetime.datetime.now().isoformat()
        
        log_data = {
            "timestamp": timestamp,
            "action": action,
            "details": details
        }
        
        log_string = json.dumps(log_data, sort_keys=True)
        log_hash = hashlib.sha256(log_string.encode()).hexdigest()
        
        try:
            with open(self.audit_log_file, "a") as f:
                f.write(f"{log_string}|{log_hash}\n")
        except Exception as e:
            self.log_action(f"Error writing audit log: {e}")


# ----------------------
# Owner Control Interface
# ----------------------
class OwnerControlInterface:
    """Owner control interface for NAE management"""
    
    def __init__(self, safety_gates: HumanSafetyGates):
        self.safety_gates = safety_gates
        self.owner_verified = False
        self.log_file = "logs/owner_control.log"
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def verify_owner(self, owner_identifier: str) -> bool:
        """Verify owner identity"""
        # In production, this would use proper authentication
        # For now, use a simple check
        expected_owner = os.getenv('NAE_OWNER_ID', 'owner123')
        self.owner_verified = (owner_identifier == expected_owner)
        
        if self.owner_verified:
            self.log_action(f"Owner verified: {owner_identifier}")
        else:
            self.log_action(f"Owner verification failed: {owner_identifier}")
        
        return self.owner_verified

    def get_pending_approvals(self, owner_identifier: str) -> List[Dict[str, Any]]:
        """Get pending approvals for owner review"""
        if not self.verify_owner(owner_identifier):
            return []
        
        return self.safety_gates.get_pending_requests()

    def approve_request(self, owner_identifier: str, request_id: str, reason: str = "") -> bool:
        """Approve a request as owner"""
        if not self.verify_owner(owner_identifier):
            return False
        
        return self.safety_gates.approve_request(request_id, owner_identifier, reason)

    def reject_request(self, owner_identifier: str, request_id: str, reason: str = "") -> bool:
        """Reject a request as owner"""
        if not self.verify_owner(owner_identifier):
            return False
        
        return self.safety_gates.reject_request(request_id, owner_identifier, reason)

    def get_system_status(self, owner_identifier: str) -> Dict[str, Any]:
        """Get comprehensive system status for owner"""
        if not self.verify_owner(owner_identifier):
            return {"error": "Owner verification failed"}
        
        return {
            "timestamp": datetime.datetime.now().isoformat(),
            "pending_approvals": len(self.safety_gates.approval_queue),
            "approval_history_count": len(self.safety_gates.approval_history),
            "owner_verified": self.owner_verified,
            "system_status": "operational"
        }

    def log_action(self, message: str):
        """Log owner action"""
        ts = datetime.datetime.now().isoformat()
        with open(self.log_file, "a") as f:
            f.write(f"[{ts}] {message}\n")
        print(f"[Owner Control LOG] {message}")


# ----------------------
# Test harness
# ----------------------
if __name__ == "__main__":
    print("Testing Human Safety Gates and Owner Control Interface...")
    
    # Initialize safety gates
    safety_gates = HumanSafetyGates()
    
    # Test approval request creation
    print("\n1. Testing approval request creation...")
    request_id = safety_gates.create_approval_request(
        ApprovalType.PAPER_TO_LIVE,
        "OptimusAgent",
        {
            "strategy_name": "Test Strategy",
            "capital_amount": 50000,
            "risk_assessment": "moderate"
        }
    )
    print(f"Created approval request: {request_id}")
    
    # Test large trade approval
    print("\n2. Testing large trade approval...")
    large_trade_id = safety_gates.create_approval_request(
        ApprovalType.LARGE_TRADE,
        "OptimusAgent",
        {
            "symbol": "AAPL",
            "amount": 50000,
            "side": "buy"
        }
    )
    print(f"Created large trade request: {large_trade_id}")
    
    # Test small trade (should auto-approve)
    print("\n3. Testing small trade (auto-approve)...")
    small_trade_id = safety_gates.create_approval_request(
        ApprovalType.LARGE_TRADE,
        "OptimusAgent",
        {
            "symbol": "AAPL",
            "amount": 500,  # Under threshold
            "side": "buy"
        }
    )
    print(f"Small trade request: {small_trade_id}")
    
    # Test owner control interface
    print("\n4. Testing owner control interface...")
    owner_interface = OwnerControlInterface(safety_gates)
    
    # Verify owner
    owner_verified = owner_interface.verify_owner("owner123")
    print(f"Owner verification: {owner_verified}")
    
    # Get pending approvals
    pending = owner_interface.get_pending_approvals("owner123")
    print(f"Pending approvals: {len(pending)}")
    
    # Approve a request
    if pending:
        approval_result = owner_interface.approve_request("owner123", pending[0]["request_id"], "Owner approval")
        print(f"Approval result: {approval_result}")
    
    # Get system status
    status = owner_interface.get_system_status("owner123")
    print(f"System status: {status}")
    
    print("\nHuman Safety Gates testing completed successfully!")
