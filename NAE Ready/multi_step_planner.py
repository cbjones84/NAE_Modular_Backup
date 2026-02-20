# NAE/multi_step_planner.py
"""
Multi-Step Planning and Execution System
Implements AutoGen-style planning, execution, and verification
"""

import os
import json
import time
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class StepStatus(Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class PlanStep:
    """Single step in a multi-step plan"""
    step_id: int
    name: str
    description: str
    agent: str
    action: str
    parameters: Dict[str, Any]
    dependencies: List[int]
    status: StepStatus = StepStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    duration: float = 0.0

@dataclass
class ExecutionPlan:
    """Multi-step execution plan"""
    plan_id: str
    goal: str
    steps: List[PlanStep]
    created_at: float
    status: StepStatus = StepStatus.PENDING
    results: Dict[str, Any] = None

class MultiStepPlanner:
    """Plan, execute, and verify multi-step operations"""
    
    def __init__(self):
        self.plans: Dict[str, ExecutionPlan] = {}
        self.execution_history: List[ExecutionPlan] = []
    
    def create_plan(self, goal: str, steps: List[Dict[str, Any]]) -> ExecutionPlan:
        """Create a multi-step execution plan"""
        plan_id = f"plan_{int(time.time())}"
        
        plan_steps = []
        for i, step_data in enumerate(steps):
            step = PlanStep(
                step_id=i + 1,
                name=step_data.get("name", f"Step {i+1}"),
                description=step_data.get("description", ""),
                agent=step_data.get("agent", ""),
                action=step_data.get("action", ""),
                parameters=step_data.get("parameters", {}),
                dependencies=step_data.get("dependencies", [])
            )
            plan_steps.append(step)
        
        plan = ExecutionPlan(
            plan_id=plan_id,
            goal=goal,
            steps=plan_steps,
            created_at=time.time(),
            status=StepStatus.PENDING
        )
        
        self.plans[plan_id] = plan
        return plan
    
    def execute_plan(self, plan_id: str) -> Dict[str, Any]:
        """Execute a multi-step plan"""
        if plan_id not in self.plans:
            return {"error": f"Plan {plan_id} not found"}
        
        plan = self.plans[plan_id]
        plan.status = StepStatus.IN_PROGRESS
        
        results = {}
        completed_steps = set()
        
        try:
            # Execute steps in dependency order
            while len(completed_steps) < len(plan.steps):
                # Find steps ready to execute
                ready_steps = []
                for step in plan.steps:
                    if step.status == StepStatus.PENDING:
                        # Check dependencies
                        if all(dep in completed_steps for dep in step.dependencies):
                            ready_steps.append(step)
                
                if not ready_steps:
                    # Check for circular dependencies or deadlock
                    if all(step.status != StepStatus.PENDING for step in plan.steps):
                        break
                    else:
                        return {"error": "Circular dependency or deadlock detected"}
                
                # Execute ready steps
                for step in ready_steps:
                    step_result = self._execute_step(step)
                    step.status = step_result["status"]
                    step.result = step_result.get("result")
                    step.error = step_result.get("error")
                    step.duration = step_result.get("duration", 0.0)
                    
                    if step.status == StepStatus.COMPLETED:
                        completed_steps.add(step.step_id)
                        results[step.step_id] = step.result
                    else:
                        # Step failed - decide if we should continue
                        if step.status == StepStatus.FAILED:
                            plan.status = StepStatus.FAILED
                            return {
                                "status": "failed",
                                "failed_step": step.step_id,
                                "error": step.error,
                                "results": results
                            }
            
            plan.status = StepStatus.COMPLETED
            plan.results = results
            
            return {
                "status": "completed",
                "plan_id": plan_id,
                "results": results
            }
            
        except Exception as e:
            plan.status = StepStatus.FAILED
            return {
                "status": "error",
                "error": str(e)
            }
    
    def _execute_step(self, step: PlanStep) -> Dict[str, Any]:
        """Execute a single plan step"""
        start_time = time.time()
        
        try:
            # Import command executor
            from command_executor import get_executor
            executor = get_executor()
            
            # Execute based on action type
            if step.action == "execute_method":
                result = executor.execute_agent_method(
                    step.agent,
                    step.parameters.get("method"),
                    step.parameters.get("args", []),
                    step.parameters.get("kwargs", {})
                )
                
                if result.status.value == "success":
                    return {
                        "status": StepStatus.COMPLETED,
                        "result": {"output": result.output},
                        "duration": time.time() - start_time
                    }
                else:
                    return {
                        "status": StepStatus.FAILED,
                        "error": result.error or result.output,
                        "duration": time.time() - start_time
                    }
            
            elif step.action == "execute_code":
                result = executor.execute_python_code(
                    step.parameters.get("code", ""),
                    step.parameters.get("context", {})
                )
                
                if result.status.value == "success":
                    return {
                        "status": StepStatus.COMPLETED,
                        "result": {"output": result.output},
                        "duration": time.time() - start_time
                    }
                else:
                    return {
                        "status": StepStatus.FAILED,
                        "error": result.error or result.output,
                        "duration": time.time() - start_time
                    }
            
            elif step.action == "execute_command":
                result = executor.execute_system_command(
                    step.parameters.get("command", "")
                )
                
                if result.status.value == "success":
                    return {
                        "status": StepStatus.COMPLETED,
                        "result": {"output": result.output},
                        "duration": time.time() - start_time
                    }
                else:
                    return {
                        "status": StepStatus.FAILED,
                        "error": result.error or result.output,
                        "duration": time.time() - start_time
                    }
            
            else:
                return {
                    "status": StepStatus.FAILED,
                    "error": f"Unknown action: {step.action}",
                    "duration": time.time() - start_time
                }
                
        except Exception as e:
            return {
                "status": StepStatus.FAILED,
                "error": str(e),
                "duration": time.time() - start_time
            }
    
    def verify_plan(self, plan_id: str) -> Dict[str, Any]:
        """Verify plan execution results"""
        if plan_id not in self.plans:
            return {"error": f"Plan {plan_id} not found"}
        
        plan = self.plans[plan_id]
        
        if plan.status != StepStatus.COMPLETED:
            return {
                "verified": False,
                "reason": f"Plan not completed: {plan.status.value}"
            }
        
        # Verify each step
        verification_results = {}
        all_verified = True
        
        for step in plan.steps:
            if step.status == StepStatus.COMPLETED:
                # Basic verification - check if result exists
                verified = step.result is not None
                verification_results[step.step_id] = verified
                if not verified:
                    all_verified = False
        
        return {
            "verified": all_verified,
            "plan_id": plan_id,
            "step_verifications": verification_results
        }
    
    def create_autogen_workflow(self, goal: str) -> ExecutionPlan:
        """Create an AutoGen-style workflow plan"""
        # Example: Ralph -> Casey -> Donnie -> Optimus workflow
        steps = [
            {
                "name": "Ralph Generate Strategies",
                "description": "Ralph generates trading strategies",
                "agent": "Ralph",
                "action": "execute_method",
                "parameters": {
                    "method": "generate_strategies",
                    "args": [],
                    "kwargs": {}
                },
                "dependencies": []
            },
            {
                "name": "Casey Review Strategies",
                "description": "Casey reviews and refines strategies",
                "agent": "Casey",
                "action": "execute_method",
                "parameters": {
                    "method": "review_strategies",
                    "args": [],
                    "kwargs": {}
                },
                "dependencies": [1]  # Depends on step 1
            },
            {
                "name": "Donnie Prepare Execution",
                "description": "Donnie prepares execution instructions",
                "agent": "Donnie",
                "action": "execute_method",
                "parameters": {
                    "method": "receive_strategies",
                    "args": [],
                    "kwargs": {}
                },
                "dependencies": [2]  # Depends on step 2
            },
            {
                "name": "Optimus Execute Trade",
                "description": "Optimus executes approved trades",
                "agent": "Optimus",
                "action": "execute_method",
                "parameters": {
                    "method": "execute_trade",
                    "args": [],
                    "kwargs": {}
                },
                "dependencies": [3]  # Depends on step 3
            }
        ]
        
        return self.create_plan(goal, steps)
    
    def get_plan_status(self, plan_id: str) -> Dict[str, Any]:
        """Get plan status"""
        if plan_id not in self.plans:
            return {"error": f"Plan {plan_id} not found"}
        
        plan = self.plans[plan_id]
        return {
            "plan_id": plan_id,
            "goal": plan.goal,
            "status": plan.status.value,
            "steps": [
                {
                    "step_id": step.step_id,
                    "name": step.name,
                    "status": step.status.value,
                    "duration": step.duration
                }
                for step in plan.steps
            ]
        }


# Global planner instance
_planner = None

def get_planner() -> MultiStepPlanner:
    """Get global multi-step planner"""
    global _planner
    if _planner is None:
        _planner = MultiStepPlanner()
    return _planner


if __name__ == "__main__":
    # Test multi-step planner
    planner = MultiStepPlanner()
    
    # Create AutoGen workflow
    plan = planner.create_autogen_workflow("Generate and execute trading strategy")
    
    print(f"Created plan: {plan.plan_id}")
    print(f"Goal: {plan.goal}")
    print(f"Steps: {len(plan.steps)}")
    
    # Execute plan
    result = planner.execute_plan(plan.plan_id)
    print(f"Execution result: {result}")


