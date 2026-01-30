# NAE/tools/model_registry.py
"""
Model Registry & CI/CD for Models

Features:
- Model versioning
- Dataset snapshot tracking
- Reproducible training pipelines
- Canary deployment
- Model comparison
- Rollback capability
"""

import os
import json
import hashlib
import shutil
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum
import pickle


class ModelStatus(Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"


class DeploymentStatus(Enum):
    PENDING = "pending"
    CANARY = "canary"
    ROLLED_OUT = "rolled_out"
    ROLLED_BACK = "rolled_back"


@dataclass
class ModelMetadata:
    """Model metadata"""
    model_id: str
    model_name: str
    version: str
    status: ModelStatus
    created_at: datetime
    created_by: str
    description: str
    dataset_version: str
    dataset_hash: str
    training_config: Dict[str, Any] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    metrics: Dict[str, float] = field(default_factory=dict)
    file_path: Optional[str] = None
    checksum: Optional[str] = None


@dataclass
class DeploymentRecord:
    """Deployment record"""
    deployment_id: str
    model_id: str
    version: str
    status: DeploymentStatus
    deployed_at: datetime
    canary_traffic_pct: float = 0.0
    baseline_model_id: Optional[str] = None
    performance_comparison: Dict[str, Any] = field(default_factory=dict)
    rollback_reason: Optional[str] = None


class ModelRegistry:
    """
    Model registry for versioning and deployment
    """
    
    def __init__(self, registry_path: str = "models/registry"):
        self.registry_path = Path(registry_path)
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, ModelMetadata] = {}
        self.deployments: List[DeploymentRecord] = []
        self._load_registry()
    
    def register_model(
        self,
        model_name: str,
        model_object: Any,
        dataset_version: str,
        dataset_hash: str,
        training_config: Dict[str, Any],
        hyperparameters: Dict[str, Any],
        metrics: Dict[str, float],
        description: str = "",
        created_by: str = "system"
    ) -> ModelMetadata:
        """
        Register a new model version
        
        Args:
            model_name: Name of model
            model_object: Model object to save
            dataset_version: Version of dataset used
            dataset_hash: Hash of dataset
            training_config: Training configuration
            hyperparameters: Model hyperparameters
            metrics: Model performance metrics
            description: Model description
            created_by: Creator identifier
        """
        # Generate model ID
        timestamp = datetime.now()
        model_id = f"{model_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        # Determine version
        existing_versions = [m.version for m in self.models.values() if m.model_name == model_name]
        if existing_versions:
            latest_version = max([int(v.split('.')[-1]) for v in existing_versions if v.replace('.', '').isdigit()], default=0)
            version = f"1.{latest_version + 1}"
        else:
            version = "1.0"
        
        # Save model file
        model_dir = self.registry_path / model_name / version
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_file = model_dir / "model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model_object, f)
        
        # Calculate checksum
        checksum = self._calculate_checksum(model_file)
        
        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            model_name=model_name,
            version=version,
            status=ModelStatus.DEVELOPMENT,
            created_at=timestamp,
            created_by=created_by,
            description=description,
            dataset_version=dataset_version,
            dataset_hash=dataset_hash,
            training_config=training_config,
            hyperparameters=hyperparameters,
            metrics=metrics,
            file_path=str(model_file),
            checksum=checksum
        )
        
        # Save metadata
        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(asdict(metadata), f, indent=2, default=str)
        
        self.models[model_id] = metadata
        self._save_registry()
        
        return metadata
    
    def load_model(self, model_id: str) -> Tuple[Any, ModelMetadata]:
        """Load model and metadata"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        metadata = self.models[model_id]
        
        if not metadata.file_path or not os.path.exists(metadata.file_path):
            raise FileNotFoundError(f"Model file not found: {metadata.file_path}")
        
        with open(metadata.file_path, 'rb') as f:
            model = pickle.load(f)
        
        return model, metadata
    
    def promote_model(self, model_id: str, target_status: ModelStatus):
        """Promote model to new status"""
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        self.models[model_id].status = target_status
        self._save_registry()
    
    def deploy_canary(
        self,
        model_id: str,
        baseline_model_id: str,
        traffic_percentage: float = 0.10
    ) -> DeploymentRecord:
        """
        Deploy model as canary
        
        Args:
            model_id: Model to deploy
            baseline_model_id: Baseline model for comparison
            traffic_percentage: Percentage of traffic to route to canary (0-1)
        """
        if model_id not in self.models:
            raise ValueError(f"Model {model_id} not found")
        
        deployment_id = f"deploy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        deployment = DeploymentRecord(
            deployment_id=deployment_id,
            model_id=model_id,
            version=self.models[model_id].version,
            status=DeploymentStatus.CANARY,
            deployed_at=datetime.now(),
            canary_traffic_pct=traffic_percentage,
            baseline_model_id=baseline_model_id
        )
        
        self.deployments.append(deployment)
        self.models[model_id].status = ModelStatus.STAGING
        self._save_registry()
        
        return deployment
    
    def compare_models(
        self,
        model_id_1: str,
        model_id_2: str,
        metric: str = "sharpe_ratio"
    ) -> Dict[str, Any]:
        """Compare two models"""
        if model_id_1 not in self.models or model_id_2 not in self.models:
            raise ValueError("One or both models not found")
        
        m1 = self.models[model_id_1]
        m2 = self.models[model_id_2]
        
        metric_1 = m1.metrics.get(metric, 0.0)
        metric_2 = m2.metrics.get(metric, 0.0)
        
        improvement = metric_1 - metric_2
        improvement_pct = (improvement / metric_2 * 100) if metric_2 != 0 else 0.0
        
        return {
            "model_1": {
                "id": model_id_1,
                "version": m1.version,
                "metric": metric_1
            },
            "model_2": {
                "id": model_id_2,
                "version": m2.version,
                "metric": metric_2
            },
            "improvement": improvement,
            "improvement_pct": improvement_pct,
            "better_model": model_id_1 if improvement > 0 else model_id_2
        }
    
    def rollback_deployment(self, deployment_id: str, reason: str):
        """Rollback a deployment"""
        deployment = next((d for d in self.deployments if d.deployment_id == deployment_id), None)
        if not deployment:
            raise ValueError(f"Deployment {deployment_id} not found")
        
        deployment.status = DeploymentStatus.ROLLED_BACK
        deployment.rollback_reason = reason
        
        # Revert model status
        if deployment.model_id in self.models:
            self.models[deployment.model_id].status = ModelStatus.ARCHIVED
        
        self._save_registry()
    
    def get_production_models(self) -> List[ModelMetadata]:
        """Get all production models"""
        return [m for m in self.models.values() if m.status == ModelStatus.PRODUCTION]
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum"""
        with open(file_path, 'rb') as f:
            return hashlib.sha256(f.read()).hexdigest()
    
    def _save_registry(self):
        """Save registry to disk"""
        registry_file = self.registry_path / "registry.json"
        
        registry_data = {
            "models": {mid: asdict(m) for mid, m in self.models.items()},
            "deployments": [asdict(d) for d in self.deployments]
        }
        
        with open(registry_file, 'w') as f:
            json.dump(registry_data, f, indent=2, default=str)
    
    def _load_registry(self):
        """Load registry from disk"""
        registry_file = self.registry_path / "registry.json"
        
        if not registry_file.exists():
            return
        
        with open(registry_file) as f:
            registry_data = json.load(f)
        
        # Load models
        for mid, mdata in registry_data.get("models", {}).items():
            mdata["created_at"] = datetime.fromisoformat(mdata["created_at"])
            mdata["status"] = ModelStatus(mdata["status"])
            self.models[mid] = ModelMetadata(**mdata)
        
        # Load deployments
        for ddata in registry_data.get("deployments", []):
            ddata["deployed_at"] = datetime.fromisoformat(ddata["deployed_at"])
            ddata["status"] = DeploymentStatus(ddata["status"])
            self.deployments.append(DeploymentRecord(**ddata))


# Global registry instance
_model_registry: Optional[ModelRegistry] = None


def get_model_registry() -> ModelRegistry:
    """Get or create global model registry"""
    global _model_registry
    if _model_registry is None:
        _model_registry = ModelRegistry()
    return _model_registry

