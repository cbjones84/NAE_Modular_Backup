# NAE/tools/data_quality.py
"""
Data Quality & Lineage System for NAE

Features:
- Central data lake with immutable snapshots
- Data validation tests
- Schema checks
- Missing data detection
- Timestamp drift detection
- Outlier detection
- Data lineage tracking
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import json
import os
import hashlib
from enum import Enum


class DataQualityLevel(Enum):
    PASS = "pass"
    WARNING = "warning"
    FAIL = "fail"


@dataclass
class DataSnapshot:
    """Immutable data snapshot"""
    dataset_name: str
    timestamp: datetime
    data_hash: str
    schema: Dict[str, str]
    row_count: int
    columns: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    file_path: Optional[str] = None


@dataclass
class DataQualityCheck:
    """Data quality check result"""
    check_name: str
    passed: bool
    level: DataQualityLevel
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataLineage:
    """Data lineage tracking"""
    source_dataset: str
    target_dataset: str
    transformation: str
    timestamp: datetime
    parameters: Dict[str, Any] = field(default_factory=dict)


class DataLake:
    """
    Central data lake with immutable snapshots
    """
    
    def __init__(self, base_path: str = "data/lake"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.snapshots: List[DataSnapshot] = []
        self.lineage: List[DataLineage] = []
    
    def save_snapshot(
        self,
        data: pd.DataFrame,
        dataset_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> DataSnapshot:
        """
        Save immutable data snapshot
        
        Args:
            data: DataFrame to save
            dataset_name: Name of dataset
            metadata: Additional metadata
        """
        timestamp = datetime.now()
        
        # Calculate data hash
        data_hash = self._calculate_hash(data)
        
        # Create snapshot
        snapshot = DataSnapshot(
            dataset_name=dataset_name,
            timestamp=timestamp,
            data_hash=data_hash,
            schema={col: str(dtype) for col, dtype in data.dtypes.items()},
            row_count=len(data),
            columns=list(data.columns),
            metadata=metadata or {},
            file_path=None
        )
        
        # Save to parquet
        snapshot_dir = self.base_path / dataset_name / timestamp.strftime("%Y%m%d")
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{dataset_name}_{timestamp.strftime('%Y%m%d_%H%M%S')}_{data_hash[:8]}.parquet"
        filepath = snapshot_dir / filename
        
        data.to_parquet(filepath, index=True)
        snapshot.file_path = str(filepath)
        
        # Save snapshot metadata
        metadata_file = snapshot_dir / f"{filename}.meta.json"
        with open(metadata_file, 'w') as f:
            json.dump(asdict(snapshot), f, indent=2, default=str)
        
        self.snapshots.append(snapshot)
        
        return snapshot
    
    def load_snapshot(self, dataset_name: str, timestamp: Optional[datetime] = None) -> Optional[pd.DataFrame]:
        """Load data snapshot"""
        dataset_dir = self.base_path / dataset_name
        
        if not dataset_dir.exists():
            return None
        
        if timestamp:
            # Load specific timestamp
            date_dir = dataset_dir / timestamp.strftime("%Y%m%d")
            if date_dir.exists():
                files = sorted(date_dir.glob("*.parquet"))
                for file in files:
                    meta_file = file.with_suffix('.parquet.meta.json')
                    if meta_file.exists():
                        with open(meta_file) as f:
                            meta = json.load(f)
                            if meta['timestamp'] == timestamp.isoformat():
                                return pd.read_parquet(file)
        else:
            # Load latest
            date_dirs = sorted(dataset_dir.glob("*"), reverse=True)
            for date_dir in date_dirs:
                files = sorted(date_dir.glob("*.parquet"), reverse=True)
                if files:
                    return pd.read_parquet(files[0])
        
        return None
    
    def _calculate_hash(self, data: pd.DataFrame) -> str:
        """Calculate hash of data"""
        # Hash the data content (excluding index for reproducibility)
        data_str = data.to_string()
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    def record_lineage(
        self,
        source_dataset: str,
        target_dataset: str,
        transformation: str,
        parameters: Optional[Dict[str, Any]] = None
    ):
        """Record data lineage"""
        lineage = DataLineage(
            source_dataset=source_dataset,
            target_dataset=target_dataset,
            transformation=transformation,
            timestamp=datetime.now(),
            parameters=parameters or {}
        )
        
        self.lineage.append(lineage)
        
        # Save lineage
        lineage_file = self.base_path / "lineage.jsonl"
        with open(lineage_file, 'a') as f:
            f.write(json.dumps(asdict(lineage), default=str) + "\n")


class DataQualityValidator:
    """
    Data quality validation system
    """
    
    def __init__(self):
        self.checks_history: List[List[DataQualityCheck]] = []
    
    def validate(
        self,
        data: pd.DataFrame,
        schema: Optional[Dict[str, str]] = None,
        required_columns: Optional[List[str]] = None
    ) -> Tuple[bool, List[DataQualityCheck]]:
        """
        Validate data quality
        
        Returns:
            (is_valid, checks)
        """
        checks: List[DataQualityCheck] = []
        
        # Schema check
        if schema:
            schema_check = self._check_schema(data, schema)
            checks.append(schema_check)
        
        # Required columns check
        if required_columns:
            cols_check = self._check_required_columns(data, required_columns)
            checks.append(cols_check)
        
        # Missing data check
        missing_check = self._check_missing_data(data)
        checks.append(missing_check)
        
        # Timestamp drift check
        if 'timestamp' in data.columns or 'date' in data.columns:
            timestamp_check = self._check_timestamp_drift(data)
            checks.append(timestamp_check)
        
        # Outlier detection
        outlier_check = self._check_outliers(data)
        checks.append(outlier_check)
        
        # Duplicate check
        duplicate_check = self._check_duplicates(data)
        checks.append(duplicate_check)
        
        # Save checks
        self.checks_history.append(checks)
        
        # Overall validity
        is_valid = all(c.passed or c.level == DataQualityLevel.WARNING for c in checks)
        
        return is_valid, checks
    
    def _check_schema(self, data: pd.DataFrame, schema: Dict[str, str]) -> DataQualityCheck:
        """Check data schema"""
        mismatches = []
        for col, expected_type in schema.items():
            if col not in data.columns:
                mismatches.append(f"{col}: missing")
            else:
                actual_type = str(data[col].dtype)
                if expected_type not in actual_type and actual_type not in expected_type:
                    mismatches.append(f"{col}: expected {expected_type}, got {actual_type}")
        
        passed = len(mismatches) == 0
        return DataQualityCheck(
            check_name="schema_check",
            passed=passed,
            level=DataQualityLevel.FAIL if not passed else DataQualityLevel.PASS,
            message=f"Schema check: {len(mismatches)} mismatches" if mismatches else "Schema check passed",
            details={"mismatches": mismatches}
        )
    
    def _check_required_columns(self, data: pd.DataFrame, required: List[str]) -> DataQualityCheck:
        """Check required columns"""
        missing = [col for col in required if col not in data.columns]
        passed = len(missing) == 0
        
        return DataQualityCheck(
            check_name="required_columns",
            passed=passed,
            level=DataQualityLevel.FAIL if not passed else DataQualityLevel.PASS,
            message=f"Missing columns: {missing}" if missing else "All required columns present",
            details={"missing": missing}
        )
    
    def _check_missing_data(self, data: pd.DataFrame) -> DataQualityCheck:
        """Check for missing data"""
        missing_counts = data.isnull().sum()
        total_missing = missing_counts.sum()
        missing_pct = (total_missing / (len(data) * len(data.columns))) * 100
        
        passed = missing_pct < 5.0  # Allow up to 5% missing
        
        return DataQualityCheck(
            check_name="missing_data",
            passed=passed,
            level=DataQualityLevel.WARNING if 5.0 <= missing_pct < 10.0 else (DataQualityLevel.FAIL if missing_pct >= 10.0 else DataQualityLevel.PASS),
            message=f"Missing data: {missing_pct:.2f}% ({total_missing} values)",
            details={"missing_counts": missing_counts.to_dict(), "missing_pct": missing_pct}
        )
    
    def _check_timestamp_drift(self, data: pd.DataFrame) -> DataQualityCheck:
        """Check timestamp drift"""
        timestamp_col = 'timestamp' if 'timestamp' in data.columns else 'date'
        
        if timestamp_col not in data.columns:
            return DataQualityCheck(
                check_name="timestamp_drift",
                passed=True,
                level=DataQualityLevel.PASS,
                message="No timestamp column found",
                details={}
            )
        
        timestamps = pd.to_datetime(data[timestamp_col])
        now = datetime.now()
        
        # Check for future timestamps
        future_count = (timestamps > now).sum()
        
        # Check for very old timestamps (>1 year)
        old_threshold = now - timedelta(days=365)
        old_count = (timestamps < old_threshold).sum()
        
        # Check for gaps
        timestamps_sorted = timestamps.sort_values()
        gaps = timestamps_sorted.diff().dropna()
        large_gaps = (gaps > timedelta(days=7)).sum()  # Gaps > 7 days
        
        passed = future_count == 0 and large_gaps < len(data) * 0.1
        
        return DataQualityCheck(
            check_name="timestamp_drift",
            passed=passed,
            level=DataQualityLevel.WARNING if not passed else DataQualityLevel.PASS,
            message=f"Timestamp issues: {future_count} future, {large_gaps} large gaps",
            details={"future_count": int(future_count), "old_count": int(old_count), "large_gaps": int(large_gaps)}
        )
    
    def _check_outliers(self, data: pd.DataFrame) -> DataQualityCheck:
        """Check for outliers using IQR method"""
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        outlier_counts = {}
        
        for col in numeric_cols:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
                outlier_counts[col] = int(outliers)
        
        total_outliers = sum(outlier_counts.values())
        outlier_pct = (total_outliers / (len(data) * len(numeric_cols))) * 100 if numeric_cols.size > 0 else 0
        
        passed = outlier_pct < 10.0  # Allow up to 10% outliers
        
        return DataQualityCheck(
            check_name="outliers",
            passed=passed,
            level=DataQualityLevel.WARNING if 10.0 <= outlier_pct < 20.0 else (DataQualityLevel.FAIL if outlier_pct >= 20.0 else DataQualityLevel.PASS),
            message=f"Outliers detected: {outlier_pct:.2f}%",
            details={"outlier_counts": outlier_counts, "outlier_pct": outlier_pct}
        )
    
    def _check_duplicates(self, data: pd.DataFrame) -> DataQualityCheck:
        """Check for duplicate rows"""
        duplicates = data.duplicated().sum()
        duplicate_pct = (duplicates / len(data)) * 100
        
        passed = duplicate_pct < 1.0  # Allow up to 1% duplicates
        
        return DataQualityCheck(
            check_name="duplicates",
            passed=passed,
            level=DataQualityLevel.WARNING if 1.0 <= duplicate_pct < 5.0 else (DataQualityLevel.FAIL if duplicate_pct >= 5.0 else DataQualityLevel.PASS),
            message=f"Duplicate rows: {duplicates} ({duplicate_pct:.2f}%)",
            details={"duplicate_count": int(duplicates), "duplicate_pct": duplicate_pct}
        )


# Global instances
_data_lake: Optional[DataLake] = None
_data_validator: Optional[DataQualityValidator] = None


def get_data_lake() -> DataLake:
    """Get or create global data lake"""
    global _data_lake
    if _data_lake is None:
        _data_lake = DataLake()
    return _data_lake


def get_data_validator() -> DataQualityValidator:
    """Get or create global data validator"""
    global _data_validator
    if _data_validator is None:
        _data_validator = DataQualityValidator()
    return _data_validator

