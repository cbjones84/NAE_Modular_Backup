#!/usr/bin/env python3
"""
Log Archival Script for NAE
===========================
Compresses old `ralph_approved_strategies_*.json` files into monthly archives
to improve directory performance and manage disk space.

Logic:
1. Scan logs/ for strategy files.
2. Filter files older than 30 days.
3. Group by YYYY-MM.
4. Create zip archives in logs/archives/.
5. Verify and delete originals.
"""

import os
import sys
import glob
import re
import time
import zipfile
import shutil
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

# Configuration
if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
    LOG_DIR = sys.argv[1]
else:
    LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
    if not os.path.exists(LOG_DIR):
        LOG_DIR = "logs"

ARCHIVE_DIR = os.path.join(LOG_DIR, "archives")
RETENTION_DAYS = 30
FILE_PATTERN = "ralph_approved_strategies_*.json"
MIN_SHARPE_TO_KEEP = 1.5  # Keep strategies with Sharpe >= 1.5

# Regex to extract timestamp: ralph_approved_strategies_1765339752.json
FILENAME_REGEX = re.compile(r"ralph_approved_strategies_(\d+)\.json")

def get_file_timestamp(filename: str) -> float:
    """Extract timestamp from filename or fallback to mtime"""
    match = FILENAME_REGEX.search(filename)
    if match:
        return float(match.group(1))
    return os.path.getmtime(filename)

def is_high_value_file(filepath: str) -> bool:
    """
    Check if file contains any 'High Value' strategies.
    Returns True if at least one strategy has Sharpe >= MIN_SHARPE_TO_KEEP.
    """
    try:
        import json
        with open(filepath, 'r') as f:
            data = json.load(f)
            
        if not isinstance(data, list):
            return False
            
        for strategy in data:
            sharpe = strategy.get('sharpe', 0.0)
            if sharpe >= MIN_SHARPE_TO_KEEP:
                return True
        return False
    except Exception:
        # If we can't read it, assume it's corrupt/not valuable and allow archival
        return False

def ensure_archive_dir():
    """Create archive directory if it doesn't exist"""
    if not os.path.exists(ARCHIVE_DIR):
        print(f"Creating archive directory: {ARCHIVE_DIR}")
        os.makedirs(ARCHIVE_DIR)

def archive_files(dry_run: bool = False):
    """Main archival logic"""
    print(f"Scanning {LOG_DIR} for {FILE_PATTERN}...")
    print(f"Policy: Keep recent (<{RETENTION_DAYS} days) OR High Performance (Sharpe >= {MIN_SHARPE_TO_KEEP})")
    
    # 1. Gather all strategy files
    pattern_path = os.path.join(LOG_DIR, FILE_PATTERN)
    all_files = glob.glob(pattern_path)
    
    if not all_files:
        print("No strategy files found.")
        return

    print(f"Found {len(all_files)} total strategy files.")
    
    # 2. Filter for old files
    cutoff_time = time.time() - (RETENTION_DAYS * 86400)
    files_to_archive: Dict[str, List[str]] = {}
    
    recent_count = 0
    protected_count = 0
    archivable_count = 0
    
    for file_path in all_files:
        basename = os.path.basename(file_path)
        ts = get_file_timestamp(basename)
        
        # 2a. Is it recent?
        if ts >= cutoff_time:
            recent_count += 1
            continue
            
        # 2b. Is it High Value?
        if is_high_value_file(file_path):
            protected_count += 1
            # print(f"  [KEEP] High Value: {basename}") # Too noisy for 100k files
            continue
            
        # Group by Month: YYYY-MM
        dt = datetime.fromtimestamp(ts)
        month_key = dt.strftime("%Y-%m")
        
        if month_key not in files_to_archive:
            files_to_archive[month_key] = []
        files_to_archive[month_key].append(file_path)
        archivable_count += 1
        
    print(f"Summary:")
    print(f"  - Kept (Recent): {recent_count}")
    print(f"  - Kept (High Value): {protected_count}")
    print(f"  - To Archive: {archivable_count} files across {len(files_to_archive)} months.")
    
    if dry_run:
        print("\n[DRY RUN] No files will be moved or deleted.")
        for month, files in files_to_archive.items():
            print(f"  - {month}: {len(files)} files -> archives/strategies_{month}.zip")
        return

    ensure_archive_dir()
    
    # 3. Create Archives
    for month, files in files_to_archive.items():
        zip_name = f"strategies_{month}.zip"
        zip_path = os.path.join(ARCHIVE_DIR, zip_name)
        
        print(f"\nProcessing {month} ({len(files)} files)...")
        
        # Open Zip in Append mode if exists, else Write
        mode = 'a' if os.path.exists(zip_path) else 'w'
        
        try:
            with zipfile.ZipFile(zip_path, mode, zipfile.ZIP_DEFLATED) as zf:
                # Get list of files already in zip to avoid duplicates
                existing_in_zip = set(zf.namelist())
                
                added_count = 0
                for file_path in files:
                    basename = os.path.basename(file_path)
                    if basename in existing_in_zip:
                        continue
                        
                    zf.write(file_path, basename)
                    added_count += 1
            
            print(f"  Added {added_count} files to {zip_name}")
            
            # 4. Verification and Deletion
            # Re-read zip to verify correctness
            with zipfile.ZipFile(zip_path, 'r') as verify_zf:
                archived_files = set(verify_zf.namelist())
                
                deleted_count = 0
                for file_path in files:
                    basename = os.path.basename(file_path)
                    if basename in archived_files:
                        os.remove(file_path)
                        deleted_count += 1
                    else:
                        print(f"  [ERROR] Failed to verify {basename} in archive. Keeping original.")
                
                print(f"  Cleaned up {deleted_count} files.")
                
        except Exception as e:
            print(f"  [CRITICAL ERROR] Failed to archive {month}: {e}")

    print("\nArchival complete.")

if __name__ == "__main__":
    is_dry_run = "--dry-run" in sys.argv
    archive_files(dry_run=is_dry_run)
