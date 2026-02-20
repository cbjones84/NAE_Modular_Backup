
import sys
import os
import traceback

# Add current directory to path
sys.path.insert(0, os.getcwd())

print("="*50)
print("DIAGNOSTIC START")
print("="*50)

# 1. Test Master Scheduler Import
print("\n[TEST 1] Importing nae_master_scheduler...")
try:
    import nae_master_scheduler
    print("✅ SUCCESS: nae_master_scheduler imported")
    print(f"File: {nae_master_scheduler.__file__}")
except Exception:
    print("❌ FAILURE: Could not import nae_master_scheduler")
    traceback.print_exc()

# 2. Test FeedbackManager
print("\n[TEST 2] Checking FeedbackManager...")
try:
    try:
        from tools.feedback_loops import FeedbackManager
        print("✅ Found FeedbackManager in tools.feedback_loops")
    except ImportError:
        print("⚠️  tools.feedback_loops not found, trying tools.feedback_loop...")
        from tools.feedback_loop import FeedbackManager
        print("✅ Found FeedbackManager in tools.feedback_loop")
    
    fm = FeedbackManager()
    print("✅ FeedbackManager instantiated")
    
    if hasattr(fm, 'get'):
        print("✅ FeedbackManager has 'get' method")
    elif hasattr(fm, 'loops'):
        print("✅ FeedbackManager has 'loops' attribute")
    else:
        print("❌ FAILURE: FeedbackManager MISSING 'get' or 'loops'")
        print(f"Attributes found: {dir(fm)}")
        
except Exception:
    print("❌ FAILURE: Error checking FeedbackManager")
    traceback.print_exc()

print("\n" + "="*50)
print("DIAGNOSTIC COMPLETE")
print("="*50)
