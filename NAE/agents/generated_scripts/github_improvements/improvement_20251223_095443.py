"""
Auto-implemented improvement from GitHub
Source: sarahvandistel/KellyCondor/check_trading_status.py
Implemented: 2025-12-23T09:54:43.261946
Usefulness Score: 80
Keywords: def , position
"""

# Original source: sarahvandistel/KellyCondor
# Path: check_trading_status.py


# Function: main
def main():
    """Main status checking function"""
    print("🔍 KellyCondor Trading Status Checker")
    print("=" * 80)
    
    # Check trading process
    print("\n📊 Trading Process Status:")
    print("-" * 40)
    
    trading_proc = check_process_running("kelly-live")
    if trading_proc:
        print(f"✅ KellyCondor Live Trading: RUNNING")
        print(f"   PID: {trading_proc['pid']}")
        print(f"   Command: {trading_proc['cmdline']}")
        
        # Get detailed process info
        proc_info = get_process_info(trading_proc['pid'])
        if proc_info:
            print(f"   CPU Usage: {proc_info['cpu_percent']:.1f}%")
            print(f"   Memory Usage: {proc_info['memory_percent']:.1f}%")
            print(f"   Started: {proc_info['create_time']}")
    else:
        print("❌ KellyCondor Live Trading: NOT RUNNING")
    
    # Check dashboard process
    print("\n📈 Dashboard Status:")
    print("-" * 40)
    
    dashboard_proc = check_process_running("app.py")
    if dashboard_proc:
        print(f"✅ Dashboard: RUNNING")
        print(f"   PID: {dashboard_proc['pid']}")
        print(f"   Command: {dashboard_proc['cmdline']}")
        
        # Check if port 8050 is listening
        if check_port_listening(8050):
            print("   🌐 Dashboard accessible at: http://localhost:8050")
        else:
            print("   ⚠️  Dashboard port 8050 not listening")
            
        proc_info = get_process_info(dashboard_proc['pid'])
        if proc_info:
            print(f"   CPU Usage: {proc_info['cpu_percent']:.1f}%")
            print(f"   Memory Usage: {proc_info['memory_percent']:.1f}%")
    else:
        print("❌ Dashboard: NOT RUNNING")
    
    # Check Redis
    print("\n🗄️  Redis Status:")
    print("-" * 40)
    
    try:
        import redis
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✅ Redis: RUNNING")
        
        # Check for trading data
        trade_keys = r.keys("trade:*")
        position_keys = r.keys("position:*")
        
        print(f"   📊 Trade records: {len(trade_keys)}")
        print(f"   📦 Position records: {len(position_keys)}")
        
    except Exception as e:
        print(f"❌ Redis: NOT RUNNING ({e})")
    
    # Check TWS connection
    print("\n🏦 TWS Connection Status:")
    print("-" * 40)
    
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', 7497))
        sock.close()
        
        if result == 0:
            print("✅ TWS Connection: AVAILABLE (port 7497)")
        else:
            print("❌ TWS Connection: NOT AVAILABLE (port 7497)")
            print("   Make sure TWS is running in paper trading mode")
    except Exception as e:
        print(f"❌ TWS Connection: ERROR ({e})")
    
    print(f"\n⏰ Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n💡 To start trading: kelly-live --paper --verbose")
    print("💡 To start dashboard: python dashboard/app.py")
    print("💡 To monitor trades: python monitor_trades.py")


