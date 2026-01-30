#!/usr/bin/env python3
"""
Demo script showing Casey's enhanced capabilities (v5)
Demonstrates all the new capabilities that make Casey as powerful as an AI assistant
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from agents.casey import CaseyAgent

def demo_file_operations():
    """Demonstrate file operations"""
    print("\n" + "="*60)
    print("DEMO: File Operations")
    print("="*60)
    
    casey = CaseyAgent()
    
    # Read a file
    print("\n1. Reading file...")
    result = casey.read_file("agents/ralph.py", offset=1, limit=10)
    print(f"   Read {result.get('lines', 0)} lines")
    print(f"   First 200 chars: {result.get('content', '')[:200]}...")
    
    # List directory
    print("\n2. Listing directory...")
    result = casey.list_directory("agents")
    print(f"   Found {len(result.get('files', []))} files and {len(result.get('directories', []))} directories")
    
    # Write a test file
    print("\n3. Writing test file...")
    test_content = "# Test file created by Casey\nprint('Hello from Casey!')\n"
    result = casey.write_file("test_casey_demo.py", test_content)
    print(f"   Success: {result.get('success')}")
    
    # Read it back
    result = casey.read_file("test_casey_demo.py")
    print(f"   Content: {result.get('content', '').strip()}")
    
    # Delete test file
    print("\n4. Deleting test file...")
    result = casey.delete_file("test_casey_demo.py")
    print(f"   Success: {result.get('success')}")

def demo_codebase_search():
    """Demonstrate codebase search capabilities"""
    print("\n" + "="*60)
    print("DEMO: Codebase Search")
    print("="*60)
    
    casey = CaseyAgent()
    
    # Grep search
    print("\n1. Grep search for 'class.*Agent'...")
    result = casey.grep_search(r"class\s+\w+Agent", max_results=5)
    print(f"   Found {len(result.get('matches', []))} matches")
    for match in result.get('matches', [])[:3]:
        print(f"   {match['file']}:{match['line']}: {match['content'][:60]}...")
    
    # Glob search
    print("\n2. Glob search for '*.py' files in agents...")
    result = casey.glob_search("agents/*.py")
    print(f"   Found {len(result.get('files', []))} Python files")
    
    # Semantic search
    print("\n3. Semantic search for 'trading strategy'...")
    result = casey.semantic_search("trading strategy", max_results=5)
    print(f"   Found {len(result.get('results', []))} relevant files")
    for res in result.get('results', [])[:3]:
        print(f"   {res['file']} (score: {res['score']})")

def demo_code_execution():
    """Demonstrate code execution capabilities"""
    print("\n" + "="*60)
    print("DEMO: Code Execution")
    print("="*60)
    
    casey = CaseyAgent()
    
    # Execute Python code
    print("\n1. Executing Python code...")
    code = """
x = 10
y = 20
result = x + y
print(f"Sum: {result}")
"""
    result = casey.execute_python_code(code)
    print(f"   Status: {result.get('status')}")
    print(f"   Output: {result.get('output', '').strip()}")
    
    # Execute terminal command
    print("\n2. Executing terminal command...")
    result = casey.execute_terminal_command("echo 'Hello from Casey!'")
    print(f"   Status: {result.get('status')}")
    print(f"   Output: {result.get('output', '').strip()}")

def demo_context_understanding():
    """Demonstrate context understanding"""
    print("\n" + "="*60)
    print("DEMO: Context Understanding")
    print("="*60)
    
    casey = CaseyAgent()
    
    # Understand context across files
    print("\n1. Understanding context across multiple files...")
    files = ["agents/casey.py", "agents/ralph.py"]
    result = casey.understand_context(files, "How do agents communicate?")
    print(f"   Analyzed {len(result.get('analysis', {}).get('files', []))} files")
    print(f"   Found {len(result.get('relationships', []))} relationships")
    
    # Debug code
    print("\n2. Debugging code...")
    result = casey.debug_code("agents/casey.py")
    print(f"   Syntax valid: {result.get('diagnosis', {}).get('syntax_valid')}")
    print(f"   Suggestions: {len(result.get('suggestions', []))}")

def demo_capabilities_summary():
    """Show capabilities summary"""
    print("\n" + "="*60)
    print("DEMO: Capabilities Summary")
    print("="*60)
    
    casey = CaseyAgent()
    capabilities = casey.get_capabilities_summary()
    
    for category, items in capabilities.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        for item in items:
            print(f"  • {item}")

def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("Casey Enhanced Capabilities Demo (v5)")
    print("="*60)
    print("\nThis demo shows Casey's new capabilities that match")
    print("an AI assistant's tool capabilities.")
    
    try:
        demo_file_operations()
        demo_codebase_search()
        demo_code_execution()
        demo_context_understanding()
        demo_capabilities_summary()
        
        print("\n" + "="*60)
        print("✅ All demos completed successfully!")
        print("="*60)
        print("\nCasey now has ALL the capabilities of an AI assistant:")
        print("  ✓ File operations (read, write, edit, delete, list)")
        print("  ✓ Codebase search (grep, glob, semantic)")
        print("  ✓ Code execution (Python, terminal, agent methods)")
        print("  ✓ Context understanding (multi-file analysis)")
        print("  ✓ Debugging & testing")
        print("\nCasey can now do EXACTLY what an AI assistant can do!")
        
    except Exception as e:
        print(f"\n❌ Error in demo: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

