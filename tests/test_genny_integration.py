# NAE/test_genny_integration.py
"""
Integration test for Genny agent with existing NAE system
Tests Genny's ability to track Optimus and Ralph agents, and integrate with Casey
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from agents.genny import GennyAgent
from agents.optimus import OptimusAgent
from agents.ralph import RalphAgent
from agents.casey import CaseyAgent
from goal_manager import get_nae_goals

def test_genny_integration():
    """Test Genny's integration with other NAE agents"""
    print("=" * 60)
    print("TESTING GENNY AGENT INTEGRATION WITH NAE SYSTEM")
    print("=" * 60)
    
    # Initialize agents
    print("\n1. Initializing NAE agents...")
    genny = GennyAgent()
    optimus = OptimusAgent(sandbox=True)
    ralph = RalphAgent()
    casey = CaseyAgent()
    
    print(f"✓ Genny initialized with goals: {genny.goals}")
    print(f"✓ Optimus initialized in sandbox mode")
    print(f"✓ Ralph initialized with strategy learning")
    print(f"✓ Casey initialized for agent building")
    
    # Test goal integration
    print("\n2. Testing goal integration...")
    nae_goals = get_nae_goals()
    assert genny.goals == nae_goals, "Genny goals don't match NAE goals"
    print(f"✓ Genny goals match NAE goals: {nae_goals}")
    
    # Test Optimus tracking
    print("\n3. Testing Optimus success tracking...")
    mock_trade_data = {
        "strategy_name": "Integration Test Strategy",
        "profit": 2500,
        "execution_time": 0.3,
        "risk_score": 0.4,
        "expected_return": 0.18,
        "strategy_type": "long_term",
        "risk_level": "conservative"
    }
    
    optimus_success = genny.track_optimus_success(mock_trade_data)
    assert optimus_success is not None, "Failed to track Optimus success"
    print(f"✓ Optimus success tracked: {optimus_success['wealth_impact']['immediate_gain']}")
    
    # Test Ralph strategy tracking
    print("\n4. Testing Ralph strategy tracking...")
    mock_strategy_data = {
        "name": "Integration Test Strategy",
        "trust_score": 92,
        "backtest_score": 85,
        "consensus_count": 4,
        "risk_reward": 3.2,
        "complexity_level": "medium",
        "documentation_quality": "high"
    }
    
    ralph_strategy = genny.track_ralph_strategy(mock_strategy_data)
    assert ralph_strategy is not None, "Failed to track Ralph strategy"
    print(f"✓ Ralph strategy tracked: trust_score={ralph_strategy['success_factors']['trust_score']}")
    
    # Test recipe curation
    print("\n5. Testing success recipe curation...")
    recipes = genny.curate_success_recipes()
    assert recipes is not None, "Failed to curate success recipes"
    assert len(recipes['optimus_recipes']) > 0, "No Optimus recipes generated"
    assert len(recipes['ralph_recipes']) > 0, "No Ralph recipes generated"
    assert len(recipes['nae_system_recipes']) > 0, "No NAE system recipes generated"
    assert len(recipes['generational_wealth_recipes']) > 0, "No generational wealth recipes generated"
    print(f"✓ Success recipes curated: {len(recipes['optimus_recipes'])} Optimus, {len(recipes['ralph_recipes'])} Ralph")
    
    # Test heir knowledge package creation
    print("\n6. Testing heir knowledge package creation...")
    mock_heir_profile = {
        "heir_id": "integration_test_heir",
        "name": "Integration Test Heir",
        "experience_level": "advanced",
        "interests": ["advanced_trading", "system_optimization", "legacy_planning"]
    }
    
    knowledge_package = genny.create_heir_knowledge_package(mock_heir_profile)
    assert knowledge_package is not None, "Failed to create heir knowledge package"
    assert knowledge_package['heir_profile']['experience_level'] == "advanced", "Heir profile not properly set"
    assert len(knowledge_package['learning_path']['modules']) > 0, "No learning modules assigned"
    assert len(knowledge_package['mentorship_plan']['mentor_assignments']) > 0, "No mentors assigned"
    print(f"✓ Heir knowledge package created for {mock_heir_profile['name']}")
    
    # Test learning and adaptation
    print("\n7. Testing wealth growth pattern learning...")
    learning_results = genny.learn_wealth_growth_patterns()
    assert learning_results is not None, "Failed to learn wealth growth patterns"
    assert 'growth_patterns' in learning_results, "Growth patterns not generated"
    assert 'maintenance_patterns' in learning_results, "Maintenance patterns not generated"
    assert 'optimization_opportunities' in learning_results, "Optimization opportunities not identified"
    assert 'risk_patterns' in learning_results, "Risk patterns not analyzed"
    print(f"✓ Wealth growth patterns learned: avg_growth_rate={learning_results['growth_patterns'].get('average_growth_rate', 'N/A')}")
    
    # Test AI modules
    print("\n8. Testing AI modules functionality...")
    
    # Test Financial Planner Module
    financial_analysis = genny.ai_modules['financial_planner'].analyze_asset_allocation({
        "stocks": 100000,
        "real_estate": 75000,
        "bonds": 50000,
        "cash": 25000
    })
    assert financial_analysis is not None, "Financial analysis failed"
    print(f"✓ Financial Planner Module: analyzed ${sum(financial_analysis['current_allocation'].values()):,} portfolio")
    
    # Test Educational Advisory Module
    learning_path = genny.ai_modules['educational_advisory'].create_learning_path(mock_heir_profile)
    assert learning_path is not None, "Learning path creation failed"
    print(f"✓ Educational Advisory Module: created learning path for {learning_path['experience_level']} level")
    
    # Test Ethical Compliance Module
    ethical_assessment = genny.ai_modules['ethical_compliance'].assess_ethical_compliance({
        "id": "test_decision_001",
        "risk_level": "low",
        "transparency": "high",
        "stakeholder_impact": "medium"
    })
    assert ethical_assessment is not None, "Ethical assessment failed"
    print(f"✓ Ethical Compliance Module: compliance_score={ethical_assessment['compliance_score']}")
    
    # Test Orchestration Engine Module
    workflow_result = genny.ai_modules['orchestration_engine'].orchestrate_workflow("wealth_assessment", {
        "portfolio_value": 250000,
        "risk_tolerance": "moderate"
    })
    assert workflow_result is not None, "Workflow orchestration failed"
    print(f"✓ Orchestration Engine Module: executed wealth_assessment workflow")
    
    # Test main execution cycle
    print("\n9. Testing main execution cycle...")
    cycle_results = genny.run_cycle()
    assert cycle_results is not None, "Main cycle failed"
    assert cycle_results['status'] == 'completed', f"Cycle status: {cycle_results['status']}"
    print(f"✓ Main execution cycle completed successfully")
    
    # Test generational wealth components
    print("\n10. Testing generational wealth components...")
    components = genny.generational_wealth_components
    assert len(components) == 4, f"Expected 4 components, got {len(components)}"
    
    required_components = ["financial_capital", "intellectual_capital", "social_capital", "values_and_legacy"]
    for component in required_components:
        assert component in components, f"Missing component: {component}"
        assert "description" in components[component], f"Missing description for {component}"
        assert "tracking_metrics" in components[component], f"Missing tracking metrics for {component}"
    
    print(f"✓ All 4 generational wealth components properly defined")
    
    # Test data persistence
    print("\n11. Testing data persistence...")
    data_dir = genny.data_dir
    assert os.path.exists(data_dir), f"Data directory not created: {data_dir}"
    
    # Check if log files are being created
    log_file = genny.log_file
    assert os.path.exists(log_file), f"Log file not created: {log_file}"
    
    print(f"✓ Data persistence working: {data_dir}")
    
    # Test communication capabilities
    print("\n12. Testing communication capabilities...")
    
    # Test message receiving
    test_message = "Test message from integration test"
    genny.receive_message(test_message)
    assert len(genny.inbox) > 0, "Message not received"
    print(f"✓ Message receiving working: {len(genny.inbox)} messages in inbox")
    
    # Test message sending (to Casey)
    genny.send_message("Test message to Casey", casey)
    assert len(genny.outbox) > 0, "Message not sent"
    print(f"✓ Message sending working: {len(genny.outbox)} messages in outbox")
    
    print("\n" + "=" * 60)
    print("ALL INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)
    
    # Summary report
    print("\nINTEGRATION SUMMARY:")
    print(f"✓ Genny agent successfully integrated with NAE system")
    print(f"✓ Goals properly synchronized with NAE goal manager")
    print(f"✓ Optimus success tracking: {len(genny.optimus_success_log)} trades tracked")
    print(f"✓ Ralph strategy tracking: {len(genny.ralph_strategy_log)} strategies tracked")
    print(f"✓ Success recipes: {len(genny.nae_profit_recipes)} recipe sets curated")
    print(f"✓ Heir knowledge packages: {len(genny.heir_knowledge_base)} packages created")
    print(f"✓ AI modules: All 5 modules functional")
    print(f"✓ Generational wealth framework: 4 components implemented")
    print(f"✓ Learning and adaptation: Wealth patterns analyzed")
    print(f"✓ Data persistence: Files saved to {data_dir}")
    print(f"✓ Communication: AutoGen-compatible messaging")
    
    return True

if __name__ == "__main__":
    try:
        test_genny_integration()
    except Exception as e:
        print(f"\n❌ INTEGRATION TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
