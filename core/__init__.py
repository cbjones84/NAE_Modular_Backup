"""
NAE Core Module
===============
Central command and control systems for NAE.

Components:
- nae_mission_control: $5M growth mission coordination
- agent_alignment_config.json: Agent directives and configuration
"""

from pathlib import Path

CORE_DIR = Path(__file__).parent
NAE_DIR = CORE_DIR.parent

__all__ = ['CORE_DIR', 'NAE_DIR']

