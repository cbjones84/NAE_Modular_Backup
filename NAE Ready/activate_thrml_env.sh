#!/bin/bash
# Activation script for Python 3.11 virtual environment with THRML

cd "$(dirname "$0")"
source venv_python311/bin/activate

echo "✅ Python 3.11 virtual environment activated"
echo "✅ Python version: $(python --version)"
echo "✅ JAX version: $(python -c 'import jax; print(jax.__version__)' 2>/dev/null || echo 'Not available')"
echo "✅ THRML version: $(python -c 'import thrml; print(thrml.__version__)' 2>/dev/null || echo 'Not available')"
echo ""
echo "To deactivate, run: deactivate"

