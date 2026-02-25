#!/usr/bin/env python
"""Script para ejecutar el notebook de XGBoost Baseline"""

import json
import sys
import subprocess
from pathlib import Path

# Rutas
notebook_path = Path('03_Baseline_XGBoost.ipynb')
venv_python = Path(r'c:\Users\DELL\Documents\GitHub\.venv\Scripts\python.exe')

# Comando para ejecutar con papermill
cmd = [
    str(venv_python),
    '-m', 'papermill',
    str(notebook_path),
    str(notebook_path)
]

print("Ejecutando notebook de XGBoost Baseline...")
print(f"Comando: {' '.join(cmd)}")
print()

result = subprocess.run(cmd, capture_output=True, text=True)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

sys.exit(result.returncode)
