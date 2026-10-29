import os
import sys

# VLS modules import each other as top-level packages (core, utils, ...), as main.py does
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
