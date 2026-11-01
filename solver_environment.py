"""Make CasADi solver DLL dependencies discoverable on Windows."""
import os
from pathlib import Path
import casadi
os.environ['PATH'] = str(Path(casadi.__file__).parent) + os.pathsep + os.environ.get('PATH', '')
