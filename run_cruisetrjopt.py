import solver_environment
from pathlib import Path
import os, runpy, sys
root = Path(__file__).resolve().parent
project = root / 'CruiseTrjOpt' / 'CruiseTrjOpt'
os.chdir(project)
(project / 'results').mkdir(exist_ok=True)
sys.path.insert(0, str(project))
runpy.run_path(str(project / 'main.py'), run_name='__main__')


