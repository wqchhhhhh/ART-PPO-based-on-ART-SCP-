import solver_environment
from pathlib import Path
import os
from cruise.cli import main
os.chdir(Path(__file__).resolve().parent / '10302026')
main(['solve', '--method', 'expert', '--output', 'results/local_expert.npz'])

