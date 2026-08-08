from pathlib import Path
import subprocess, sys
HERE=Path(__file__).resolve().parent
PYTHON=sys.executable

def run_script(name, *args):
    subprocess.run([PYTHON, str(HERE/name), *map(str,args)], check=True, cwd=HERE)

run_script('canonical_analysis.py', '--core')
run_script('all_pairwise_tests.py')
run_script('canonical_analysis.py', '--stress-all')
print('Reproduction complete:', HERE/'generated_outputs')
