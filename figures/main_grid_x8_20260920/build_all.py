from common import export
import runpy
from pathlib import Path
export()
for script in ['gen_tradeoff.py','gen_stability.py','gen_accuracy_heatmap.py','gen_interventions.py','gen_length_ratio.py']:runpy.run_path(str(Path(__file__).parent/script),run_name='__main__')
print('Validated 216 dataset/config rows; generated six figures in PDF, SVG and PNG.')
