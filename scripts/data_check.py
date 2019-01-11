import glob, yaml
from utils import *
from task_configs import tasks
from tqdm import tqdm
import IPython

buildings = yaml.load(open("data/split.txt"))
for task in tasks:
	count = 0
	num_missing = 0
	for building in (buildings['train_buildings'] + buildings['val_buildings']):
		exp =  f'{task}/{building}/**' if USE_RAID else f'{building}_{task}/**'
		num_files = len(get_files(exp, recursive=True))
		if num_files < 2: num_missing += 1
		count += num_files
	print(f'{task}: {count} files and missing {num_missing} buildings')
