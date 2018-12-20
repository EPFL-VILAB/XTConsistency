import glob, yaml
from utils import *
from task_configs import tasks
from tqdm import tqdm

data_dir = "mount/data/taskonomy3"
buildings = yaml.load(open("data/split.txt"))
for task in tasks:
	count = 0
	num_missing = 0
	for building in buildings['train_buildings']:
		num_files = len(glob.glob(f'{data_dir}/{building}_{task}/**', recursive=True))
		if num_files < 2: num_missing += 1
		count += num_files
	print(f'{task}: {count} files and missing {num_missing} buildings')
