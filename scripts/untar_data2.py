
import subprocess, glob, os
from multiprocessing import Pool
import random
import yaml
import IPython

def process_file(file, flags="-C"):
	# res = parse.parse("http://downloads.cs.stanford.edu/downloads/taskonomy_data/{task}/{building}_{task}.tar", file)
	# building, task = res['building'], res['task']
	# data_full_len = len(glob.glob(f'mount/data_full/data/taskonomy3/{building}_{task}/**', recursive=True))
	# data_len = len(glob.glob(f'mount/data_full/data/taskonomy3/{building}_depth_zbuffer/**', recursive=True))
	# if data_full_len >= data_len:
	# 	print(f'skipping {building} for {task}...')
	# 	return 0, f'{building}_{task}'
	try:
		*rest, archive = file.split('/')
		result_dir = "/cvgl/group/taskonomy/processed/" + archive[:-4]
		os.makedirs(result_dir, exist_ok=True)
		return_code = subprocess.call(["tar", "xf", file, flags, result_dir, "--no-same-owner"])
		return return_code, result_dir

	except Exception as e:
		print (e, file)
		return 1, result_dir

if __name__ == "__main__":

	tasks = ['normal', 'rgb', 'principal_curvature', 'depth_zbuffer']
	tars = sorted(glob.glob("/cvgl/group/taskonomy/public_data/tars/*.tar"))
	buildings = ["almena", "albertville"]

	tars = [(tar, tar.split('/')) for tar in tars]
	tars = [file for (file, (*rest, archive)) in tars \
		if (
			any(task in archive[(-len(task)-4):-4] for task in tasks)
			# and any(building in archive for building in buildings)
		)
	]

	print (tars)
	
	with Pool() as pool:
		for i, res in enumerate(pool.imap_unordered(process_file, tars)):
			return_code, result_dir = res
			if return_code != 0:
				print("non zero exit code trying to unzip and compress, trying again")
				pool.apply_async(process_file, (tars[i],))

			print (i, len(tars))

