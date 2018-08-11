
import subprocess, glob, os
from multiprocessing import Pool

tasks = ['rgb', 'normal','principal_curvature', 'depth_zbuffer']
links = [link[:-1] for link in open('data/alllinks.txt', 'r')]

def process_file(file):
	*rest, task, archive = file.split('/')
	if task not in tasks: return "skipped"
	result_dir = "/result/" + archive[:-4]
	os.makedirs(result_dir, exist_ok=True)

	curl = subprocess.Popen(["curl", "-s", file], stdout=subprocess.PIPE)
	tar = subprocess.Popen(["tar", "xf", "-", "-C", result_dir, "--no-same-owner"], stdin=curl.stdout, stdout=subprocess.PIPE)
	tar.wait()

	return result_dir

with Pool() as pool:
	for i, result_dir in enumerate(pool.imap_unordered(process_file, links)):
		if result_dir != "skipped":
			print (f"Downloaded {result_dir}: {i}/{len(links)} files")
