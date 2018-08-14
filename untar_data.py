
import subprocess, glob, os
from multiprocessing import Pool
from utils import *
from logger import VisdomLogger

tasks = ['rgb', 'normal','principal_curvature', 'depth_zbuffer']
links = [link[:-1] for link in open('data/alllinks.txt', 'r')]

logger = VisdomLogger("train", server='35.230.67.129', port=7000, env=JOB)

def process_file(file):
	try:

		*rest, task, archive = file.split('/')
		if task not in tasks: return "skipped"
		result_dir = "/result/" + archive[:-4]
		os.makedirs(result_dir, exist_ok=True)

		curl = subprocess.Popen(["curl", "-s", file], stdout=subprocess.PIPE)
		tar = subprocess.Popen(["tar", "xf", "-", "-C", result_dir, "--no-same-owner"], stdin=curl.stdout, stdout=subprocess.PIPE)
		tar.wait()

		return result_dir
	
	except:
		return "error"


with Pool() as pool:
	for i, result_dir in enumerate(pool.imap_unordered(process_file, links)):
		if result_dir != "skipped":
			logger.text (f"Downloaded {result_dir}: {i}/{len(links)} files")
