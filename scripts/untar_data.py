
import subprocess, glob, os
from multiprocessing import Pool
from fire import Fire

def process_file(file, result_loc="result"):
	try:
		*rest, task, archive = file.split('/')
		result_dir = f"{result_loc}/{archive[:-4]}"
		os.makedirs(result_dir, exist_ok=True)

		curl = subprocess.Popen(["curl", "-s", file], stdout=subprocess.PIPE)
		tar = subprocess.Popen(["tar", "xf", "-", "-C", result_dir, "--no-same-owner"], stdin=curl.stdout, stdout=subprocess.PIPE)
		tar.wait()
		return result_dir
	except Exception as e:
		print (e)
		return "error"

def main(filename="data/alllinks.txt", 
		tasks=['rgb', 'normal','principal_curvature', 'depth_zbuffer']):
	
	links = [link.strip() for link in open(filename, 'r')]
	links = [(link, link.split('/')) for link in links]
	links = [file for (file, (*rest, task, archive)) in links if task in tasks]

	with Pool() as pool:
		for i, result_dir in enumerate(pool.imap_unordered(process_file, links)):
			print (f"Downloaded {result_dir}: {i}/{len(links)} files")

if __name__ == "__main__":
	Fire(main)
