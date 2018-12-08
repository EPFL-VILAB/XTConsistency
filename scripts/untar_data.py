
import subprocess, glob, os, parse
from multiprocessing import Pool
from fire import Fire
import random
import yaml
import IPython

def process_file(file, result_loc="local/small_data", flags="-C"):
	try:
		*rest, task, archive = file.split('/')
		result_dir = f"{result_loc}/{archive[:-4]}"
		os.makedirs(result_dir, exist_ok=True)
		print (["wget", file, "-P", result_loc])
		print (["tar", "xf", f"{result_loc}/{archive}", flags, result_dir, "--no-same-owner"])
		return_code = subprocess.call(["wget", file, "-P", result_loc])
		return_code += subprocess.call(["tar", "xf", f"{result_loc}/{archive}", flags, result_dir, "--no-same-owner"])
		return_code += subprocess.call(["rm", "-rf", f"{result_loc}/{archive}"])

		return return_code, result_dir
	except Exception as e:
		print (e, file)
		return 1, result_dir

def main(filename="data/alllinks.txt", tasks=['rgb', 'normal', 
		'principal_curvature', 'depth_zbuffer', 'reshading', 
		'sobel_edges', 'keypoints3d', 'keypoints2d', 'edge_occlusion', 'edge_texture']):

	links = [link.strip() for link in open(filename, 'r')]
	links = [(link, link.split('/')) for link in links]
	links = [file for (file, (*rest, task, archive)) in links if task in tasks]
	
	with Pool() as pool:
		for i, res in enumerate(pool.imap_unordered(process_file, links)):
			return_code, result_dir = res
			if return_code != 0:
				print(f"{result_dir} non zero exit code ({return_code}), trying to unzip and compress, trying again")
				pool.apply_async(process_file, (links[i],))

			print (f"Downloaded {result_dir}: {i}/{len(links)} files")

if __name__ == "__main__":
	Fire(main)
