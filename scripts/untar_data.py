
import subprocess, glob, os, parse
from multiprocessing import Pool
from fire import Fire
import IPython

def process_file(file, result_loc="/result", flags="-C"):
	try:
		*rest, task, archive = file.split('/')
		result_dir = f"{result_loc}/{archive[:-4]}"
		os.makedirs(result_dir, exist_ok=True)
		curl = subprocess.Popen(["curl", "-s", file], stdout=subprocess.PIPE)
		tar = subprocess.Popen(["tar", "xf", "-", flags, result_dir, "--no-same-owner"], stdin=curl.stdout, stdout=subprocess.PIPE)
		tar.wait()
		return tar.returncode, result_dir
	except Exception as e:
		print (e, file)
		return 1, result_dir

# def unzip_untar(file, result_loc="/result"):
# 	try:
# 		*rest, task, archive = file.split('/')
# 		result_dir = f"{result_loc}/{archive[:-4]}"
# 		os.makedirs(result_dir, exist_ok=True)
# 		curl = subprocess.Popen(["curl", "-s", file], stdout=subprocess.PIPE)
# 		unzip = subprocess.Popen(["gunzip", f"{result_dir}*", "--no-same-owner"], stdin=curl.stdout, stdout=subprocess.PIPE)
# 		tar = subprocess.Popen(["tar", "xf", "-", "-C", result_dir, "--no-same-owner"], stdin=curl.stdout, stdout=subprocess.PIPE)
# 		tar.wait()
# 		return tar.returncode, result_dir
# 	except Exception as e:
# 		print (e, file)
# 		return 1, result_dir

def main(filename="data/alllinks.txt", tasks=['segment_semantic']):

	links = [link.strip() for link in open(filename, 'r')]
	links = [(link, link.split('/')) for link in links]
	total_tasks = [task for (file, (*rest, task, archive)) in links]
	links = [file for (file, (*rest, task, archive)) in links if task in tasks]
	# links[719:719 + 119] , links[719+119:719 + 119 + 119], links[719+119+119:]
	# links = links[719+119+119:]
	print("total length of links:", len(links))
	for i, link in enumerate(links):
		res = parse.parse("http://downloads.cs.stanford.edu/downloads/taskonomy_data/{extra}/{building}_{task}.tar", link)

		if len(glob.glob(f"/semantic2/{res['building']}_{res['task']}/**", recursive=True)) > 1:
			print(f"files found in {res['building']}_{res['task']}, skipping!")
			continue
		exit_code, result_dir = process_file(link)
		i = 0
		while exit_code != 0 and i < 10:
			print(f"{i} non zero exit code, trying to unzip and compress, trying again")
			if i > 5:
				exit_code, result_dir = process_file(link)
			else:
				exit_code, result_dir = process_file(link, flags="-zC")
			i += 1

		print (f"Downloaded {result_dir}: {i}/{len(links)} files {exit_code}")
	# with Pool() as pool:
	# 	for i, result_dir in enumerate(pool.imap_unordered(process_file, links)):
			# print (f"Downloaded {result_dir}: {i}/{len(links)} files")

if __name__ == "__main__":
	Fire(main)
