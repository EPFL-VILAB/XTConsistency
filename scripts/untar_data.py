
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
		print (f"{tar.returncode} Downloaded {result_dir}")
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

def check_dir(data_dirs):
	count = 0
	for data_dir in data_dirs:
		files = glob.glob(f"{data_dir}/{res['building']}_{res['task']}/**", recursive=True)
		print(f"{data_dir}: {len(files)}")
		count += len(files)

def main(filename="data/alllinks.txt", tasks=['keypoints2d']):

	links = [link.strip() for link in open(filename, 'r')]
	links = [(link, link.split('/')) for link in links]
	total_tasks = [task for (file, (*rest, task, archive)) in links]
	links = [file for (file, (*rest, task, archive)) in links if task in tasks]
	# links[719:719 + 119] , links[719+119:719 + 119 + 119], links[719+119+119:]
	# links = links[719+119+119:]
	print("total length of links:", len(links))
	for i, link in enumerate(links):
		res = parse.parse("http://downloads.cs.stanford.edu/downloads/taskonomy_data/{extra}/{building}_{task}.tar", link)

		# if check_dir(['/semantic2', '/semantic3']) > 1:
		# 	print(f"files found in {res['building']}_{res['task']}, skipping!")
		# 	continue
		exit_code, result_dir = process_file(link)
		j = 0
		while exit_code != 0 and j < 10:
			print(f"{j} non zero exit code ({exit_code}), trying to unzip and compress, trying again")
			exit_code, result_dir = process_file(link)
			j += 1

	# 	print (f"Downloaded {result_dir}: {i}/{len(links)} files {exit_code}")
	# with Pool() as pool:
	# 	for i, res in enumerate(pool.imap_unordered(process_file, links)):
	# 		return_code, result_dir = res
	# 		if return_code != 0:
	# 			print(f"{result_dir} non zero exit code ({return_code}), trying to unzip and compress, trying again")
	# 			pool.apply_async(process_file, (links[i],))

	# 		print (f"Downloaded {result_dir}: {i}/{len(links)} files")

if __name__ == "__main__":
	Fire(main)
