
import random, sys, os, glob, yaml, time
import argparse, subprocess, shutil
from fire import Fire


def sync():
	subprocess.run("git add -A".split())
	subprocess.run("git commit -am sync".split())
	subprocess.run("git push origin master".split())

def update(cmd):
	subprocess.run(["nvidia-docker", "run", "nvcr.io/stanfordsvl00/task_discovery:latest", "/bin/bash",
					"-c", cmd])
	
	id = subprocess.check_output("docker container ls -lq".split()).decode()[:-1]
	subprocess.run(["nvidia-docker", "commit", id, "task_discovery"])
	subprocess.run("docker tag task_discovery nvcr.io/stanfordsvl00/task_discovery".split())
	subprocess.run("docker push nvcr.io/stanfordsvl00/task_discovery".split())

def dry_run(cmd, config="job", sync_code=True):
	if sync_code: sync()
	cmd = "cd scaling && echo \'" + config + '\' > jobinfo.txt && ' + cmd
	subprocess.run(["nvidia-docker", "run", "nvcr.io/stanfordsvl00/task_discovery:latest", "/bin/bash",
					"-c", cmd])

def run(cmd, config="job", sync_code=True, datasets={"data": 11449, "models": 11863}, return_command=False):
	if sync_code: sync()

	dataset_format = sum(["--dataset", str(dataset) + ":/" + mount] for mount, dataset in datasets.items())
	print (dataset_format)
	cmd = "cd scaling && git pull && echo \'" + config + '\' > jobinfo.txt && ' + cmd
	cmd = ["ngc", "batch", "run", "--instance", "ngcv8", "--name", config, "--dataset", str(dataset) + ":/data",
				"--image", "stanfordsvl00/task_discovery:latest", "--result", "/result",
				"--command", cmd]
	print (" ".join(cmd))
	if not return_command: 
		subprocess.call(cmd)

def result(job_id, show_index=False):
	if show_index: subprocess.run(["ngc", "result", "get", str(job_id)])
	subprocess.run(["ngc", "result", "download", str(job_id), "-f", "joblog.log"], stdout=subprocess.DEVNULL)
	subprocess.run(["cat", str(job_id) + "/joblog.log"])
	subprocess.run(["rm", "-rf", str(job_id)])


if __name__ == "__main__":
	Fire({"run": run, "sync": sync, "update": update, "dry_run": dry_run, "result": result})


