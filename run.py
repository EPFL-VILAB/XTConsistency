
import random, sys, os, glob, yaml, time
import argparse, subprocess, shutil
from fire import Fire

def sync():
	subprocess.run("git add -A".split())
	subprocess.run("git commit -am sync".split())
	subprocess.run("git push origin master".split())

	cmd = "cd scaling; git config --global user.name nikcheerla;"\
			"git config --global user.email nikcheerla@gmail.com;"\
			"git stash; git pull"
	subprocess.run(["nvidia-docker", "run", "nvcr.io/stanfordsvl00/taskonomy2:latest", "/bin/bash",
					"-c", cmd])
	
	id = subprocess.check_output("docker container ls -lq".split()).decode()[:-1]
	subprocess.run(["nvidia-docker", "commit", id, "taskonomy2"])
	subprocess.run("docker tag taskonomy2 nvcr.io/stanfordsvl00/taskonomy2".split())
	subprocess.run("docker push nvcr.io/stanfordsvl00/taskonomy2".split())

def update(cmd):
	
	cmd = "cd scaling && " + cmd
	subprocess.run(["nvidia-docker", "run", "nvcr.io/stanfordsvl00/taskonomy2:latest", "/bin/bash",
					"-c", cmd])
	
	id = subprocess.check_output("docker container ls -lq".split()).decode()[:-1]
	subprocess.run(["nvidia-docker", "commit", id, "taskonomy2"])
	subprocess.run("docker tag taskonomy2 nvcr.io/stanfordsvl00/taskonomy2".split())
	subprocess.run("docker push nvcr.io/stanfordsvl00/taskonomy2".split())

def dry_run(cmd, config="job", sync_code=True):
	if sync_code: sync()
	cmd = "cd scaling && echo \'" + config + '\' > jobinfo.txt && ' + cmd
	subprocess.run(["nvidia-docker", "run", "nvcr.io/stanfordsvl00/taskonomy2:latest", "/bin/bash",
					"-c", cmd])

def run(cmd, config="job", sync_code=True, dataset=11319, return_command=False):
	if sync_code: sync()

	cmd = "cd scaling && git pull && echo \'" + config + '\' > jobinfo.txt && ' + cmd
	cmd = ["ngc", "batch", "run", "--instance", "ngcv8", "--name", config, "--dataset", str(dataset) + ":/data",
				"--image", "stanfordsvl00/taskonomy2:latest", "--result", "/result",
				"--command", cmd]
	print (" ".join(cmd))
	if not return_command: 
		subprocess.call(cmd)

def results(job_id, show_index=False):
	if show_index: subprocess.run(["ngc", "result", "get", str(job_id)])
	subprocess.run(["ngc", "result", "download", str(job_id), "-f", "joblog.log"], stdout=subprocess.DEVNULL)
	subprocess.run(["cat", str(job_id) + "/joblog.log"])
	subprocess.run(["rm", "-rf", str(job_id)])


if __name__ == "__main__":
	Fire()


