
import random, sys, os, glob, yaml, time
import argparse, subprocess, shutil
from fire import Fire


def sync(config='sync'):
    subprocess.run("git add -A".split())
    subprocess.run(["git", "commit", "-am", config])
    subprocess.run("git push origin master".split())


def update(cmd):
    subprocess.run(["nvidia-docker", "run", "nvcr.io/stanfordsvl00/task_discovery:latest", "/bin/bash", "-c", cmd])

    id = subprocess.check_output("docker container ls -lq".split()).decode()[:-1]
    subprocess.run(["nvidia-docker", "commit", id, "task_discovery"])
    subprocess.run("docker tag task_discovery nvcr.io/stanfordsvl00/task_discovery".split())
    subprocess.run("docker push nvcr.io/stanfordsvl00/task_discovery".split())

def upload(exp_id, resume=0):
    os.system("echo " + exp_id + ", " + str(resume) + " > scripts/jobinfo.txt")
    subprocess.run(["rsync", "-av", "--progress", ".", "checkpoints/" + exp_id, "--exclude", 
        "checkpoints", "--exclude", ".gi""t""", "--exclude", "data/snapshots", "--exclude", "data/results"]);
    data = subprocess.check_output(["ngc", "dataset", "upload", "-y", exp_id, "-s", "checkpoints/" + exp_id + "/"]).decode()
    i = data.index("Dataset: ")
    return int(data[(i+9):(i+14)])


def dry_run(cmd, config="job", sync_code=True):
    if sync_code:
        sync()
    cmd = "cd scaling && echo '" + config + "' > jobinfo.txt && " + cmd
    subprocess.run(["nvidia-docker", "run", "nvcr.io/stanfordsvl00/task_discovery:latest", "/bin/bash", "-c", cmd])


def snapshot(job_id):
    if not os.path.exists("data/snapshots/" + str(job_id)):
        subprocess.call(["ngc", "result", "download", str(job_id), "-s", "data/snapshots"])
    subprocess.call("ngc dataset upload snapshots -s data/snapshots -y".split())


def experiment_id(config):
    i = 1
    while os.path.exists("checkpoints/" + config + "_" + str(i)):
        i += 1
    return config + "_" + str(i)


def run(cmd, config="job", sync_code=True, datasets={"data": 11449, "models": 11863, "snapshots": 12920}, resume=0, return_command=False):
    
    exp_id = experiment_id(config)
    dataset_id = upload(exp_id, resume)

    datasets.update({"code": dataset_id})

    dataset_format = sum((["--dataset", str(dataset) + ":/" + mount] for mount, dataset in datasets.items()), [])
    cmd = "cd /code && " + cmd
    cmd = [
        "ngc",
        "batch",
        "run",
        "--instance",
        "ngcv8",
        "--name",
        exp_id,
        *dataset_format,
        "--image",
        "stanfordsvl00/task_discovery:latest",
        "--result",
        "/result",
        "--command",
        cmd,
    ]
    if not return_command:
        subprocess.call(cmd)

    print ("Experiment: ", exp_id)


def result(job_id, show_index=False):

    if show_index:
        subprocess.run(["ngc", "result", "get", str(job_id)])
    subprocess.run(["ngc", "result", "download", str(job_id), "-f", "joblog.log"], stdout=subprocess.DEVNULL)
    subprocess.run(["cat", str(job_id) + "/joblog.log"])
    subprocess.run(["rm", "-rf", str(job_id)])


if __name__ == "__main__":
    Fire()
