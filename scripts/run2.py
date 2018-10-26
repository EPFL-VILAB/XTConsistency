
import random, sys, os, glob, yaml, time
import argparse, subprocess, shutil, shlex
from fire import Fire
from utils import elapsed

import IPython


def execute(cmd, config="default", experiment_id=None, shutdown=False, debug=False):

    elapsed()
    try:
        run_log = yaml.load(open("checkpoints/runlog.yml"))
    except:
        run_log = {}

    mode = config
    run_data = run_log[mode] = run_log.get(mode, {})
    run_data["runs"] = run_data.get("runs", 0) + 1
    run_name = experiment_id or (mode + "_" + str(run_data["runs"]))
    run_data[run_name] = run_data.get(run_name, {"config": config, "cmd": cmd, "status": "Running"})
    run_data = run_data[run_name]

    print(f"Running job: {run_name}")

    shutil.rmtree("output/", ignore_errors=True)
    os.makedirs("output/")
    os.makedirs(f"checkpoints/{run_name}", exist_ok=True)
    os.system("echo " + run_name + ", 0, mount > scripts/jobinfo.txt")

    cmd = shlex.split(cmd)
    if cmd[0] == "python" and debug:
        cmd[0] = "ipython"
        cmd.insert(1, "-i")
    elif cmd[0] == "python":
        cmd.insert(1, "-u")

    print(" ".join(cmd))
    process = subprocess.Popen(cmd, shell=False, stdout=subprocess.PIPE, universal_newlines=True)

    try:
        with open(f"checkpoints/{run_name}/stdout.txt", "w") as outfile:
            for stdout_line in iter(process.stdout.readline, ""):
                print(stdout_line, end="")
                outfile.write(stdout_line)

        return_code = process.wait()
        run_data["status"] = "Error" if return_code else "Complete"
    except KeyboardInterrupt:
        print("\nKilled by user.")
        process.kill()
        run_data["status"] = "Killed"
    except OSError:
        print("\nSystem error.")
        process.kill()
        run_data["status"] = "Error"

    process.kill()

    if debug and run_data["status"] != "Complete":
        return

    subprocess.run(["rsync", "-av", "--progress", ".", "checkpoints/" + run_name, "--exclude", 
        "checkpoints", "--exclude", ".git", "--exclude", "data/snapshots", "--exclude", "data/results", "--exclude", "mount"],
        stdout=subprocess.DEVNULL);

    yaml.safe_dump(run_log, open("checkpoints/runlog.yml", "w"), allow_unicode=True, default_flow_style=False)
    yaml.safe_dump(run_data, open(f"checkpoints/{run_name}/comments.yml", "w"), allow_unicode=True, default_flow_style=False)

    interval = elapsed()
    print(f"Program ended after {interval:0.4f} seconds.")
    if shutdown and run_data["status"] != "Killed" and interval > 60:
        print(f"Shutting down in 1 minute.")
        time.sleep(60)
        subprocess.call("sudo shutdown -h now", shell=True)

def run(cmd, config="default", experiment_id=None, shutdown=False, debug=False):
    cmd = f""" screen -S {config} bash -c "sudo /home/shared/anaconda3/bin/python -m scripts.run2 execute \\"{cmd}\\" --config {config} --experiment-id {experiment_id} --shutdown {shutdown} --debug {debug}; bash" """
    subprocess.call(shlex.split(cmd))


if __name__ == "__main__":
    Fire({"run": run, "execute": execute})
