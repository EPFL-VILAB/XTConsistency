
import os, sys, math, random, itertools, glob
import numpy as np

from utils import *
from logger import Logger, VisdomLogger
from collections import Counter

logger = VisdomLogger("train", server='35.230.67.129', port=7000, env=JOB)

buildings = [file[6:-7] for file in glob.glob("/data/*_normal")]
tasks = ['rgb', 'normal', 'depth_zbuffer', 'principal_curvature']

for building in buildings:
	logger.text (f"\nBuilding {building}: ", end="")
	task_dict = {}
	for task in tasks:
		files = glob.glob(f"/data/{building}_{task}/{task}/*.png")
		task_dict[task] = len(files)

	if len(set(task_dict.values())) == 1:
		print (logger.text("All equal"))
	else:
		for task in tasks:
			logger.text (f"{task}={task_dict[tasks]}", end=", ")