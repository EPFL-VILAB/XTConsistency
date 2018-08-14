
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
	for task in tasks:
		logger.text (f"{task}={len(glob.glob(f"/data/{building}_{task}/{task}/*.png"))}", end="")