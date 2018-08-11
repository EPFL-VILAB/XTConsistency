
import os, sys, math, random, itertools, glob
import numpy as np

from utils import *
from logger import Logger, VisdomLogger
from collections import Counter

logger = VisdomLogger("train", server='35.230.67.129', port=7000, env=JOB)

def count(task):
	counts = []
	for directory in glob.glob(f"/data/*_{task}/{task}"):
		counts.append(len(glob.glob(f"{directory}/*")))
	return Counter(counts)

logger.text (f"Num rgb: {count('rgb')}")
logger.text (f"Num normal: {count('normal')}")
logger.text (f"Num zdepth: {count('depth_zbuffer')}")
logger.text (f"Num curvature: {count('principal_curvature')}")