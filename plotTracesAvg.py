#!/usr/bin/env python3

#import sys
import os
import json

import argparse

import numpy as np
import numpy.linalg as la
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
matplotlib.style.use('ggplot')

origin = (245,606)
originLow = (240,606)
vowColors = {}
order = []
replaces = []
roundedVs = []

def setGlobals(lang):
	global vowColors
	global order
	global replaces
	global rdVowLineType
	global roundedVs

	darkred = (0.7, 0.0, 0.0, 1.0)
	red = (1.0, 0.0, 0.0, 1.0)
	lightred = (0.97, 0.40, 0.40, 1.0)
	orange = (1.0, 0.5, 0.2, 1.0)
	purple = (0.7, 0.0, 0.8, 1.0)
	cyan = (0.0, 0.7, 1.0, 1.0)
	blue = (0.0, 0.0, 1.0, 1.0)

	if lang=="kaz":
		vowColors = {
			"ы": darkred,
			"ұ": darkred,
			"о": lightred,
			"а": lightred,
			"ә": purple,
			"е": cyan,
			"ө": cyan,
			"ү": blue,
			"і": blue
		}
		roundedVs = ["ұ", "ү", "о", "ө"]

		order = ["ы", "ұ", "о", "а", "ә", "е", "ө", "ү", "і"]
		replaces = ["ə", "ʊ", "uʊ", "ɑ", "æ", "iɘ", "yʉ", "ʉ", "ɘ"]

	elif lang=="tur":
		vowColors = {
			"ı": darkred,
			"u": darkred,
			"o": lightred,
			"a": lightred,
			"â": purple,
			"e": cyan,
			"ö": cyan,
			"ü": blue,
			"i": blue
		}

		order = ["ı", "u", "o", "a", "â", "e", "ö", "ü", "i"]
		replaces = ["ɯ", "u", "o", "a", "ʲa", "e", "œ", "y", "i"]


	rdVowLineType = "dashed" # 'dashdot'


def loadData(fns):
	metadatas = []
	for fn in fns:
		if os.path.exists(fn):
			with open(fn, 'r') as metadataFile:
				fileContents = metadataFile.read()
			metadatas.append(json.loads(fileContents))

	return metadatas

def filterData(metadatas, vow=None):
	outdata = []
	seriescount=0
	for metadata in metadatas:
		if vow is not None:
			if 'meta' not in metadata:
				print("WARNING: no metadata")
			else:
				if metadata['meta']['vowel'] == vow:
					xycount = 0
					for point in metadata['trace']['points']:
						toAppend = [seriescount, xycount, point[0], point[1]]
						outdata.append(toAppend)
						xycount+=1
					seriescount+=1
		else:
			if 'meta' not in metadata:
				print("WARNING: no metadata")
			else:
				vowel = metadata['meta']['vowel']
				xycount = 0
				for point in metadata['trace']['points']:
					toAppend = [seriescount, xycount, vowel, point[0], point[1]]
					outdata.append(toAppend)
					xycount+=1
				seriescount+=1

	return outdata

def calculateDegrees(pdData):
	dx = pdData.x - origin[0]
	dy = origin[1] - pdData.y
	angle = np.degrees(np.arctan(dx/dy))
	return angle

def calculateDistance(pdData):
	global origin
	out = []
	for row in zip(pdData.x, pdData.y):
		out.append(distance.euclidean(origin, row))
	return out

def line_intersection(line1, line2):
	xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
	ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

	def det(a, b):
		return a[0] * b[1] - a[1] * b[0]

	div = det(xdiff, ydiff)
	if div == 0:
		raise Exception('lines do not intersect')

	d = (det(*line1), det(*line2))
	x = det(d, xdiff) / div
	y = det(d, ydiff) / div
	return x, y


def findIntersection(vector1, originPt, degsFromOrigin):
	destx = originPt[0] - originPt[1]*np.tan(np.radians(-degsFromOrigin))
	desty = 0
	vector2 = [list(originPt), [destx, desty]]
	intersect = line_intersection(vector1, vector2)

	return intersect

def findCoordinates(degrees, distance):
	global origin

	xOut = origin[0] + distance * np.sin(np.radians(degrees))
	yOut = origin[1] - distance * np.cos(np.radians(degrees))

	return (xOut, yOut)

def findIntersectionsVowelMin(byVowel):
	degrees = byVowel["degrees"]
	distance = byVowel["mean"]-byVowel["std"]

	return findCoordinates(degrees, distance)

def findIntersectionsVowelMax(byVowel):
	degrees = byVowel["degrees"]
	distance = byVowel["mean"]+byVowel["std"]

	return findCoordinates(degrees, distance)

def findIntersectionsVowel(byVowel):
	degrees = byVowel["degrees"]
	distance = byVowel["mean"]

	return findCoordinates(degrees, distance)

def findIntersections(pdData, minDeg=-90, maxDeg=90, everyDeg=2):
	global origin
	traces = pdData.trace.unique()
	slices = list(range(minDeg,maxDeg,everyDeg))

	iterables = [traces, slices]
	index = pd.MultiIndex.from_product(iterables, names=['trace', 'degrees'])
	intersections = pd.DataFrame(columns=['vowel', 'distance', 'x', 'y'], index=index)

	for thisTrace in traces:
		thisVowel = str(pdData[('vowel')][pdData.trace == thisTrace].iat[0])
		for thisDeg in slices:
			higherPt = pdData[['point', 'x', 'y', 'degrees']][pdData.trace == thisTrace][pdData.degrees < thisDeg].sort_values(by="point").head(1)
			lowerPt = pdData[['point', 'x', 'y', 'degrees']][pdData.trace == thisTrace][pdData.degrees > thisDeg].sort_values(by="point").tail(1)

			if len(lowerPt.degrees)>0 and len(higherPt.degrees)>0:
				thisIntersection = findIntersection([[float(lowerPt.x), float(lowerPt.y)], [float(higherPt.x), float(higherPt.y)]], origin, thisDeg)
				intersections.ix[(thisTrace, thisDeg), ('x','y')] = thisIntersection
				intersections.ix[(thisTrace, thisDeg), ('vowel')] = thisVowel
				intersections.ix[(thisTrace, thisDeg), ('distance')] = distance.euclidean(origin, thisIntersection)

	return intersections

def simplifyVowelMatrix(byVowel):
	outs = {'avgs': {}, 'mins': {}, 'maxes': {}}
	for line in byVowel.reset_index().to_dict('records'):
		vow = line['vowel']
		if vow not in outs['avgs']:
			outs['avgs'][vow] = {'x': [], 'y': []}
		outs['avgs'][vow]['x'].append(line['x'])
		outs['avgs'][vow]['y'].append(line['y'])
		if vow not in outs['mins']:
			outs['mins'][vow] = {'x': [], 'y': []}
		outs['mins'][vow]['x'].append(line['minX'])
		outs['mins'][vow]['y'].append(line['minY'])
		if vow not in outs['maxes']:
			outs['maxes'][vow] = {'x': [], 'y': []}
		outs['maxes'][vow]['x'].append(line['maxX'])
		outs['maxes'][vow]['y'].append(line['maxY'])

	return outs

def plotAvgs(outs):
	global vowColors
	global roundedVs
	global rdVowLineType

	axis = plt.gca()

	for vowel in outs['avgs']:
		if 'x' in outs['avgs'][vowel] and 'y' in outs['avgs'][vowel]:
			ls = rdVowLineType if vowel in roundedVs else 'solid'
			plt.plot(outs['avgs'][vowel]['x'], outs['avgs'][vowel]['y'], color=vowColors[vowel], label=vowel, ls=ls)

			xys = np.array(list(zip(outs['mins'][vowel]['x'], outs['mins'][vowel]['y']))+list(reversed(list(zip(outs['maxes'][vowel]['x'], outs['maxes'][vowel]['y'])))))
			xysNoNaN = xys[~np.isnan(xys).any(1)]

			polygon = Polygon(xysNoNaN, closed=True, lw=None, capstyle='round', color=vowColors[vowel], alpha=0.2)
			axis.add_patch(polygon)

		else:
			print("WARNING: No x and/or y for {}: {}".format(vowel, outs['avgs'][vowel]))


if __name__ == "__main__":

	# initialisation: argument parsing
	parser = argparse.ArgumentParser(description="Dumps target words and slide numbers from a slides file")
	parser.add_argument("--lang", "-l", help="language we're dealing with", default="kaz")
	parser.add_argument("--points", "-p", help="output graph with all radian intersection points", default=False, action="store_true")
	parser.add_argument('files', nargs='+', help="files to read traces from, e.g. *.measurement")

	args = parser.parse_args()

	setGlobals(args.lang)


	# load and parse data
	metadatas = loadData(args.files)
	data = filterData(metadatas)


	# put data into matrix
	pdData = pd.DataFrame(data, columns=['trace', 'point', 'vowel', 'x', 'y'])

	pdData['distance'] = pd.Series(calculateDistance(pdData), index=pdData.index)
	pdData['degrees'] = pd.Series(calculateDegrees(pdData), index=pdData.index)

	intersects = findIntersections(pdData, minDeg=-90, maxDeg=90, everyDeg=2)

	intersects[['distance']] = intersects[['distance']].astype(float)

	grouped = intersects.reset_index()[["degrees", "vowel", "distance"]].groupby(["vowel", "degrees"])

	byVowel = grouped['distance'].agg([np.mean, np.std])

	byVowel["x"], byVowel["y"] = zip(*byVowel.reset_index().apply(findIntersectionsVowel, axis=1))
	byVowel["maxX"], byVowel["maxY"] = zip(*byVowel.reset_index().apply(findIntersectionsVowelMax, axis=1))
	byVowel["minX"], byVowel["minY"] = zip(*byVowel.reset_index().apply(findIntersectionsVowelMin, axis=1))


	# load data for outputting avgs graph
	outs = simplifyVowelMatrix(byVowel)

	# write all points graph
	if args.points:
		ax = intersects.plot(kind='scatter', x='x', y='y')
		ax.invert_yaxis()
		fig = ax.get_figure()
		fig.savefig('graph_allintersects.pdf')


	# reset plot
	plt.clf()
	plt.rcParams['font.family'] = "DejaVu Sans"
	#plt.rcParams['image.cmap'] = 'bwr'

	# plot averages graph
	plotAvgs(outs)

	axis = plt.gca()

	handles, labels = axis.get_legend_handles_labels()
	handles2 = list(range(len(order)))
	labels2 = list(range(len(order)))
	for hl in zip(handles, labels):
		idx = order.index(hl[1])
		handles2[idx] = hl[0]
		labels2[idx] = replaces[idx]
	plt.legend(handles2, labels2, loc=2)


	axis.invert_yaxis()
	axis.set_aspect('equal')
	for thisFormat in ["pdf", "svg"]:
		plt.savefig('graph_averages.{}'.format(thisFormat), format=thisFormat)
	plt.close()

