#!/usr/bin/env python3

#import sys
import os
import json

import argparse

import itertools
from operator import sub

import numpy as np
import numpy.linalg as la
import pandas as pd
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import cm
from matplotlib.patches import Polygon
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection
#matplotlib.style.use('ggplot')
matplotlib.style.use('seaborn-white')
plt.rcParams.update({'figure.autolayout': True})

origin = (245,606) # P08
#origin = (245, 609) # P04
#origin = (245, 600) # P02
#(242, 601)
#origin = (246,607) # P07
originLow = (240,606)
vowColors = {}
order = []
replaces = []
roundedVs = []

def setGlobals(lang, code):
	global vowColors
	global order
	global replaces
	global rdVowLineType
	global roundedVs
	global tashta
	global origin
	global orthToIPA
	global vowColorsIPA
	global backVs
	global frontVs

	if code=="P08":
		origin = (245,606) # P08
	elif code=="P04":
		origin = (245, 609) # P04
	elif code=="P02":
		origin = (245, 600) # P02
		#(242, 601)
	elif code=="P07":
		origin = (246,607) # P07
	elif code=="P05":
		origin = (247,610) # P05
	elif code=="P09":
		origin = (247,613) # P09
	elif code=="P10":
		origin = (244,597) # P10
	elif code=="P11":
		origin = (245,601) # P11
	elif code=="P12":
		origin = (243,601) # P11



	darkred = (0.7, 0.0, 0.0, 1.0)
	red = (1.0, 0.0, 0.0, 1.0)
	#lightred = (0.97, 0.40, 0.40, 1.0)
	#lightred = (1.0, 0.60, 0.40, 1.0) # light orange
	lightred = (1.0, 0.41, 0.71, 1.0)  # Pink
	#lightred = (1.0, 0.41, 0.2, 1.0)  # darkish orange
	#orange = (1.0, 0.5, 0.2, 1.0)
	purple = (0.7, 0.0, 0.8, 1.0)
	#cyan = (0.0, 0.7, 1.0, 1.0)
	cyan = (0.0, 0.85, 1.0, 1.0)
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
		frontVs = ["е", "ө", "ү", "і"]
		backVs = ["а", "о", "ұ", "ы"]

		#order = ["ы", "ұ", "о", "а", "ә", "е", "ө", "ү", "і"]
		#replaces = ["ə", "ʊ", "uʊ", "ɑ", "æ", "iɘ", "yʉ", "ʉ", "ɘ"]
		order = ["ұ", "ы", "о", "а", "ә", "і", "ү", "е", "ө"]
		replaces = ["ʊ", "ə", "uʊ", "ɑ", "æ", "ɘ", "ʉ", "iɘ", "yʉ"]
		tashta = []
		vowColorsIPA = {}
		orthToIPA = {}
		for (char, IPAchar) in zip(order, replaces):
			vowColorsIPA[IPAchar] = vowColors[char]
			orthToIPA[char] = IPAchar

	elif lang=="kaz_arab":
		vowColors = {
			"ə": darkred,
			"ʊ": darkred,
			"uʊ": lightred,
			"ɑ": lightred,
			"æ": purple,
			"iɘ": cyan,
			"yʉ": cyan,
			"ʉ": blue,
			"ɘ": blue
		}
		roundedVs = ["ʊ", "ʉ", "uʊ", "yʉ"]
		frontVs = ["iɘ", "yʉ", "ʉ", "ɘ"]
		backVs = ["ɑ", "uʊ", "ʊ", "ə"]

		#order = ["ы", "ұ", "о", "а", "ә", "е", "ө", "ү", "і"]
		#replaces = ["ə", "ʊ", "uʊ", "ɑ", "æ", "iɘ", "yʉ", "ʉ", "ɘ"]
		order = ["ʊ", "ə", "uʊ", "ɑ", "æ", "ɘ", "ʉ", "iɘ", "yʉ"]
		replaces = ["ʊ", "ə", "uʊ", "ɑ", "æ", "ɘ", "ʉ", "iɘ", "yʉ"]
		tashta = ["ى", "ۇ", "ە", "و", "ا"]
		vowColorsIPA = vowColors
		orthToIPA = {}
		for (char, IPAchar) in zip(order, replaces):
			orthToIPA[char] = IPAchar

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
		roundedVs = ["u", "ü", "o", "ö"]
		frontVs = ["e", "ö", "ü", "i"]
		backVs = ["a", "o", "u", "ı"]

		#order = ["ı", "u", "o", "a", "â", "e", "ö", "ü", "i"]
		#replaces = ["ɯ", "u", "o", "a", "ʲa", "e", "œ", "y", "i"]
		order = ["u", "ı", "o", "a", "â", "i", "ü", "e", "ö"]
		replaces = ["u", "ɯ", "o", "a", "ʲa", "i", "y", "e", "œ"]
		tashta = []
		vowColorsIPA = {}
		orthToIPA = {}
		for (char, IPAchar) in zip(order, replaces):
			vowColorsIPA[IPAchar] = vowColors[char]
			orthToIPA[char] = IPAchar

	elif lang=="kir":
		vowColors = {
			"ы": darkred,
			"у": darkred,
			"о": lightred,
			"а": lightred,
			"е": cyan,
			"ө": cyan,
			"ү": blue,
			"и": blue
		}
		roundedVs = ["у", "ү", "о", "ө"]
		frontVs = ["е", "ө", "ү", "и"]
		backVs = ["а", "о", "у", "ы"]

		order = ["у", "ы", "о", "а", "и", "ү", "е", "ө"]
		replaces = ["u", "ɯ", "o", "a", "i", "y", "e", "œ"]
		tashta = []
		vowColorsIPA = {}
		orthToIPA = {}
		for (char, IPAchar) in zip(order, replaces):
			vowColorsIPA[IPAchar] = vowColors[char]
			orthToIPA[char] = IPAchar

	rdVowLineType = "dashed" # 'dashdot'


def loadData(fns):
	metadatas = {}
	for fn in fns:
		if os.path.exists(fn):
			with open(fn, 'r') as metadataFile:
				fileContents = metadataFile.read()
			metadatas[fn] = json.loads(fileContents)

	return metadatas

def filterData(metadatas, vow=None):
	global tashta
	outdata = []
	seriescount=0
	for fn in metadatas:
		metadata = metadatas[fn]
		# "fix" vowels before proceeding
		if 'meta' in metadata:
			if metadata['meta']['vowel'] in tashta and 'vowel_IPA' in metadata['meta']:
				# since perso-arabic characters are not 1-to-1, cf. P02's data
				metadata['meta']['vowel'] = metadata['meta']['vowel_IPA']
				metadata['meta']['vowel'] = metadata['meta']['vowel'].replace('[', '').replace(']', '')
		#print(metadata['meta']['vowel'])
		else:
			print("WARNING: no metadata ({})".format(fn))

		if vow is not None:
			if 'meta' not in metadata:
				#print("WARNING: no metadata")
				print("WARNING: no metadata ({})".format(fn))
			else:
				# filter by provided vowels
				for vowel in vow:
					if metadata['meta']['vowel'] == vowel:
						xycount = 0
						for point in metadata['trace']['points']:
							toAppend = [seriescount, xycount, vowel, point[0], point[1]]
							outdata.append(toAppend)
							xycount+=1
						seriescount+=1
		else:
			if 'meta' not in metadata:
				#print("WARNING: no metadata")
				print("WARNING: no metadata ({})".format(fn))
			else:
				vowel = metadata['meta']['vowel']
				xycount = 0
				if 'trace' in metadata:
					for point in metadata['trace']['points']:
						toAppend = [seriescount, xycount, vowel, point[0], point[1]]
						outdata.append(toAppend)
						xycount+=1
					seriescount+=1
				else:
					print("WARNING: no trace data ({})".format(fn))

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
			plt.plot(outs['avgs'][vowel]['x'], outs['avgs'][vowel]['y'], color=vowColors[vowel], label=vowel, ls=ls, lw=1)

			xys = np.array(list(zip(outs['mins'][vowel]['x'], outs['mins'][vowel]['y']))+list(reversed(list(zip(outs['maxes'][vowel]['x'], outs['maxes'][vowel]['y'])))))
			xysNoNaN = xys[~np.isnan(xys).any(1)]
			#print(vowel)
			#print(xysNoNaN)
			# plot standard mins and maxes, but only if there's more than one trace:
			#if np.NaN not in outs['mins'][vowel]['x'] and np.NaN not in outs['mins'][vowel]['y'] and np.NaN not in outs['maxes'][vowel]['x'] and np.NaN not in outs['maxes'][vowel]['y']:
			#print(vowel, np.all(np.isnan(outs['mins'][vowel]['x'])), np.all(np.isnan(outs['mins'][vowel]['y'])), np.all(np.isnan(outs['maxes'][vowel]['x'])), np.all(np.isnan(outs['maxes'][vowel]['y'])))
			if not (np.all(np.isnan(outs['mins'][vowel]['x'])) or np.all(np.isnan(outs['mins'][vowel]['y'])) or np.all(np.isnan(outs['maxes'][vowel]['x'])) or np.all(np.isnan(outs['maxes'][vowel]['y']))):
				#print(vowel, outs['mins'][vowel]['x'])
				polygon = Polygon(xysNoNaN, closed=True, lw=None, capstyle='round', color=vowColors[vowel], alpha=0.2)
				axis.add_patch(polygon)

		else:
			print("WARNING: No x and/or y for {}: {}".format(vowel, outs['avgs'][vowel]))


def main(args):
	setGlobals(args.lang, args.code)

	if args.filterby != None:
		filter_vowels = args.filterby.split(",")
	else:
		filter_vowels = None
	# load and parse data
	metadatas = loadData(args.files)
	data = filterData(metadatas, filter_vowels)


	# put data into matrix
	pdData = pd.DataFrame(data, columns=['trace', 'point', 'vowel', 'x', 'y'])

	pdData['distance'] = pd.Series(calculateDistance(pdData), index=pdData.index)
	pdData['degrees'] = pd.Series(calculateDegrees(pdData), index=pdData.index)

	intersects = findIntersections(pdData, minDeg=-90, maxDeg=90, everyDeg=int(args.resolution))

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

	def label_point(x, y, val, ax):
		global orthToIPA
		global vowColorsIPA
		a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
		for i, point in a.iterrows():
			ax.text(point['x'], point['y'], orthToIPA[str(point['val'])], horizontalalignment="center", verticalalignment="center", color=vowColors[point['val']])


	if args.extremes:
		#extremes = intersects.max()
		extremes = intersects.reset_index()[["trace", "degrees", "vowel", "distance", "x", "y"]].sort_values(by='distance', ascending=False).groupby(["trace"]).first()
		ax = extremes.plot(kind='scatter', x='x', y='y', alpha=0)
		label_point(extremes.x, extremes.y, extremes.vowel, ax)
		ax.invert_yaxis()
		fig = ax.get_figure()
		for thisFormat in ["pdf", "svg"]:
			plt.savefig('graph_extremes.{}'.format(thisFormat), format=thisFormat)

	if args.avgdist:
		#grouped['anteriority'] = "back" if grouped['vowel'] in backVs else "front"
		#grouped['anteriority'] = grouped['vowel'] in backVs
		#mask = grouped['vowel'] in backVs
		#print(mask)
		#print(grouped.index(), grouped.head())
		#groupedBack = grouped.where(grouped['vowel'] in backVs)
		#groupedFront = grouped.where(grouped['vowel'] in frontVs)
		#print(groupedBack, groupedFront)
		#for degree in grouped.groupby(['degrees']).nunique().reset_index():
		#for degree in grouped.groupby(['degrees']).apply(lambda x: len(x.unique())):
		#for degree in grouped.reset_index().groupby(['degrees']):
		#	print(degree)
		backVmask = intersects['vowel'].isin(backVs)
		frontVmask = intersects['vowel'].isin(frontVs)
		allDegrees = intersects.reset_index()[["degrees", "vowel", "distance"]]
		backDegrees = intersects[backVmask].reset_index()[["degrees", "vowel", "distance"]]
		frontDegrees = intersects[frontVmask].reset_index()[["degrees", "vowel", "distance"]]
		#print(backDegrees.sort_by('degrees').head(), frontDegrees.head())
		#print(allDegrees.dropna()["degrees"].unique())
		allDegrees = np.sort(allDegrees.dropna()["degrees"].unique())
		outMeans = pd.DataFrame(columns=['mean', 'σminus', 'σplus', 'stddev', 'stddevs'], index=allDegrees)#, index='degrees')
		#print(outMeans)
		for degree in allDegrees:
			#print(degree)
		#print(backDegrees.head())
			#print(backDegrees[backDegrees["degrees"] == degree]['distance'], frontDegrees[frontDegrees["degrees"] == degree]['distance'])
			#print(backDegrees)
			#for element in itertools.product(backDegrees[backDegrees["degrees"] == degree]['distance'], frontDegrees[frontDegrees["degrees"] == degree]['distance']):
			#	print(element)
			# JNW: FIX BELOW!  find how to get diff of product
			lenBackDegs = len(backDegrees[backDegrees["degrees"] == degree]['distance'].dropna())
			lenFrontDegs = len(frontDegrees[frontDegrees["degrees"] == degree]['distance'].dropna())
			if lenBackDegs > 2 and lenFrontDegs > 2:
				thisDist = itertools.product(backDegrees[backDegrees["degrees"] == degree]['distance'], frontDegrees[frontDegrees["degrees"] == degree]['distance'])
				#print("====", degree, [x for x in thisDist])
				#print([x for x in thisDist])
				#thisDiffs = set(map(sub, thisDist))
				#print([x[0] - x[1] for x in (thisDist)])
				thisDiffs = [x[0] - x[1] for x in (thisDist)]
				#print(thisDiffs)
				(thisStd, thisMean) = (np.std(thisDiffs), np.mean(thisDiffs))
				#print("====", degree, thisDiffs)
				#outMeans.append({"degrees": degree, "mean": thisMean, "sminus": thisMean-thisStd, "splus": thisMean+thisStd}, ignore_index=True)
				#print(thisMean, thisStd)
				#outMeans.ix[degree] = {"mean": thisMean, "σminus": thisMean-thisStd, "σplus": thisMean+thisStd} #, "stddev": thisStd}
				outMeans.ix[degree] = {"mean": thisMean, "σminus": thisMean-thisStd, "σplus": thisMean+thisStd, "stddevs": thisMean/thisStd, "stddev": thisStd}
				#for tup in thisDist:
				#	print("tuple:", tup)
				#	print("difference:", tup[0] - tup[1])
				#thisDiffs = np.subtract(np.array(thisDist))
				#thisDiff = thisDist[0] - thisDist[1]
				#print(thisDiffs)
			else:
				print(degree, lenBackDegs, lenFrontDegs)
		print(outMeans)

		last = None
		for (col, line) in outMeans.iterrows():
			#print(line)
			if last is not None:
				if last['stddevs'] > 0:
					if line['stddevs'] < 0:
						zeroIntersect = (lastcol, col)
						#print(lastcol, col)

			last = line
			lastcol = col

		# reset plot
		plt.clf()
		#plt.style.use('grayscale')
		plt.style.use('bmh')
		plt.rcParams['font.family'] = "DejaVu Sans"
		fig, ax1 = plt.subplots()



		ax1.set_xlabel('angle from origin (degrees)')
		ax1.set_ylabel('distance between anterior and posterior (px)')
		#ax = outMeans.plot(kind='line')#, y='mean')
		ax1.axhline(0, color='darkgrey', zorder=1)
		lineMean, = ax1.plot(outMeans['mean'], zorder=3)

		#print(outMeans.index, outMeans['σminus'].values)
		lineRange = ax1.fill_between([float(i) for i in outMeans.index], [float(i) for i in outMeans['σminus'].values], [float(i) for i in outMeans['σplus'].values], alpha=0.2, zorder=2)
		ax2 = ax1.twinx()
		#ax1.get_shared_y_axes().join(ax1, ax2)
		#ax2.set_yticks(ax1, ax2)
		ax1ylim = ax1.get_ylim()
		ax2.set_ylim(ax1ylim[0]/10, ax1ylim[1]/10)
		#print(ax1ylim)
		#print(ax1yticks[0], ax1yticks[:-1])

		ax2.tick_params(axis='y', colors='red')
		ax2.grid(None)

		maxStds = outMeans['stddevs'].max()
		maxStdsIntersect = outMeans['stddevs'].idxmax()
		minStds = outMeans['stddevs'].min()
		minStdsIntersect = outMeans['stddevs'].idxmin()
		#print(maxStds, minStds)
		if maxStds >= 0:
			lineExtreme = ax2.axvline(maxStdsIntersect, zorder=6, color='red', linewidth=0.5)
		if minStds <= 0:
			lineExtreme = ax2.axvline(minStdsIntersect, zorder=6, color='red', linewidth=0.5)
		#ax.invert_yaxis()

		lineStds, = ax2.plot(outMeans['stddevs'], color='red', zorder=5, alpha=0.5)
		ax2.set_yticks(np.linspace(ax2.get_yticks()[0],ax2.get_yticks()[-1],len(ax1.get_yticks())))
		#ax2.set_ylabel('standard deviations of mean from 0', color='red')
		ax2.set_ylabel('standard deviations of mean from 0')#, color='red')
		#fig = ax1.get_figure()
		rectRange = Rectangle((0, 0), 1, 1, facecolor=plt.getp(lineRange, 'facecolor')[0])
		#plt.legend([lineMean, rectRange, lineStds, lineExtreme], ["mean diff", "1 standard deviation", "№ stddev from mean", "№ stddev extremes"], loc=3)
		ax1.legend([lineMean, rectRange], ["mean diff", "±1σ"], loc=3)
		ax2.legend([lineStds, lineExtreme], ["mean diff Z-score", "max/min Z-scores"], loc=1)
		for thisFormat in ["pdf", "svg"]:
			plt.savefig('graph_differences.{}'.format(thisFormat), format=thisFormat)
		print(np.mean(zeroIntersect), maxStdsIntersect, minStdsIntersect)


	# reset plot
	plt.clf()
	plt.style.use('seaborn-white')
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
	plt.legend(handles2, labels2, loc=4, ncol=2)


	axis.invert_yaxis()
	axis.set_aspect('equal')
	plt.tight_layout()
	for thisFormat in ["pdf", "svg"]:
		plt.savefig('graph_averages.{}'.format(thisFormat), format=thisFormat)
	plt.close()


if __name__ == "__main__":

	# initialisation: argument parsing
	parser = argparse.ArgumentParser(description="Dumps target words and slide numbers from a slides file")
	parser.add_argument("--lang", "-l", help="language we're dealing with", default="kaz")
	parser.add_argument("--points", "-p", help="output graph with all radian intersection points", default=False, action="store_true")
	parser.add_argument("--code", "-c", help="participant code, e.g. P01", default="P08")
	parser.add_argument("--extremes", "-e", help="find extremes / constriction points", default=False, action="store_true")
	parser.add_argument("--avgdist", "-a", help="find average distances between front and back", default=False, action="store_true")
	parser.add_argument('files', nargs='+', help="files to read traces from, e.g. *.measurement")
	parser.add_argument("--filterby", "-f", help="limit to only certain vowels, separated by commas", default=None)
	parser.add_argument("--resolution", "-r", help="number of degrees to interpolate by", default=2, type=int)

	args = parser.parse_args()

	main(args)
