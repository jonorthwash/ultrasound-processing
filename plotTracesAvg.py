#!/usr/bin/env python3

import sys
import os
import json
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
				#vow = metadata['meta']['vowel_IPA']
				xycount = 0
				for point in metadata['trace']['points']:
					toAppend = [seriescount, xycount, vowel, point[0], point[1]]
					#print(toAppend)
					outdata.append(toAppend)
					xycount+=1
				seriescount+=1

	return outdata
	
#def py_ang(v1, v2):
	#""" Returns the angle in degrees between vectors 'v1' and 'v2' """
	#print(v1, v2)
	#cosang = np.dot(v1, v2)
	#sinang = la.norm(np.cross(v1, v2))
	#return np.degrees(np.arctan2(sinang, cosang))

#def findAnglesBetweenTwoVectors(v1s, v2s):
	#dot = np.einsum('ijk,ijk->ij',[v1s,v1s,v2s],[v2s,v1s,v2s])
	#return np.degrees(np.arccos(dot[0,:]/(np.sqrt(dot[1,:])*np.sqrt(dot[2,:]))))

#def calculateDegrees(pdData):
	#out = []
	#global origin
	#global originLow
	#v1 = np.array([origin, originLow])
	#for row in zip(pdData.x, pdData.y):
		#v2 = np.array([origin, row])
		##c = np.dot(v1,v2)/la.norm(v1)/la.norm(v2) # -> cosine of the angle
		##angle = np.arccos(np.clip(c, -1, 1))
		##angle = np.angle(v1,v2)
		#angle = findAnglesBetweenTwoVectors(v1, v2)
		##angle = py_ang(v1, v2)
		##print(np.angle(v1,v2))
		##print(angle[1])
		#out.append(angle[1])
	#return out

def calculateDegrees(pdData):
	#tan(θ) = dx/dy
	#θ = atan(dx/dy)

	## old-fashioned way:
	#out = []
	#for row in zip(pdData.x, pdData.y):
	#	dx = row[0] - origin[0]
	#	dy = origin[1] - row[1]
	#	angle = np.degrees(np.arctan(dx/dy))
	#	print(dx, dy, angle)
	#return out

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

	#return distance.euclidean(origin, (pdData.x, pdData.y))

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
	# tan(θ) = slope of the vector at angle θ
	# (x1-x2)/(y1-y2) = slope of a ray where x2,y2 is the origin point of the ray
	#destx=originx-(originy*tan(θ))
	destx = originPt[0] - originPt[1]*np.tan(np.radians(-degsFromOrigin))
	desty = 0
	#vector = [vector1.x, vector1.y]
	vector2 = [list(originPt), [destx, desty]]
	#print(degsFromOrigin, vector1, vector2)
	intersect = line_intersection(vector1, vector2)
	#print(degsFromOrigin, vector1, vector2, intersect)
	
	return intersect

def findCoordinates(degrees, distance):
	global origin

	#xOut = origin[0] + distance * np.cos(np.radians(degrees))
	#yOut = origin[1] + distance * np.sin(np.radians(degrees))

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

#def findIntersectionsVowel(degrees, distance):
def findIntersectionsVowel(byVowel):
	#global origin
	#degrees = byVowel.reset_index()["degrees"]
	#print(byVowel)
	degrees = byVowel["degrees"]
	distance = byVowel["mean"]
	
	return findCoordinates(degrees, distance)


def findIntersections(pdData, minDeg=-90, maxDeg=90, everyDeg=2):
	global origin
	traces = pdData.trace.unique()
	slices = list(range(minDeg,maxDeg,everyDeg))
	#intersects = pd.DataFrame(index=traces, columns=slices)
	#intersections = pd.DataFrame(columns=['trace', 'slice', 'x', 'y'])

	iterables = [traces, slices]
	index = pd.MultiIndex.from_product(iterables, names=['trace', 'degrees'])
	intersections = pd.DataFrame(columns=['vowel', 'distance', 'x', 'y'], index=index)

	#iterables = [traces]
	#index = pd.MultiIndex.from_product(iterables, names=['trace'])
	#intersections = pd.DataFrame(columns=['degrees', 'vowel', 'distance', 'x', 'y'], index=index)

	#intersections = pd.DataFrame(columns=['trace', 'degrees', 'distance', 'vowel', 'x', 'y'])

	#print(intersections)

	#print(intersects)
	for thisTrace in traces:
		thisVowel = str(pdData[('vowel')][pdData.trace == thisTrace].iat[0])
		#print(thisVowel)
		for thisDeg in slices:
			#print(thisDeg)
			higherPt = pdData[['point', 'x', 'y', 'degrees']][pdData.trace == thisTrace][pdData.degrees < thisDeg].sort_values(by="point").head(1)
			lowerPt = pdData[['point', 'x', 'y', 'degrees']][pdData.trace == thisTrace][pdData.degrees > thisDeg].sort_values(by="point").tail(1)
			#print(thisDeg, pdData[['point', 'x', 'y', 'degrees']][pdData.trace == thisTrace][pdData.degrees > thisDeg].sort_values(by="point"))
			#print(type(lowerPt.degrees), type(thisDeg), type(higherPt.degrees))
			#print(lowerPt.degrees, thisDeg, higherPt.degrees)
			#print(thisDeg, higherPt['degrees'], lowerPt['degrees'])
			
			#print()
			#if len(lowerPt.degrees)>0:
			#	hp = float(lowerPt.degrees)
			#else:
			#	hp = None
			#if len(higherPt.degrees)>0:
			#	lp = float(higherPt.degrees)
			#else:
			#	lp = None
			#if lp is not None and hp is not None:
			#	print(lp, thisDeg, hp)
			if len(lowerPt.degrees)>0 and len(higherPt.degrees)>0:
				#print(float(lowerPt.degrees), thisDeg, float(higherPt.degrees))
				#print(thisTrace,thisDeg)
				#intersects[thisDeg][thisTrace] = findIntersection([[float(lowerPt.x), float(lowerPt.y)], [float(higherPt.x), float(higherPt.y)]], origin, thisDeg)
				thisIntersection = findIntersection([[float(lowerPt.x), float(lowerPt.y)], [float(higherPt.x), float(higherPt.y)]], origin, thisDeg)
				#intersects[thisDeg][thisTrace] = thisIntersection
				#intersections[thisTrace, thisDeg] = pd.Series(thisIntersection)
				#intersections[thisTrace, thisDeg]['y'] = thisIntersection[1]
				intersections.ix[(thisTrace, thisDeg), ('x','y')] = thisIntersection
				intersections.ix[(thisTrace, thisDeg), ('vowel')] = thisVowel
				#tempPt = pd.Series({'x': thisIntersection[0], 'y': thisIntersection[1]})
				#intersections.ix[(thisTrace, thisDeg), ('distance')] = calculateDistance(tempPt)
				intersections.ix[(thisTrace, thisDeg), ('distance')] = distance.euclidean(origin, thisIntersection)
				#print(thisIntersection)

			#print(float(lowerPt.degrees), thisDeg, float(higherPt.degrees))
		#print(intersects.iloc[0:thisTrace,20:50])
	#print(intersections)
	#return intersects
	return intersections

if __name__ == "__main__":
	metadatas = loadData(sys.argv[1:])
	data = filterData(metadatas)
	#print(data)
	#npData = np.array(data, dtype=[('trace', 'int'), ('point', 'int'), ('vowel', 'str'), ('x', 'float'), ('y', 'float')])
	#print(npData)
	pdData = pd.DataFrame(data, columns=['trace', 'point', 'vowel', 'x', 'y'])
	#print(pdData[pdData.vowel == 'е'])
	pdData['distance'] = pd.Series(calculateDistance(pdData), index=pdData.index)
	pdData['degrees'] = pd.Series(calculateDegrees(pdData), index=pdData.index)
	#print(pdData[pdData.vowel == 'е'])
	#for row in data:
	#	print('\t'.join(map(str, row)))
	
	#f = plt.figure()

	#print(pdData.groupby(["vowel", "degrees"]).mean()) #["distance"].mean())

	intersects = findIntersections(pdData, minDeg=-90, maxDeg=90, everyDeg=2)
	#plottable = pd.DataFrame()

	#print(intersects[intersects != np.nan])
	#plt.figure()
	#intersects.plot()
	#print(intersects.dtypes)


	#print(intersects.mean())
	#print(intersects.reset_index()[["degrees", "vowel", "distance"]].groupby(["degrees", "vowel"]).mean())
	#print(intersects.reset_index().set_index(["trace", "x", "y"]).describe())
	#grouped = intersects.reset_index()[["degrees", "vowel", "distance"]].groupby(["degrees", "vowel"])
	intersects[['distance']] = intersects[['distance']].astype(float)	
	#grouped = intersects.set_index("distance").groupby(["degrees", "vowel"])
	grouped = intersects.reset_index()[["degrees", "vowel", "distance"]].groupby(["vowel", "degrees"])
	#print(grouped.ffill())
	byVowel = grouped['distance'].agg([np.mean, np.std])
	#print(byVowel) #.reset_index()[("degrees", "mean")])
	#byVowel[['x', 'y']] = findIntersectionsVowel(byVowel.reset_index()[['degrees']], byVowel[['mean']])
	#byVowel["x"], byVowel["y"] = zip(*byVowel.reset_index()[("degrees", "mean")].map(findIntersectionsVowel))
	#byVowel["x"], byVowel["y"] = zip(*map(findIntersectionsVowel, [byVowel.reset_index()["degrees"], byVowel["mean"]]))
	#byVowel["x"], byVowel["y"] = zip(*[byVowel.reset_index()["degrees"], byVowel["mean"]].map(findIntersectionsVowel))
	#byVowel["x"], byVowel["y"] = zip(*byVowel.map(findIntersectionsVowel))
	#byVowel["x"], byVowel["y"] = zip(*byVowel.reset_index().degrees.combine(byVowel.mean, func=findIntersectionsVowel))
	byVowel["x"], byVowel["y"] = zip(*byVowel.reset_index().apply(findIntersectionsVowel, axis=1))
	byVowel["maxX"], byVowel["maxY"] = zip(*byVowel.reset_index().apply(findIntersectionsVowelMax, axis=1))
	byVowel["minX"], byVowel["minY"] = zip(*byVowel.reset_index().apply(findIntersectionsVowelMin, axis=1))
	#print(byVowel)
	#.groupby(level=["degrees"])) #["distance"].mean()) #.groupby(['vowel']).mean())

	#for vowel in intersects.vowel.unique():
	#	print(intersects)
	#	print(vowel)
	#	print(intersects[intersects.vowel == vowel])
	#	for degrees in intersects[intersects.vowel == vowel].degrees.unique():
	#	# 
	#		print(vowel, degrees)

	#outs = pd.DataFrame([""])
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

	#print(outs)
	#for vowel in intersects.vowel.unique():
		#print(intersects[["x", "y"]].groupby(["vowel"])
		#outs[vowel] = 
		#print(vowel)

#	#byVowel.set_index("vowel")
#	vowelPlot = byVowel.reset_index()[['vowel', 'x', 'y', 'degrees']] #.unstack(0) #.transpose() #.reset_index().set_index(["x", "y"])
#	vowelPlot = vowelPlot.set_index(["x", "vowel"]).unstack("vowel")
#	#vowelPlot = vowelPlot.stack('x')
#	#print(vowelPlot)
#	mask=~np.isnan(vowelPlot)
#	maskedVP=vowelPlot[mask]
#	print(maskedVP)

	ax = intersects.plot(kind='scatter', x='x', y='y')
	ax.invert_yaxis()
	fig = ax.get_figure()
	fig.savefig('hargle.pdf')

	#ax2 = maskedVP.y.plot(marker='o', colormap=cm.gist_rainbow) #level=1, axis=1, kind='scatter')
	#ax2.invert_yaxis()
	#ax2.axis('equal')
	#fig2 = ax2.get_figure()
	#fig2.savefig('bargle.pdf')

	cmap = cm.get_cmap('bwr')
	
	outVowels = list(outs['avgs'].keys())
	vowColors = {}
	for i in range(len(outVowels)):
		c = cmap(float(i)/(len(outVowels)-1))
		vowColors[i] = c
		#plot(np.arange(10), (i+1) * np.arange(10), color=c, label=repr(i))
	
	#vowColors = {
	#	"ы": cmap(0.99),
	#	"ұ": cmap(0.8),
	#	"о": cmap(0.7),
	#	"а": cmap(0.6),
	#	"ә": cmap(0.5),
	#	"е": cmap(0.4),
	#	"ө": cmap(0.3),
	#	"ү": cmap(0.2),
	#	"і": cmap(0.01)
	#}

	#print(cmap(0.01), cmap(5), cmap(0.99))
	red = (1.0, 0.0, 0.0, 1.0)
	orange = (1.0, 0.5, 0.2, 1.0)
	purple = (0.7, 0.0, 0.8, 1.0)
	cyan = (0.0, 0.7, 1.0, 1.0)
	blue = (0.0, 0.0, 1.0, 1.0)
	vowColors = {
		"ы": red,
		"ұ": red,
		"о": orange,
		"а": orange,
		"ә": purple,
		"е": cyan,
		"ө": cyan,
		"ү": blue,
		"і": blue
	}
	roundedVs = ["ұ", "ү", "о", "ө"]
	
	#transparentVowColors = {
	#	"ы": (1.0, 0.0, 0.0, 0.3),
	#	"ұ": (1.0, 0.0, 0.0, 0.3),
	#	"о": (0.9, 0.4, 0.0, 0.3),
	#	"а": (0.9, 0.4, 0.0, 0.3),
	#	"ә": (0.7, 0.0, 0.7, 0.3),
	#	"е": (0.2, 0.8, 0.2, 0.3),
	#	"ө": (0.2, 0.8, 0.2, 0.3),
	#	"ү": (0.0, 0.0, 1.0, 0.3),
	#	"і": (0.0, 0.0, 1.0, 0.3)
	#}
	plt.clf()
	plt.rcParams['font.family'] = "DejaVu Sans"	
	plt.rcParams['image.cmap'] = 'bwr'

	axis = plt.gca()

	#colours = []
	#patches = []
	for vowel in outs['avgs']:
		if 'x' in outs['avgs'][vowel] and 'y' in outs['avgs'][vowel]:
			ls = 'dashdot' if vowel in roundedVs else 'solid'
			plt.plot(outs['avgs'][vowel]['x'], outs['avgs'][vowel]['y'], color=vowColors[vowel], label=vowel, ls=ls) # fontsize=20, 
			#plt.fill(outs['mins'][vowel]['x'], outs['mins'][vowel]['y'], vowColors[vowel], outs['maxes'][vowel]['x'], outs['maxes'][vowel]['y'], vowColors[vowel])
			#plt.plot(outs['mins'][vowel]['x'], outs['mins'][vowel]['y'], color=vowColors[vowel], label=vowel)
			#plt.plot(outs['maxes'][vowel]['x'], outs['maxes'][vowel]['y'], color=vowColors[vowel], label=vowel)
			xys = np.array(list(zip(outs['mins'][vowel]['x'], outs['mins'][vowel]['y']))+list(reversed(list(zip(outs['maxes'][vowel]['x'], outs['maxes'][vowel]['y'])))))
			xysNoNaN = xys[~np.isnan(xys).any(1)]
			#print(xysNoNaN)
			polygon = Polygon(xysNoNaN, closed=True, lw=None, capstyle='round', color=vowColors[vowel], alpha=0.2)
			axis.add_patch(polygon)
			#patches.append(polygon)
			#colours.append(vowColors[vowel])
		else:
			print(vowel, outs['avgs'][vowel])
	


	#fig, ax = plt.subplots()
	#patches = []
	#N = 5

	#for i in range(N):
	#	polygon = Polygon(np.random.rand(N,2), True)
	#	patches.append(polygon)

	#p = PatchCollection(patches, cmap=colours, alpha=0.3)

	#colors = 100*np.random.rand(len(patches))
	#p.set_array(np.array(colors))




	#ax.add_collection(p)
	#axis.add_collection(p)

	handles, labels = axis.get_legend_handles_labels()
	order = ["ы", "ұ", "о", "а", "ә", "е", "ө", "ү", "і"]
	replaces = ["ə", "ʊ", "uʊ", "ɑ", "æ", "iɘ", "yʉ", "ʉ", "ɘ"]
	handles2 = list(range(len(order)))
	labels2 = list(range(len(order)))
	for hl in zip(handles, labels):
		idx = order.index(hl[1])
		handles2[idx] = hl[0]
		labels2[idx] = replaces[idx]
	#print(handles, labels)
	#handles2, labels2 = zip(*hl)
	plt.legend(handles2, labels2, loc=2)


	#plt.xlim(minmax[0][0]-5, minmax[1][0]+5)
	#plt.ylim(minmax[0][1]-5, minmax[1][1]+5)
	axis.invert_yaxis()
	axis.set_aspect('equal')
	#plt.tight_layout()
	plt.savefig('bargle.pdf', format='pdf')
	#pp.savefig(dpi=2400)
	plt.close()
	
