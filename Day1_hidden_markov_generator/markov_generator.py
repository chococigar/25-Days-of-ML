#-*- coding: utf-8 -*-

"""
Modified from original code @ https://github.com/llSourcell/Rap_Lyric_Generator/blob/master/MarkovRap.py
"""

import random, re, os

# freqDict is a dict of dict containing frequencies
def addToDict(fileName, freqDict):
	f = open(fileName, 'r', errors='ignore')
	words = re.sub("\n", " \n", f.read()).lower().split(' ')

	# count frequencies curr -> succ
	for curr, succ in zip(words[1:], words[:-1]):
		# check if curr is already in the dict of dicts
		if curr not in freqDict:
			freqDict[curr] = {succ: 1}
		else:
			# check if the dict associated with curr already has succ
			if succ not in freqDict[curr]:
				freqDict[curr][succ] = 1;
			else:
				freqDict[curr][succ] += 1;

	# compute percentages
	probDict = {}
	for curr, currDict in freqDict.items():
		probDict[curr] = {}
		currTotal = sum(currDict.values())
		for succ in currDict:
			probDict[curr][succ] = currDict[succ] / currTotal
	return probDict

def markov_next(curr, probDict):
	if curr not in probDict:
		return random.choice(list(probDict.keys()))
	else:
		succProbs = probDict[curr]
		randProb = random.random()
		currProb = 0.0
		for succ in succProbs:
			currProb += succProbs[succ]
			if randProb <= currProb:
				return succ
		return random.choice(list(probDict.keys()))

def generateLyrics(curr, probDict, T = 200):
	lyrics = [curr]
	for t in range(T):
		lyrics.append(markov_next(lyrics[-1], probDict))
	return " ".join(lyrics)

if __name__ == '__main__':
	lyricsFreqDict = {}
	lyrics_dir = './../BTS_lyrics'
	for lyrics in os.listdir(lyrics_dir):
		lyricsProbDict = addToDict(lyrics_dir+"/"+lyrics, lyricsFreqDict)

	startWord = input("What do you want to start your lyrics with?\n > ")
	print("Alright, here's your lyrics:")
	print(generateLyrics(startWord, lyricsProbDict))