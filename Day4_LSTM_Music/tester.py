from music21 import converter, instrument, note, chord
import glob    # 원문에는 없지만 아래에서 사용하기 때문에 glob 을 import 해줘야합니다.
import numpy

notes = []
for file in glob.glob("./../midi_songs/*.mid"):
	print(file)
	midi = converter.parse(file)
