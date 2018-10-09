from music21 import converter, instrument, note, chord
import glob    # 원문에는 없지만 아래에서 사용하기 때문에 glob 을 import 해줘야합니다.
import numpy

notes = []
for file in glob.glob("./../midi_songs/*.mid"):
    midi = converter.parse(file)
    notes_to_parse = None
    try:       # 학습 데이터 중 TypeError 를 일으키는 파일이 있어서 해놓은 예외처리
        parts = instrument.partitionByInstrument(midi)
    except TypeError:
        print('## 1 {} file occur error.'.format(file))
    if parts: # file has instrument parts
        print('## 2 {} file has instrument parts'.format(file))
        notes_to_parse = parts.parts[0].recurse()
    else: # file has notes in a flat structure
        print('## 3 {} file has notes in a flat structure'.format(file))
        notes_to_parse = midi.flat.notes
    for element in notes_to_parse:
        if isinstance(element, note.Note):
            notes.append(str(element.pitch))
        elif isinstance(element, chord.Chord):
            notes.append('.'.join(str(n) for n in element.normalOrder))

print ("note length is \n", len(notes))

sequence_length = 100 #100 previous notes
# get all pitch names
pitchnames = sorted(set(item for item in notes))
n_vocab = len(pitchnames)
# create a dictionary to map pitches to integers
note_to_int = dict((note, number) for number, note in enumerate(pitchnames))
network_input = []
network_output = []
# create input sequences and the corresponding outputs
for i in range(0, len(notes) - sequence_length, 1):
    sequence_in = notes[i:i + sequence_length]
    sequence_out = notes[i + sequence_length]
    network_input.append([note_to_int[char] for char in sequence_in])
    network_output.append(note_to_int[sequence_out])
n_patterns = len(network_input)
# reshape the input into a format compatible with LSTM layers
network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
# normalize input
network_input = network_input / float(n_vocab)
network_output = np_utils.to_categorical(network_output)

print ("input length is \n", len(network_input))
print ("output length is \n", len(network_output))
