from mido import MidiFile

# From the midi file 'filename', returns a list of tuples
# (note, onset, offset) which is sorted by onset such that
# the first tuple was the first note in music.
def load_midi(filename):
    midi = MidiFile(filename)

    #print midi.tracks
    notes_onsets_offsets = []
    time = 0
    for message in midi:
        time += message.time
        # velocity == 0 equivalent to note_off, see here:
        # http://www.kvraudio.com/forum/viewtopic.php?p=4167096
        if message.type == 'note_on' and message.velocity != 0:
            # some midis seem to have timing info on channel 0
            # but not intended to be played? (e.g. ravel)
            #if message.channel==0:
            #    continue
            notes_onsets_offsets.append((message.note, time, -1))
        elif (message.type == 'note_off') or (message.type == 'note_on' and message.velocity == 0):
            # Find the last time this note was played and update that
            # entry with offset.
            for i, e in reversed(list(enumerate(notes_onsets_offsets))):
                (note, onset, offset) = e
                if note == message.note:
                    notes_onsets_offsets[i] = (note, onset, time)
                    break
    # only keep the entries with have an offset
    notes_onsets_offsets = [x for x in notes_onsets_offsets if not x[2] == -1]
    # Make sure offset is always bigger than onset
    for note, onset, offset in notes_onsets_offsets:
        assert onset <= offset
    assert time == midi.length
    print("length of midi file" + str(midi.length))
    return notes_onsets_offsets

