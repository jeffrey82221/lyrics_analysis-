#read in random song names and their corresponding artist names from file

songdata = []
for line in open('ransong.txt'):
    songdata.append(line)

len(songdata)

songdata

artistnames = []
songnames = []

for element in songdata:
    theSongData = element.split("-")
    artistnames.append(theSongData[0])
    songnames.append(theSongData[1])

len(artistnames)
len(songnames)
songcount = len(artistnames)


# retrieve lyrics of each song from lyricswiki or lyricsnmusic by accessing "songtext" command line tool

import os
import subprocess
import string

songcount


env1 = os.environ.copy()
env1['SONGTEXT_DEFAULT_API'] = 'lyricsnmusic'
env1["LYRICSNMUSIC_API_KEY"] = '\"<0395c36be304b59176979f5ced5d3a>\"'

env2 = os.environ.copy()
env2['SONGTEXT_DEFAULT_API'] = 'lyricwiki'
env2["LYRICSNMUSIC_API_KEY"] = '\"<0395c36be304b59176979f5ced5d3a>\"'

# Filter out No Found message:
# The requested track is not viewable.
# Your query did not match any tracks.
lyrics = []
for i in range(songcount):
    try:
        #new_env['SONGTEXT_DEFAULT_API'] = 'lyricsnmusic'
        p = subprocess.Popen(["songtext", "--api", "lyricwiki", "-a", artistnames[
                             i], "-t", songnames[i]], env=env2, stdin=PIPE, stdout=PIPE)
        theLyric = p.communicate()[0].decode('utf-8')
        if "Your query did not match any tracks." in theLyric:
            #new_env['SONGTEXT_DEFAULT_API'] = 'lyricwiki'
            q = subprocess.Popen(["songtext", "--api", "lyricsnmusic", "-a", artistnames[
                                 i], "-t", songnames[i]], env=env1, stdin=PIPE, stdout=PIPE)
            theLyric = q.communicate()[0].decode('utf-8')
        lyrics.append(theLyric)
        if i % 10 == 0:
            print(i)
    except:
        lyrics.append("no result")
        print("Error happened in ", i)

i= 23
artistnames[i]
songnames[i]
lyrics[i]

# check if the lyrics are real :
for i in range(len(lyrics)):
    if artistnames[i] not in lyrics[i] and songnames[i] not in lyrics[i]:
        print(i)


# clean out the strange result with "A Day to Remember: Here's to the Past"

for i in range(len(lyrics)):
    if "A Day to Remember: Here's to the Past" in lyrics[i]:
        print(i)
        p = subprocess.Popen(["songtext", "--api","lyricwiki", "-a", artistnames[
                             i], "-t", songnames[i]], stdin=PIPE, stdout=PIPE)
        lyrics[i] = p.communicate()[0].decode('utf-8')


# check if the lyrics are real again:
for i in range(len(lyrics)):
    if artistnames[i] not in lyrics[i] and songnames[i] not in lyrics[i]:
        print(i)

for i in range(len(lyrics)):
    if "Your query did not match any tracks." in lyrics[i]:
        print(i)
        p = subprocess.Popen(["songtext", "--api", "lyricsnmusic", "-a", artistnames[
                             i], "-t", songnames[i]], env=env1, stdin=PIPE, stdout=PIPE)
        lyrics[i] = p.communicate()[0].decode('utf-8')

i = 21
artistnames[i],songnames[i]
print(lyrics[i])

len(set(lyrics))


lyrics

lyrics_set = set(lyrics)

lyrics_set = list(lyrics_set)
len(lyrics_set)
lyrics_set[217]
lyrics_set.pop(217)

len(lyrics_set)

artistnames_clean = []
songnames_clean = []
lyric_clean = []



for i in range(len(lyrics_set)):
    try:
        a,b = lyrics_set[i].split('\n--') #split into title + lyric
        a1,a2 = a.split(':') #split into artist + songname
        songnames_clean.append(a2)
        artistnames_clean.append(a1.split('\n')[-1])
        lyric_clean.append(b.split('--\n')[-1])
    except:
        print(i)
len(artistnames_clean)
len(songnames_clean)
len(lyric_clean)


theArtistFile = open('artists','w')
for artist in artistnames_clean:
    theArtistFile.write(artist)
    theArtistFile.write('\n')
# 3, 21, 22, 28, 30, 33, 57, 58, 59, 60, 63, 65, 68, 69, 81, 83, 97, 101, 139, 144, 152, 154, 161, 171, 181, 182, 184

#write the artist name in file by sequence
theArtistFile.close()
len(lyrics)
lyrics
print(lyrics[7])

#write the song name in file by sequence
theTitleFile = open('title','w')
for songtitle in songnames_clean:
    theTitleFile.write(songtitle)
    theTitleFile.write('\n')
theTitleFile.close()


#write all lyrics into file. Each named by their name and title.
for i in range(len(lyric_clean)):
    try:
        theFile = open('lyrics/'+artistnames_clean[i]+'-'+songnames_clean[i],'w')
        theFile.write(lyric_clean[i])
        theFile.close()
    except:
        print(i)

# something happend while using '/' as filename , clean this up by hand
artistnames_clean[73]
songnames_clean[73]
theFile = open('lyrics/'+artistnames_clean[73]+'-'+' Erase&Rewind','w')
theFile.write(lyric_clean[73])
theFile.close()
