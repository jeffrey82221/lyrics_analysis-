#NOTE ######################################################################

# NOTE some special sentences needs to be remove
# if all character are not in Ascii letters
# word that includes [( on left or ]) on right
# sentence that contain no words
#  - it's OK
# remove the sentences that startTime is 0 and endTime is also 0
# sentence with same starttime and end time
# song adjecent sentences with same startitme and endtime : (might be title or have other functionality)
# sentence with : at the end might be some kind of title


# NOTE some special sentences needs to be alter
# all uppercase sentences
# remove the words in () or []
# remove the ... after some words, or before some words
# sometimes there are chinese tanslation cat after the original sentence (this kind of sentence should be clean particularly)
#   - remove the none ascii part
# sometimes the front sentences contain <sec_st> , TODO understand the meaning and try to clean it
# sometime the front sentences might be the introduction sentence (eg. with the titles and the artists)

# NOTE some spectial words in lyrics might need to be stem!
# words that contain lot of '-'
# words that are an abbreviation ex. I'm ,  thinkin' , He's , ...
import nltk

class SongInfo:
    ID = None
    title = ""
    album = ""
    artist = ""
    def __init__(self,array):
        self.ID = int(array[0])
        self.title = array[1]
        self.album = array[2]
        self.artist = array[3]
    def print_info(self):
        print self.ID,",",self.title,",",self.album,",",self.artist
class SongInfoData:
    songinfos = []
    ids = []
    def __init__(self,array):
        for element in array:
            si = SongInfo(element)
            self.songinfos.append(si)
            self.ids.append(si.ID)
    def findInfobyID(self,id):
        try:
            return self.songinfos[self.ids.index(id)]
        except:
            return SongInfo(["0","unknown","unknown","unknown"])


# TODO tokenize each sentences + stem each sentences


# TODO : Before tokenize :
# Remove none ascii letters
# remove words in parenthese
# set all uppercase to lower case (by .lower)

###Functions for cleaning the words in lyrics sentences

def removenoneAscii(array):
    for char in array:
        if ord(char)>=128:
            array = array.replace(char,'')
    return array
def removeparenthese(array,type):
    if type==1:
        start = array.find( '(' )
        end = array.find( ')' )
    elif type==2:
        start = array.find( '[' )
        end = array.find( ']' )
    elif type==3:
        start = array.find( '<' )
        end = array.find( '>' )
    else:
        return array
    if start != -1 and end != -1:
        result = array[:start] + array[end+1:]
        return result
    else:
        return array
def sentence_cleaning(array):
    array = removenoneAscii(array)
    array = removeparenthese(array,2)
    array = removeparenthese(array,3)
    return array.lower()



# TODO remove the sentences with following properties :
# remove the senteces ends with "" (with no abc in them):
# remove the sentences with starttime = 0 and end time = 0
# remove the adj sentenes with same starttime and same endtime #REVIEW


def noabc(array):
    if sum([element.isalpha() for element in array])==0:
        return True
    else:
        return False



# TODO : After tokenize : (stemming process)
# remove the ... after or before words
# remove special character such as . , ! ? ... check by eyes #REVIEW
# recover the word with "-" inside it

def remove_front_mark(word):
    try:
        if (not word[0].isalpha()):
            return remove_front_mark(word[1:])
        else:
            return word
    except:
        return word
def remove_end_mark(word):
    try:
        if (not word[-1].isalpha()):
            word = word[:-1]
            return remove_end_mark(word)
        else:
            return word
    except:
        return word


def applystemming(term,function_array):
    for element in function_array:
        term = element(term)
    return term

def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return nltk.corpus.wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return nltk.corpus.wordnet.VERB
    elif treebank_tag.startswith('N'):
        return nltk.corpus.wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return nltk.corpus.wordnet.ADV
    else:
        return ''

wnl = nltk.WordNetLemmatizer()
tagger = nltk.tag.PerceptronTagger()
t = ['','']
t.remove('')
t
x = []
x[:] = (value for value in x if value != '')
x
class SentenceInfo:
    startTime = 0
    endTime = 0
    sentenceType = 0
    # TODO: What's the meaning of the types of the sentences?
    # A.
    # meaning sing by which singer
    # meaning not the lyrics sentence (introduction sentences)
    # meaning the same singer singing with different vocal voice ! (ex.different pitch, different timbre, hibrid voice...)
    # meaning rap!!
    sentence = ""
    tokenized_sentences = []
    pos_tags = []
    def __init__(self,string):
        time_upper_bound = string.index('[')
        time_lower_bound = string.index(']')
        type_upper_bound = string.index('<')
        type_lower_bound = string.index('>')
        time_string = string[time_upper_bound+1:time_lower_bound]
        start_string,end_string = time_string.split(" ")
        self.startTime = float(start_string.split(':')[0])*60+float(start_string.split(':')[1])
        self.endTime = float(end_string.split(':')[0])*60+float(end_string.split(':')[1])
        self.sentenceType = int(string[type_lower_bound-1])
        self.sentence = sentence_cleaning(string[type_lower_bound+1:])
        self.tokenized_sentences = [applystemming(t,[remove_front_mark,remove_end_mark]) for t in nltk.word_tokenize(self.sentence)]
        self.tokenized_sentences[:] = (value for value in self.tokenized_sentences if value != '') #remove all '' in tokenized_sentences
        if(len(self.tokenized_sentences)==0):
            None
        else:
            self.pos_tags = nltk.tag._pos_tag(self.tokenized_sentences, None, tagger)
            pos_tag_lem = [(element[0],get_wordnet_pos(element[1])) for element in self.pos_tags]
            self.tokenized_sentences = [wnl.lemmatize(t[0],pos=t[1]) if t[1]!='' else t[0] for t in pos_tag_lem]

    def print_info(self):
        print self.startTime,",",self.endTime,",",self.sentenceType,",",self.sentence

class LyricsInfo:
    ID = None
    sentenceInfos = []
    def __init__(self,array):
        self.ID = int(array[0])
        sentenceInfos = []
        for element in array[1][:-1]:
            sen_info = SentenceInfo(element)
            sentenceInfos.append(sen_info)


        self.sentenceInfos = sentenceInfos
    def print_lyrics(self):
        for element in self.sentenceInfos:
            element.print_info()
    def print_info(self):
        for element in self.sentenceInfos:
            element.print_info()
            print element.pos_tags
    def voc_set(self):
        voc_set = set()
        for element in self.sentenceInfos:
            voc_set = set().union(*[voc_set,element.tokenized_sentences])
        return voc_set


class LyricsData:
    lyricsinfos = []
    ids = []
    voc_dict = []
    def __init__(self,array):
        for element in array:
            li = LyricsInfo(element)
            self.lyricsinfos.append(li)
            self.ids.append(li.ID)
    # TODO: the function that can find an info of an corresponding lyrics
    def findInfobyID(self,id):
        return self.lyricsinfos[self.ids.index(id)]

    def dict_generate(self):
        if len(self.voc_dict)==0:
            voc_set = set()
            for element in self.lyricsinfos:
                voc_set = set().union(*[voc_set,element.voc_set()])
            voc_array = list(voc_set)
            voc_array.sort()
            self.voc_dict = (dict(zip(voc_array,range(len(voc_array)))),dict(zip(range(len(voc_array)),voc_array)))
            #string = ""
            #for lyrics in self.lyricsinfos:
            #    for sentenceinfo in lyrics.sentenceInfos:
            #        string = string.join(sentenceinfo.sentence)
            #        string.join(" ")

            #all_tokenized = string.split(' ')
            #voc_set = set(all_tokenized)
            #voc_array = list(voc_set)
            #voc_array.sort()
            #self.voc_dict = (dict(zip(voc_array,range(len(voc_array)))),dict(zip(range(len(voc_array)),voc_array)))
            return self.voc_dict
        else:
            return self.voc_dict
    def indexify(self,monitor=False):
        for lyrics in self.lyricsinfos:
            for sentence in lyrics.sentenceInfos:
                for i in range(len(sentence.tokenized_sentences)):
                    sentence.tokenized_sentences[i] = self.voc_dict[0][sentence.tokenized_sentences[i]]
            if monitor:
                print lyrics.ID
