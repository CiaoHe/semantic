import sys,re,collections,nltk
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# patterns that used to find or/and replace particular chars or words
# to find chars that are not a letter, a blank or a quotation
# pat_letter = re.compile(r'[^a-zA-Z \']+') #去掉非字母的所有词
# to find the 's following the pronouns. re.I is refers to ignore case
pat_is = re.compile("(it|he|she|that|this|there|here)(\'s)", re.I)
# to find the 's following the letters
pat_s = re.compile("(?<=[a-zA-Z])\'s")
# to find the ' following the words ending by s
pat_s2 = re.compile("(?<=s)\'s?")
# to find the abbreviation of not
pat_not = re.compile("(?<=[a-zA-Z])n\'t")
# to find the abbreviation of would
pat_would = re.compile("(?<=[a-zA-Z])\'d")
# to find the abbreviation of will
pat_will = re.compile("(?<=[a-zA-Z])\'ll")
# to find the abbreviation of am
pat_am = re.compile("(?<=[I|i])\'m")
# to find the abbreviation of are
pat_are = re.compile("(?<=[a-zA-Z])\'re")
# to find the abbreviation of have
pat_ve = re.compile("(?<=[a-zA-Z])\'ve")


lmtzr = WordNetLemmatizer()

# 判断是否是标点
def is_biaodian(stri):
    bd = [',', '.', '\'', '\"', '/', '[', ']', '(', ')', '!', ':', ';']
    if stri in bd:
        return False
    return True

# 把句子中的/和-的词语分开
def took_spe_word(line):
    new_line = ''
    for w in line.split():
        if '/' in w:
            new_line += w.split('/')[0] + ' ' + w.split('/')[1] + ' '
        elif '-' in w:
            new_line += w.split('-')[0] + ' ' + w.split('-')[1] + ' ' 
        else:
            new_line += w + ' '
    new_line = new_line.strip()
    return new_line

# count word frequency
def get_words(file):  
    with open (file) as f:  
        words_box=[]
        pat = re.compile(r'[^a-zA-Z \']+')
        for line in f:                           
            #if re.match(r'[a-zA-Z]*',line): 
            #    words_box.extend(line.strip().strip('\'\"\.,').lower().split())
            # words_box.extend(pat.sub(' ', line).strip().lower().split())
            line = replace_abbreviations(line)
            words = word_tokenize(line)
            words_box.extend(merge(words))
    return collections.Counter(words_box)  

# lemmatize words
def merge(words):
    new_words = []
    for word in words:
        # if word and word not in stopwords.words('english'): # remove stopwords
        if word:  # keep stopwords
            tag = nltk.pos_tag(word_tokenize(word)) # tag is like [('bigger', 'JJR')]
            pos = get_wordnet_pos(tag[0][1])
            if pos:
                lemmatized_word = lmtzr.lemmatize(word, pos)
                new_words.append(lemmatized_word)
            else:
                new_words.append(word)
    return new_words


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

# replace the abbreviations
def replace_abbreviations(text):
    new_text = text
    # new_text = pat_letter.sub(' ', text).strip().lower()
    new_text = pat_is.sub(r"\1 is", new_text)
    new_text = pat_s.sub("", new_text)
    new_text = pat_s2.sub("", new_text)
    new_text = pat_not.sub(" not", new_text)
    new_text = pat_would.sub(" would", new_text)
    new_text = pat_will.sub(" will", new_text)
    new_text = pat_am.sub(" am", new_text)
    new_text = pat_are.sub(" are", new_text)
    new_text = pat_ve.sub(" have", new_text)
    new_text = new_text.replace('\'', ' ')
    return new_text

def append_ext(words):
    new_words = []
    for item in words:
        word, count = item
        tag = nltk.pos_tag(word_tokenize(word))[0][1] # tag is like [('bigger', 'JJR')]
        new_words.append((word, count, tag))
    return new_words

def write_to_file(words, file='1.txt'):
    f = open(file, 'w')
    for item in words:
        for field in item:
            f.write(str(field)+',')
        f.write('\n')

#token, lemmatize and remove punctuation
def process(infile, outfile):
    f = open(infile)
    f1 = open(outfile, 'a+')

    pat = re.compile(r'[^a-zA-Z \']+')
    for line in f:                           
        line = replace_abbreviations(line) # replace abbreviation
        new_line = took_spe_word(line) # token '-' and '/'
        words = word_tokenize(new_line)
        firter_words = list(filter(is_biaodian, words)) # remove punctuation
        li = merge(firter_words)
        s1 = ''
        s2 = ''
        is_another  = False
        for w in li:
            if not is_another:
                s1 += w + ' '
            else:
                s2 += w + ' '
            if w == '\t':
                is_another = True
        f1.write(s1.strip()  + '\t' + s2.strip() + '\n')
    f1.close()
    f.close()
