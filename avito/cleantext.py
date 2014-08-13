#!/usr/bin/python
# coding=utf-8

import aspell
import codecs
import string
import sys
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import RussianStemmer
import re
import utils


class WhitespaceOnlyTokenizer(RegexpTokenizer):
    def __init__(self):
        RegexpTokenizer.__init__(self, '\S+|\n')


russtop = map(lambda x: x.decode('utf-8'), stopwords.words('russian'))
russtop.append('url')
russtop.append('phone')

tokenizer = WhitespaceOnlyTokenizer()
stemmer = RussianStemmer()

ruspell_dict = {}
ruspell = aspell.Speller(('lang', 'ru_RU'), ('encoding', 'utf-8'))


def convert_word(word, stem, spell):
    if spell:
        word_encode = word.encode('utf-8')
        if ruspell.check(word_encode) == 0:
            if word in ruspell_dict:
                word = ruspell_dict[word]
            else:
                suggests = ruspell.suggest(word_encode)
                if len(suggests) != 0:
                    suggest_word = suggests[0].decode('utf-8')
                    ruspell_dict[word] = suggest_word
                    word = suggest_word

    if stem:
        word = stemmer.stem(word)

    return word


def clean(txt, stem=False, spell=False):
    # print txt
    txt = txt.lower().strip()
    txt = txt.replace('^p', '')
    txt = re.sub('[%s]' % re.escape(string.punctuation + u'“”«»–—―◦℅™•№▪'), ' ', txt)
    txt = u' '.join([convert_word(i, stem, spell) for i in tokenizer.tokenize(txt) if i not in russtop])

    # print txt
    return txt


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print "Usage " + sys.argv[0] + " file cleanfile"
        sys.exit(1)

    train_file = sys.argv[1]
    clean_filename = sys.argv[2]

    prev_l = ''

    utils.reset_progress()
    with codecs.open(clean_filename, 'w', 'utf-8') as fw:
        for parts in utils.read_train(train_file):
            parts[3] = clean(parts[3], False, False)
            parts[4] = clean(parts[4], False, False)
            fw.write('\t'.join(parts))
            utils.log_progress()