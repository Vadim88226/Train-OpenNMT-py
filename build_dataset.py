import os
import deepcut
from nltk import tokenize, word_tokenize
from sklearn.model_selection import train_test_split

en_sentences = []
th_sentences = []
with open("EN_TH_data/EN_TH_ML.txt", encoding="utf-8") as file:
    for line in file:
        s = line.split('|')
        en_s = s[0].rstrip()
        th_s = s[1].rstrip()
        en_sentences.append(" ".join(word_tokenize(en_s)))
        th_sentences.append(" ".join([token  for token in deepcut.tokenize(th_s) if token.strip()]))
with open("EN_TH_data/EN_dataset.txt", "w", encoding="utf-8") as en_file:
    en_file.writelines("%s\n" % s for s in en_sentences)
    en_file.close()

with open("EN_TH_data/TH_dataset.txt", "w", encoding="utf-8") as th_file:
    th_file.writelines("%s\n" % s for s in th_sentences)
    th_file.close()


en_train, en_test, th_train, th_test = train_test_split(en_sentences, th_sentences, test_size=0.1)
en_train, en_val, th_train, th_val = train_test_split(en_train, th_train, test_size=0.2)

with open("EN_TH_data/src_train.txt", "w", encoding="utf-8") as en_file:
    en_file.writelines("%s\n" % s for s in en_train)
    en_file.close()

with open("EN_TH_data/src_test.txt", "w", encoding="utf-8") as en_file:
    en_file.writelines("%s\n" % s for s in en_test)
    en_file.close()

with open("EN_TH_data/src_val.txt", "w", encoding="utf-8") as en_file:
    en_file.writelines("%s\n" % s for s in en_val)
    en_file.close()

with open("EN_TH_data/tgt_train.txt", "w", encoding="utf-8") as th_file:
    th_file.writelines("%s\n" % s for s in th_train)
    th_file.close()

with open("EN_TH_data/tgt_test.txt", "w", encoding="utf-8") as th_file:
    th_file.writelines("%s\n" % s for s in th_test)
    th_file.close()

with open("EN_TH_data/tgt_val.txt", "w", encoding="utf-8") as th_file:
    th_file.writelines("%s\n" % s for s in th_val)
    th_file.close()