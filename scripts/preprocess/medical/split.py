import os
import random

import requests

m2m_languages = {'af', 'da', 'nl', 'de', 'en', 'is', 'lb', 'no', 'sv', 'fy',
                'yi', 'ast', 'ca', 'fr', 'gl', 'it', 'oc', 'pt', 'ro', 'es',
                'be', 'bs', 'bg', 'hr', 'cs', 'mk', 'pl', 'ru', 'sr', 'sk',
                'sl', 'uk', 'et', 'fi', 'hu', 'lv', 'lt', 'sq', 'hy', 'ka',
                'el', 'br', 'ga', 'gd', 'cy', 'az', 'ba', 'kk', 'tr', 'uz',
                'ja', 'ko', 'vi', 'zh', 'bn', 'gu', 'hi', 'kn', 'mr', 'ne',
                'or', 'pa', 'sd', 'si', 'ur', 'ta', 'ceb', 'ilo', 'id', 'jv',
                'mg', 'ms', 'ml', 'su', 'tl', 'my', 'km', 'lo', 'th', 'mn',
                'ar', 'he', 'ps', 'fa', 'am', 'ff', 'ha', 'ig', 'ln', 'lg',
                'nso', 'so', 'sw', 'ss', 'tn', 'wo', 'xh', 'yo', 'zu', 'ht'}


corpus  = "EMEA"
get_lang_url ="http://opus.nlpl.eu/opusapi/?languages=True&corpus={}".format(corpus)
response = requests.get(get_lang_url)
langs = response.json()['languages']
print("List languages in {} corpus {}".format(corpus, langs))

processed = set()
num_duplicate = 0
dev_set = set()
random.seed(123)
for src_lang in langs:
  if src_lang not in m2m_languages:
    continue
  response = requests.get("https://opus.nlpl.eu/opusapi/?languages=True&corpus={}&source={}".format(corpus, src_lang))
  tgt_langs = response.json()['languages']
  for tgt_lang in tgt_langs:
    if tgt_lang not in m2m_languages:
      continue
    if tgt_lang in processed:
      continue
    cur_dev_set = set()
    data_dir = "./{}/tmp/{}-{}".format(corpus, src_lang, tgt_lang)
    id_file = "{}/{}.{}-{}.ids".format(data_dir, corpus, src_lang, tgt_lang)
    src_file = "{}/{}.{}-{}.{}".format(data_dir, corpus, src_lang, tgt_lang, src_lang)
    tgt_file = "{}/{}.{}-{}.{}".format(data_dir, corpus, src_lang, tgt_lang, tgt_lang)

    test_size = 2000
    dev_size = 2000
    ratio = 0.2

    out_dir = "./{}/processed/{}-{}".format(corpus, src_lang, tgt_lang)
    os.makedirs(out_dir, exist_ok=True)
    src_train = "{}/train.{}".format(out_dir, src_lang)
    tgt_train = "{}/train.{}".format(out_dir, tgt_lang)
    src_dev = "{}/dev.{}".format(out_dir, src_lang)
    tgt_dev = "{}/dev.{}".format(out_dir, tgt_lang)
    src_test = "{}/test.{}".format(out_dir, src_lang)
    tgt_test = "{}/test.{}".format(out_dir, tgt_lang)

    num_dev = 0
    num_train = 0
    num_test = 0

    with open(src_file, 'r') as fsrc, open(tgt_file, 'r') as ftgt, \
    open(src_train, 'w', encoding='utf-8') as fsrc_train, open(tgt_train, 'w', encoding='utf-8') as ftgt_train, \
    open(src_dev, 'w', encoding='utf-8') as fsrc_dev, open(tgt_dev, 'w', encoding='utf-8') as ftgt_dev, \
    open(src_test, 'w', encoding='utf-8') as fsrc_test, open(tgt_test, 'w', encoding='utf-8') as ftgt_test :
      for idx, (src, tgt) in enumerate(zip(fsrc, ftgt)):
        if src.strip() in cur_dev_set or tgt.strip() in cur_dev_set:
          print("Found duplicate")
          num_duplicate += 1
          continue

        if (random.random() < ratio) and num_test < test_size:
          fsrc_test.write(src)
          ftgt_test.write(tgt)
          dev_set.add(src.strip())
          dev_set.add(tgt.strip())
          cur_dev_set.add(src.strip())
          cur_dev_set.add(tgt.strip())
          num_test += 1
          continue

        if (random.random() < ratio) and num_dev < dev_size:
          fsrc_dev.write(src)
          ftgt_dev.write(tgt)
          num_dev += 1
          dev_set.add(src.strip())
          dev_set.add(tgt.strip())
          cur_dev_set.add(src.strip())
          cur_dev_set.add(tgt.strip())
        else:
          if src.strip() in dev_set or tgt.strip() in dev_set:
            print("Found duplicate")
            num_duplicate += 1
            continue
          fsrc_train.write(src)
          ftgt_train.write(tgt)
          num_train += 1

    print("Finish processing {}-{}".format(src_lang, tgt_lang))
    print("* Train size: {}".format(num_train))
    print("* Dev size: {}".format(num_dev))
    print("* Test size: {}".format(num_test))
  print("* Num duplicate: {}".format(num_duplicate))
  processed.add(src_lang)
