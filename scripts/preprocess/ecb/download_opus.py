import os
import zipfile

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


def download(url: str, dest_folder: str):
  if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)  # create folder if it does not exist

  filename = url.split('/')[-1].replace(" ", "_")  # be careful with file names
  file_path = os.path.join(dest_folder, filename)

  if os.path.exists(file_path):
    print(f"File {file_path} exist. Skip downloading")
    return file_path

  r = requests.get(url, stream=True)
  if r.ok:
    print("saving to", os.path.abspath(file_path))
    with open(file_path, 'wb') as f:
      for chunk in r.iter_content(chunk_size=1024 * 8):
        if chunk:
          f.write(chunk)
          f.flush()
          os.fsync(f.fileno())
    return file_path
  else:  # HTTP status code 4XX/5XX
    print("Download failed: status code {}\n{}".format(r.status_code, r.text))
    return None

corpus  = "ECB"
get_lang_url ="http://opus.nlpl.eu/opusapi/?languages=True&corpus={}".format(corpus)

response = requests.get(get_lang_url)
langs = response.json()['languages']
print("List languages in {} corpus {}".format(corpus, langs))

processed = set()
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
    url = "https://object.pouta.csc.fi/OPUS-ECB/v1/moses/{}-{}.txt.zip".format(src_lang, tgt_lang)
    print("Download {}".format(url))
    filename = download(url, dest_folder="./{}/orig".format(corpus))
    print("Unzip...")
    if filename is not None:
      with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall("./{}/tmp/{}-{}".format(corpus, src_lang, tgt_lang))
  processed.add(src_lang)
