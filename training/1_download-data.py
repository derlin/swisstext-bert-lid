#!/usr/bin/env python3

"""
Download Leipzig data for all languages classes except GSW.
By default, the text files will be downloaded in ./leipzig.

Note: those files were available at the time of the SwissText publication,
I cannot ensure it will still be in the future.
"""

import requests
import tarfile
from io import BytesIO
import os

_sources_map = {
    'afr': 'afr_web_2013_300K.txt', 'bar': 'bar_wikipedia_2016_30K.txt',
    'cat': 'cat_wikipedia_2016_100K.txt', 'dan': 'dan_wikipedia_2016_100K.txt',
    'deu': 'deu-ch_web_2002_100K.txt', 'eng': 'eng-eu_web_2015_1M.txt',
    'epo': 'epo_wikipedia_2016_100K.txt', 'est': 'est_newscrawl_2016_100K.txt',
    'fin': 'fin_wikipedia_2016_100K.txt', 'fra': 'fra_wikipedia_2010_100K.txt',
    'frr': 'frr_wikipedia_2016_10K.txt', 'gle': 'gle_newscrawl_2014_10K.txt',
    'glg': 'glg_wikipedia_2016_100K.txt', 'hrv': 'hrv_wikipedia_2016_100K.txt',
    'isl': 'isl_wikipedia_2016_100K.txt', 'ita': 'ita_wikipedia_2016_100K.txt',
    'jav': 'jav_wikipedia_2016_100K.txt', 'knn': 'knn-in_web_2015_10K.txt',
    'ksh': 'ksh_wikipedia_2016_10K.txt', 'lim': 'lim_wikipedia_2016_100K.txt',
    'ltz': 'ltz-lu_web_2015_300K.txt', 'nds': 'nds_wikipedia_2016_100K.txt',
    'nld': 'nld_mixed_2012_300K.txt', 'pap': 'pap_newscrawl_2016_10K.txt',
    'pfl': 'pfl_wikipedia_2016_10K.txt', 'por': 'por_wikipedia_2016_100K.txt',
    'ron': 'ron_wikipedia_2011_100K.txt', 'slv': 'slv_wikipedia_2016_100K.txt',
    'spa': 'spa_wikipedia_2016_100K.txt', 'swa': 'swa_wikipedia_2016_100K.txt',
    'swe': 'swe_wikipedia_2016_100K.txt'
}


class LeipzigDownloader:
    DOWNLOAD_URL = 'http://pcai056.informatik.uni-leipzig.de/downloads/corpora'

    @classmethod
    def get_url(cls, corpora_name):
        return f'{cls.DOWNLOAD_URL}/{corpora_name}'

    @classmethod
    def download_sentences(cls, url):
        """Download sentences from a leipzig resource URL."""
        # get tar archive
        res = requests.get(url)
        # extract sentences file from archive
        tar = tarfile.open(mode='r:gz', fileobj=BytesIO(res.content))
        tar_info = [member for member in tar.getmembers() if member.name.endswith('sentences.txt')][0]
        handle = tar.extractfile(tar_info)
        # read sentence file
        raw_text = handle.read().decode('utf-8')
        return [line.split('\t')[1] for line in raw_text.split('\n') if '\t' in line]


def check_resources_availability():
    print('Trying each resource url...')
    for lang, target_file in _sources_map.items():
        corpora_name = target_file.replace('.txt', '.tar.gz')
        r = requests.head(LeipzigDownloader.get_url(corpora_name))
        if r.status_code != 200:
            print(f'{lang}\t{corpora_name}\t{r.status_code}:{r.text[:50]}')


def download(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for lang, target_file in _sources_map.items():
        url = LeipzigDownloader.get_url(target_file.replace('.txt', '.tar.gz'))
        print(f'Downloading {lang} from {url}...', end=' ', flush=True)
        try:
            with open(os.path.join(output_dir, target_file), 'w') as f:
                f.write('\n'.join([
                    l for l in LeipzigDownloader.download_sentences(url)
                    if len(l) > 0 and not l.isspace()
                ]))
                print('OK.')
        except Exception as e:
            print(f'Error! {e}')


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['check', 'download'], default='check')
    parser.add_argument('-o', '--output-dir', default='./leipzig', help='Where to download the files.')
    args = parser.parse_args()

    if args.action == 'download':
        download(args.output_dir)
    else:
        check_resources_availability()