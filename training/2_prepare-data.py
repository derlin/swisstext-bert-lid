#!/usr/bin/env python3

import pandas as pd
from glob import glob
import os


# ===== utils

def _find_and_read_file_for_lang(data_dir, lang) -> pd.DataFrame:
    pattern = os.path.join(data_dir, f'{lang}*.txt')
    matches = glob(pattern)
    if len(matches) != 1:
        raise Exception(f'Missing Leipzig file for lang={lang}: '
                        f'expected one file matching {pattern}. Found {len(matches)}.')

    with open(matches[0]) as f:
        print(f'Loading {lang} from {matches[0]}')
        return pd.DataFrame(
            data=[l.strip() for l in f if len(l) and not l.isspace()],
            columns=['text'])


def _sample(df, num_samples):
    if num_samples > 0 and len(df) > num_samples:
        return df.sample(num_samples, random_state=1)
    return df


def _sample_balanced(df, target_size, col='lang'):
    res = pd.DataFrame()
    remainder = target_size
    per_class = df[col].value_counts().sort_values()

    for i, (cls, cnt) in enumerate(zip(per_class.index, per_class)):
        sample = _sample(df[df[col] == cls], remainder // (len(per_class) - i))
        res = res.append(sample)
        remainder -= len(sample)
    return res


# == loading / sampling data

def sample_file(data_dir, lang, num_samples=-1):
    df = _find_and_read_file_for_lang(data_dir, lang)
    df['lang'], df['label'] = lang, lang
    return _sample(df, num_samples)


def sample_mix(data_dir, langs, label, num_samples=-1):
    results = pd.DataFrame()
    for lang in langs:
        try:
            df = _find_and_read_file_for_lang(data_dir, lang)
            df['lang'], df['label'] = lang, label
            results = results.append(df)
        except Exception as e:
            print(e)

    return _sample_balanced(results, num_samples)


# == actual data preparation

def prepare_data(data_dir, gsw_filepath):
    try:
        gsw = pd.read_csv(gsw_filepath)[['text']]
    except:
        print(f'Error loading GSW data from {gsw_filepath}. Should be a valid CSV file with a "text" column.')
        exit(1)

    gsw['lang'], gsw['label'] = 'gsw', 'gsw'
    num_samples = len(gsw)

    big_langs = []
    for lang in ['deu', 'eng', 'nld', 'afr', 'ltz']:  # German, English, Dutch, Afrikaans, Luxembourgish
        big_langs.append(sample_file(data_dir, lang, num_samples))

    gsw_like = sample_mix(data_dir, [
        'bar',  # Bavarian
        'frr',  # Northern Frisian
        'ksh',  # Kölsch
        'lim',  # Limburgan
        'nds',  # Low German
        'pfl',  # Pfaelzisch
    ], 'gsw_like', num_samples)

    other = sample_mix(data_dir, [
        'cat',  # Catalan
        'hrv',  # Croatian
        'dan',  # Danish
        'epo',  # Esperanto
        'est',  # Estonian
        'fin',  # Finnish
        'fra',  # French
        'gle',  # Irish
        'glg',  # Galician
        'isl',  # Icelandic
        'ita',  # Italian
        'jav',  # Javanese
        'knn',  # Konkani
        'pap',  # Papiamento
        'por',  # Portuguese
        'ron',  # Romanian
        'slv',  # Slovenian
        'spa',  # Spanish
        'swa',  # Swahili
        'swe',  # Swedish
    ], 'other', num_samples)

    return pd.concat(big_langs + [gsw_like, other])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--gsw', required=True,
                        help='Path to a CSV file containing GSW sentences ("text" column)')
    parser.add_argument('-l', '--leipzig-dir', default='./leipzig',
                        help='Root directory of the Leipzig data. See script 0.')
    parser.add_argument('-o', '--output', default='./data',
                        help='Directory where to write the output.')
    args = parser.parse_args()

    df = prepare_data(data_dir=args.leipzig_dir, gsw_filepath=args.gsw)
    assert df.isna().sum().sum() == 0, 'Found some NaN values !'
    assert (df.text == '').sum() == 0, 'Found some empty text entries !'
    
    print(f'Writing output to {args.output}...')
    os.makedirs(args.output, exist_ok=True)
    df.to_csv(os.path.join(args.output, 'train.csv'))
    labels = sorted(df.label.unique())
    with open(os.path.join(args.output, 'environ.txt'), 'w') as f:
        f.write('export BERT_LABELS="{}"'.format(','.join(labels)))

    print('Done.')
    print('\nLangs\n=====\n')
    print(df.groupby('lang')[['text']].count())
    print('\nLabels\n=======\n')
    print(df.groupby('label')[['text']].count())