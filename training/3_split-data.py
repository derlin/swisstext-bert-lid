#!/usr/bin/env python3

import os

import pandas as pd
from sklearn.model_selection import train_test_split

def create_splits(df, test_ratio, dev_ratio):
    devtest_ratio = test_ratio+dev_ratio
    train, devtest = train_test_split(df, test_size=devtest_ratio, random_state=45678900)
    dev, test = train_test_split(devtest, test_size=test_ratio/devtest_ratio, random_state=45678900)
    print(f'Splits: train={len(train)}, dev={len(dev)}, test={len(test)}')
    return train, dev, test


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--data', default='data.csv',
                        help='Path to a CSV file containing the data (text+label+lang columns)')
    parser.add_argument('-o', '--output', default='./data',
                        help='Directory where to write the output.')
    parser.add_argument('-ts', '--test-split', type=float, default=0.1,
                        help='Percentage of samples in the test split.')
    parser.add_argument('-ds', '--dev-split', type=float, default=0.05,
                        help='Percentage of samples in the dev/validation split.')
    args = parser.parse_args()

    assert 0 <= args.test_split + args.dev_split < 1, 'Wrong splits: expecting 0 <= test+dev < 1'

    try:
        df = pd.read_csv(args.data)
        for required_column in ['text', 'lang', 'label']:
            assert required_column in df, f'Missing {required_column} column in data file'
    except:
        print('Data file not found:', args.data)
        exit(1)

    train, dev, test = create_splits(df[['text', 'lang', 'label']], 
        test_ratio=args.test_split, dev_ratio=args.dev_split)
    
    print(f'Writing output to "{args.output}"...')
    os.makedirs(args.output, exist_ok=True)
    for name, df in zip(['train', 'dev', 'test'], [train, dev, test]):
        df.to_csv(os.path.join(args.output, f'{name}.csv'), index=False)

    labels = sorted(df.label.unique())
    with open(os.path.join(args.output, 'environ.txt'), 'w') as f:
        f.write('export BERT_LABELS="{}"'.format(','.join(labels)))