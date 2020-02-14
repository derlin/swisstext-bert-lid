#!/usr/bin/env python3

from bert_lid import BertLid
from sklearn.metrics import classification_report
import pandas as pd
import os


def print_stats(y_true, y_pred, labels):
    print(classification_report(y_true, y_pred, target_names=labels, digits=4))


def print_cm(y_true, y_pred):
    cm = pd.crosstab(y_true, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True, normalize='index')
    print((cm*100).round(3))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--data-dir', default='./data',
        help='Path to a directory containing the train, dev and test CSV files.')
    parser.add_argument(
        '-m', '--bert-out-dir', default='./out',
        help='Path to a directory containing the "bert" folder with the finetuned bert model and the environ.txt')
    args = parser.parse_args()

    lid = BertLid(model_dir=args.bert_out_dir)

    for split in ['dev', 'test']:
        print(f'Predicting {split}.csv ...')
        df = pd.read_csv(os.path.join(args.data_dir, f'{split}.csv'))
        df['pred'], df['proba'] = lid.predict_lang(df.text.values)

        print(f'\nSTATISTICS {split}\n')
        print_stats(df.label, df.pred, lid.labels)

        print(f'\nCM {split}\n')
        print_cm(df.label, df.pred)

        #df.to_csv(f'/tmp/{split}.csv', index=False)