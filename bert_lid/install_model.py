#!/usr/bin/env python3
import os
import shutil

# model installation

def install_model(model_location, model_name='default', overwrite=False):
    # verify files are there
    if not check_model_files(model_location):
        raise Exception('Some mandatory model files are missing. Cannot install.')

    # get location of the bert_lid module
    bert_lid_location = os.path.dirname(__file__)
    print(f'Found module in {bert_lid_location}. Installing model...')
    target_dir = os.path.join(bert_lid_location, 'models', model_name)
    if os.path.exists(target_dir):
        if overwrite:
            shutil.rmtree(target_dir)
        else:
            raise Exception(f'Model directory {target_dir} already exists and "override" is set to False.')

    shutil.copytree(model_location, target_dir)

def check_model_files(model_location):
    ok = True

    if not os.path.isfile(os.path.join(model_location, 'environ.txt')):
        print('Missing environ.txt')
        ok = False

    bert_dir = os.path.join(model_location, 'bert')
    if not os.path.isdir(bert_dir):
        print('Missing bert directory')
        ok = False
    else:
        for mandatory_file in ['pytorch_model.bin', 'vocab.txt', 'config.json']:
            if not os.path.isfile(os.path.join(bert_dir, mandatory_file)):
                print(f'Missing {mandatory_file} in bert directory')
                ok = False
    return ok


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--bert-out-dir', required=True,
        help='Path to a directory containing the "bert" folder with the finetuned bert model and the environ.txt')
    parser.add_argument(
        '-o', '--model-name', default='default',
        help='Name of the model, which will be the name of the folder to create inside bert_lid.')
    parser.add_argument(
        '--overwrite', action='store_true', default=False,
        help='If set, overwrite the previous model installed (if any).')
    args = parser.parse_args()

    try:
        install_model(args.bert_out_dir, args.model_name, args.overwrite)
        print('Done.')
    except Exception as e:
        print('!!!! ERROR', e)
        exit(1)
