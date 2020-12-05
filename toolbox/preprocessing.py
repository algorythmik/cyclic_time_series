import glob
import os

import pandas as pd
import tqdm
import argparse

data_dir = os.environ.get('DATA_DIR')
ok_nok_label = {'OK': 'Negative', 'NOK': 'Positive'}
column_names = ['time', 'position', 'force', 'position_2']


def read_files(dataset):
    file_paths = glob.glob(
        os.path.join(data_dir, 'raw', dataset, '**', '*.csv'), recursive=True)
    for file_path in tqdm.tqdm(file_paths):
        if 'data_description' in file_path:
            continue
        if 'Neuer' in file_path:
            continue

        df = pd.read_csv(
            file_path, sep=';',
            skiprows=279,
            skipfooter=4,
            names=column_names,
            decimal=',')
        group, nest, file_name = file_path.split(os.sep)[-3:]
        run_id, prod_algo_label = os.path.splitext(file_name)[0].split('__')

        yield df, {
            'file_path': file_path,
            'group': group,
            'nest': nest,
            'file_name': file_name,
            'run_id': run_id,
            'prod_algo_label': ok_nok_label[prod_algo_label],
        }


def process_dataset_20191031():
    dataset = '20191031'
    dfs = []
    prod_algo = []
    data_desc = pd.read_csv(os.path.join(
        data_dir, 'raw', dataset, 'data_description.csv'),
        sep=';',
        index_col='dataset')

    for df, meta_data in tqdm.tqdm(read_files(dataset)):
        nest = meta_data['nest']
        group = meta_data['group']
        df['group'] = group
        df['operation_mode'] = data_desc.loc[group, 'operation_mode']
        df['run_id'] = meta_data['run_id']
        df['nest'] = nest
        df['truth'] = data_desc.loc[group, 'true_label']
        df['color'] = data_desc.loc[group, 'color']
        df['ppu'] = data_desc.loc[group, f'{nest}_PPU_settings']
        prod_algo.append(
            {'run_id': meta_data['run_id'],
             'label': meta_data['prod_algo_label']})
        dfs.append(df)

    return pd.concat(dfs), pd.DataFrame(prod_algo)


def process_dataset_201903():
    # TODO: add a data_description.csv
    # so we can use the same function
    dataset = '201903'
    date_label = {
        '2019-03-13': 1,
        '2019-03-14': 1,
        '2019-03-21': 1,
        '2019-03-22': 0,
        '2019-04-26': 1,
    }

    dfs = []
    prod_algo = []

    for df, meta_data in tqdm.tqdm(read_files(dataset)):
        dfs.append(df)
        df['run_id'] = meta_data['run_id']
        df['truth'] = date_label[meta_data['run_id'].split('_')[3]]
        prod_algo.append(
            {'run_id': meta_data['run_id'],
             'label': meta_data['prod_algo_label']})

    return pd.concat(dfs), pd.DataFrame(prod_algo)


def preprocess(dataset):
    if dataset == '20191031':
        dfs, prod_algo = process_dataset_20191031()
    elif dataset == '201903':
        dfs, prod_algo = process_dataset_201903()

    dest_dir = os.path.join(data_dir, 'dataframes', dataset)
    os.makedirs(dest_dir, exist_ok=True)
    dfs.to_pickle(os.path.join(dest_dir, 'df.pkl'))
    prod_algo.to_pickle(os.path.join(dest_dir, 'prod_algo.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Preprocessing')
    parser.add_argument(
        'dataset',
        type=str,
        choices={'20191031', '201903'},
        nargs='+')
    args = vars(parser.parse_args())
    for dataset in args['dataset']:
        preprocess(dataset)
