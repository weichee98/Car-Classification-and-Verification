import argparse
import pandas as pd
import math


def split(df, test_ratio, valid_ratio):
    train_ratio = 1 - test_ratio - valid_ratio
    test_idx = list()
    valid_idx = list()
    train_idx = list()
    for g, gdf in df.groupby(['model_id', 'view_point']):
        n = len(gdf)
        idx = gdf.index
        train_n = math.ceil(train_ratio * n)
        test_n = int(round(train_n + test_ratio * n))
        train_idx += list(idx[:train_n])
        test_idx += list(idx[train_n:test_n])
        valid_idx += list(idx[test_n:])
    return df.iloc[train_idx], df.iloc[valid_idx], df.iloc[test_idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--data', required=True, help='Path of whole dataset', default='../data/data.csv')
    parser.add_argument('-t', '--test_ratio', required=True, help='Test ratio', default=0.3)
    parser.add_argument('-v', '--valid_ratio', required=True, help='Valid ratio', default=0)
    parser.add_argument('-train', '--train_path', required=True, help='Train dataset path', default='../data/train.csv')
    parser.add_argument('-valid', '--valid_path', required=True, help='Valid dataset path', default=None)
    parser.add_argument('-test', '--test_path', required=True, help='Test dataset path', default='../data/test.csv')
    parser.add_argument('-r', '--result', help='Split result analysis', default='../data/result.csv')
    args = parser.parse_args()

    test_ratio = float(args.test_ratio)
    valid_ratio = float(args.valid_ratio)

    df = pd.read_csv(args.data)
    train_df, valid_df, test_df = split(df, test_ratio, valid_ratio)

    if args.train_path:
        train_df.to_csv(args.train_path, index=False, header=True)
    if args.valid_path:
        valid_df.to_csv(args.valid_path, index=False, header=True)
    if args.test_path:
        test_df.to_csv(args.test_path, index=False, header=True)

    if args.result:
        split_result = pd.concat([
                train_df.groupby(['make_id', 'model_id', 'view_point']).size(), 
                valid_df.groupby(['make_id', 'model_id', 'view_point']).size(), 
                test_df.groupby(['make_id', 'model_id', 'view_point']).size()
            ], axis=1
        ).fillna(0).astype(int)
        split_result.columns = ['train', 'valid', 'test']
        split_result.to_csv(args.result, header=True, index=True)

    # python src/train_test_split.py -d data/data.csv -t 0.3 -v 0.2 -train data/train.csv -valid data/valid.csv -test data/test.csv -r data/result.csv