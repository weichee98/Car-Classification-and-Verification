import argparse
import glob
import pandas as pd

from tqdm import tqdm


def read_file(filename):
    make_id, model_id, released_year, image_name = filename.split('/')[-4:]
    year = lambda x: int(x) if x.isdigit() else None
    with open(filename, 'r') as file:
        view_point = int(file.readline())
        file.readline()
        x1, y1, x2, y2 = list(map(int, file.readline().split()))
    return {
        "make_id": int(make_id),
        "model_id": int(model_id),
        "released_year": year(released_year),
        "image_name": image_name[:-4] + ".jpg",
        "view_point": view_point,
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2
    }


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--directory', required=True, help='Path of directory "data/label"', default='../data/label')
    parser.add_argument('-o', '--output', required=True, help='Path of output directory', default='../data')
    args = parser.parse_args()

    path = args.directory + "/*/*/*/*.txt"
    with tqdm(glob.glob(path)) as pbar:
        df = dict()
        for i, filename in enumerate(pbar):
            df[i] = read_file(filename)
        df = pd.DataFrame(df).T

    output_path = args.output + '/data.csv'
    df.to_csv(output_path, header=True, index=False)
