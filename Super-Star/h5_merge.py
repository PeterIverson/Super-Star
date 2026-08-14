import os
import h5py
import argparse

def merge_h5_files(input_dir, output_h5_path):
    data_dict = {}
    # iterate over all .h5 files in the directory
    for fname in os.listdir(input_dir):
        print(fname)
        if fname.endswith('.h5'):
            fpath = os.path.join(input_dir, fname)
            key = os.path.splitext(fname)[0]
            with h5py.File(fpath, 'r') as f:
                # assume each h5 file contains only one dataset, take the data of the first key
                keys = list(f.keys())
                if len(keys) == 1:
                    data = f[keys[0]][:]
                    data_dict[key] = data
                else:
                    # if there are multiple datasets, load all of them
                    data_dict[key] = {k: f[k][:] for k in keys}

    # save as a new h5 file
    with h5py.File(output_h5_path, 'w') as out_f:
        for k, v in data_dict.items():
            if isinstance(v, dict):
                g = out_f.create_group(k)
                for subk, subv in v.items():
                    g.create_dataset(subk, data=subv)
            else:
                out_f.create_dataset(k, data=v)


if __name__ == '__main__':
    # region Args Parser
    print("start merging h5 files...")
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='./Data/sg_processed/h5')
    parser.add_argument('--save_h5', type=str, default='./Data/sg_processed/merged.h5')

    args = parser.parse_args()

    # endregion

    merge_h5_files(args.data_dir, args.save_h5)
