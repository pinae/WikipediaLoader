#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals
import os
from random import shuffle
import numpy as np
import h5py
from encoder import encode


def read_data():
    raw_data = []
    raw_data_size = 0
    filenames = os.listdir("articles")
    shuffle(filenames)
    for filename in filenames:
        with open(os.path.join("articles", filename), "r") as article:
            chunk = encode(article.read() + "\n\n\n\n")
            raw_data.append(chunk)
            raw_data_size += len(chunk)
    print("Raw data size: " + str(raw_data_size))
    return b''.join(raw_data)


def convert_to_batches(serial_data, length, bs):
    assert serial_data.size % length == 0
    num_sequences = serial_data.size // length
    assert num_sequences % bs == 0
    num_batches = num_sequences // bs
    serial_data = serial_data.reshape((bs, num_batches * length))
    serial_data = np.vstack(np.hsplit(serial_data, num_batches)).T[:, :, None]
    return serial_data


def main():
    batch_size = 100
    # Batch size which will be used for training.
    # Needed to maintain continuity of data across batches.
    seq_len = 100
    # Number of characters in each sub-sequence.
    # Limits the number of time-steps that the gradient is back-propagated.
    num_test_chars = 5000000
    # Number of characters which will be used for testing.
    # An equal number of characters will be used for validation.

    bs_data_dir = os.environ.get('BRAINSTORM_DATA_DIR', '.')
    hdf_file = os.path.join(bs_data_dir, 'WikipediaDE.hdf5')

    print("Using data directory:", bs_data_dir)
    print("Loading data ...")
    raw_data = read_data()
    print("Done.")

    print("Preparing data for Brainstorm ...")
    data = np.fromstring(raw_data, dtype=np.uint8, count=100000000)
    # unique, data = np.unique(raw_data, return_inverse=True)

    train_data = data[: -2 * num_test_chars]
    train_targets = data[1: -2 * num_test_chars + 1]
    valid_data = data[-2 * num_test_chars: -num_test_chars]
    valid_targets = data[-2 * num_test_chars + 1: -num_test_chars + 1]
    test_data = data[-num_test_chars:]
    test_targets = np.append(data[-num_test_chars + 1:], [0])

    # Convert to batches
    train_data = convert_to_batches(train_data, seq_len, batch_size)
    train_targets = convert_to_batches(train_targets, seq_len, batch_size)

    valid_data = convert_to_batches(valid_data, seq_len, batch_size)
    valid_targets = convert_to_batches(valid_targets, seq_len, batch_size)

    test_data = convert_to_batches(test_data, seq_len, batch_size)
    test_targets = convert_to_batches(test_targets, seq_len, batch_size)
    print("Done.")

    print("Creating WikipediaDE character-level HDF5 dataset ...")
    f = h5py.File(hdf_file, 'w')
    description = """
    The Wikipedia.de dataset, prepared for character-level language modeling.

    Attributes
    ==========
    description: This description.
    unique: A 1-D array of unique characters (0-255 Jonny-Encoded values) in the
    dataset. The index of each character was used as the class ID for preparing
    the data.

    Variants
    ========
    split: Split into 'training', 'validation' and 'test' tests of size 90, 10 and
    10 million characters respectively. Each sequence is {} characters long. The
    dataset has been prepared expecting minibatches of {} sequences.
    """.format(seq_len, batch_size)
    f.attrs['description'] = description
    # f.attrs['unique'] = unique

    variant = f.create_group('split')
    group = variant.create_group('training')
    group.create_dataset(name='default', data=train_data, compression='gzip')
    group.create_dataset(name='targets', data=train_targets, compression='gzip')

    group = variant.create_group('validation')
    group.create_dataset(name='default', data=valid_data, compression='gzip')
    group.create_dataset(name='targets', data=valid_targets, compression='gzip')

    group = variant.create_group('test')
    group.create_dataset(name='default', data=test_data, compression='gzip')
    group.create_dataset(name='targets', data=test_targets, compression='gzip')

    f.close()

if __name__ == "__main__":
    main()
