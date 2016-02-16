#!/usr/bin/python3
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals
import brainstorm as bs
import numpy as np
import h5py
from time import sleep
from encoder import encode, decode, character_count
import os


def get_char(output):
    # Take network output and return predicted character and its index
    cumsum = np.cumsum(output.squeeze())
    sample = np.random.rand()
    idx = np.argmax(cumsum > sample)
    return decode(bytes([idx])), idx


data_dir = os.environ.get('BRAINSTORM_DATA_DIR', '../data')
data_file = os.path.join(data_dir, 'WikipediaDE.hdf5')
with h5py.File(data_file, 'r') as f:
    eye = np.eye(character_count)
network = bs.Network.from_hdf5('WikipediaDE_best.hdf5')
start = 'Auf der Konsole starten Programme '
array = np.array([int(encode(char)[0]) for char in start], dtype=np.uint8).reshape((len(start), 1, 1))
context = None
network._buffer_manager.clear_context()

# np.random.seed(1337)

text = []
i = 0
print(start)
while True:
    data = eye[array].reshape((array.shape[0], array.shape[1], character_count))
    network.provide_external_data({'default': data}, all_inputs=False)
    if i == 0:
        network.forward_pass()
    else:
        network.forward_pass(context=context)
    context = network.get_context()
    output = network.get(network.output_name)
    char, next_input = get_char(output)
    text.append(char)
    array = next_input.reshape((1, 1, 1))
    if (len(text) > 70 and ''.join([char]) == " ") or len(text) >= 99:
        print(''.join(text))
        text = []
    line = ''.join(text)
    if "\n" in line:
        print(''.join(text))
        text = []
    else:
        print(''.join(text), end="\r", flush=True)
    i += 1
    sleep(0.01)
