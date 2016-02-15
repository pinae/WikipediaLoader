#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, unicode_literals

from encoder import character_count

import os

import h5py

import brainstorm as bs
from brainstorm.data_iterators import OneHot, Minibatches
#from brainstorm.handlers import PyCudaHandler

bs.global_rnd.set_seed(42)


# ---------------------------- Set up Iterators ----------------------------- #

data_dir = os.environ.get('BRAINSTORM_DATA_DIR', '../data')
data_file = os.path.join(data_dir, 'WikipediaDE.hdf5')
ds = h5py.File(data_file, 'r')['split']
x_tr, y_tr = ds['training']['default'][:], ds['training']['targets'][:]
x_va, y_va = ds['validation']['default'][:], ds['validation']['targets'][:]

getter_tr = OneHot(Minibatches(100, default=x_tr, targets=y_tr, shuffle=False),
                   {'default': character_count})
getter_va = OneHot(Minibatches(100, default=x_va, targets=y_va, shuffle=False),
                   {'default': character_count})

# ----------------------------- Set up Network ------------------------------ #

network = bs.tools.create_net_from_spec('classification', character_count, character_count,
                                        'L1000 L800 L500')
# network = bs.Network.from_hdf5('WikipediaDE_best.hdf5')

# Uncomment next line to use the GPU
# network.set_handler(PyCudaHandler())
network.initialize({"default": bs.initializers.Gaussian(0.1), "Lstm*": {"bf": 1}})

# ----------------------------- Set up Trainer ------------------------------ #

trainer = bs.Trainer(bs.training.MomentumStepper(learning_rate=0.01,
                                                 momentum=0.9))
trainer.add_hook(bs.hooks.ProgressBar())
scorers = [bs.scorers.Accuracy(out_name='Output.outputs.predictions')]
trainer.add_hook(bs.hooks.MonitorScores('valid_getter', scorers,
                                        name='validation', interval=3000,
                                        timescale='update'))
trainer.add_hook(bs.hooks.SaveBestNetwork('validation.total_loss',
                                          filename='WikipediaDE_best.hdf5',
                                          name='best weights',
                                          criterion='min'))
trainer.add_hook(bs.hooks.StopAfterEpoch(500))

# -------------------------------- Train ------------------------------------ #

trainer.train(network, getter_tr, valid_getter=getter_va)
print("Best validation set loss:", max(trainer.logs["validation"]["total_loss"]))
