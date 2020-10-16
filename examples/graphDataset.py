# -*- coding: utf-8 -*-

from Datasets.tudataset import datasets,reader

datasets.get_dataset("PROTEINS")
graphs_nx = reader.tud_to_networkx("PROTEINS")
