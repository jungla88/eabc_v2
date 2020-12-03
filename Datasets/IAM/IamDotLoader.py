#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 11:35:11 2020

@author: luca

A simple loader class containing a function that create a dictionary whose key is the graph id as integer and a tuple made of
a networkx graph loaded from a dot file and its label as value.
The function can be initilize with two arguments:
1)A string of delimiters for parsing the id and class from dot
2)A parser function that set appropriate node and edge labels structure
The loader reset the node id read from the file and convert it into an integer.
load() returns a dict {id: (nx.graph, label)}
"""


from os import listdir
from re import split, compile, escape
# import re
from networkx import nx_pydot, Graph, convert_node_labels_to_integers
import logging

LOG_FILENAME = 'log_loader.out'
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)

class DotLoader:

    def __init__(self, parserFunction, delimiters):

        self.__delimiters = delimiters
        self.__parseGraph = parserFunction

    def load(self, folder):
        graphPattern = {}
        regexPattern = '|'.join(map(escape, self.__delimiters))
        compile(regexPattern)
        for dot in listdir(folder):
            #DEBUG
                try:
                    g = convert_node_labels_to_integers(Graph(nx_pydot.read_dot(folder + dot)), 0, 'sorted', None)
                    """ User Defined Parser"""
                    self.__parseGraph(g)
                    """ """
                    
                    expr = split(regexPattern, dot)
                    graphPattern[int(expr[1])] = (g, expr[2])
                except Exception:
                    print("This object makes some errors: ", dot)
                    print("Traceback the following to",LOG_FILENAME)
                    logging.exception("Exception")
                    print("Process keeps running")

        return graphPattern
