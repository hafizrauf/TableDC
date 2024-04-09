'''
This script is used to generate random walks starting from a given edgelist without the overhead required when running
the full algorithm. The parameters used here are the same as what is used in the main algorithm, so please refer to the
readme for more details.

@author: riccardo cappuzzo

'''

'''
try:
    from embdi_master.EmbDI.utils import *
except:
    from EmbDI.utils import *
else:
    pass
'''

from embdi_master.EmbDI.utils import *
from embdi_master.EmbDI.graph import graph_generation
from embdi_master.EmbDI.sentence_generation_strategies import random_walks_generation  

def main_walks(input_file, output_file):

    # Default parameters
    configuration = {
        'walks_strategy': 'basic',
        'flatten': 'all',
        'n_sentences': 'default',
        'sentence_length': 10,
        'write_walks': True,
        'intersection': False,
        'backtrack': True,
        'repl_numbers': False,
        'repl_strings': False,
        'follow_replacement': False,
        'mlflow': False
    }

    configuration['input_file'] = input_file
    configuration['output_file'] = output_file


    prefixes, edgelist = read_edgelist(configuration['input_file'])

    graph = graph_generation(configuration, edgelist, prefixes, dictionary=None)
    if configuration['n_sentences'] == 'default':
        #  Compute the number of sentences according to the rule of thumb.
        configuration['n_sentences'] = graph.compute_n_sentences(int(configuration['sentence_length']))
    walks = random_walks_generation(configuration, graph)

