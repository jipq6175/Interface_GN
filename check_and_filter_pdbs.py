

# check and filter the pdbs
import os
import matplotlib.pyplot as plt
import logging
logging.getLogger().setLevel(logging.CRITICAL)
logging.basicConfig(level="INFO")
logging.getLogger('matplotlib').setLevel(logging.CRITICAL)

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.distance import *
from functools import partial

from tqdm import tqdm


edge_funcs = {"edge_construction_functions": [add_peptide_bonds,
                                              partial(add_distance_threshold, long_interaction_threshold=5, threshold=7.0)]}
CONFIG = ProteinGraphConfig(**edge_funcs)



def contain_interfaces(pdbfile, config=CONFIG): 

    assert os.path.isfile(pdbfile)
    g = construct_graph(config=config, pdb_path=pdbfile)
    
    rlt = False
    for u, v in g.edges():
        if g.nodes[u]['chain_id'] != g.nodes[v]['chain_id']: 
            return True
    return False


if __name__ == '__main__':

    path = './data/raw_data/'
    pdbfiles = [x for x in os.listdir(path) if x.endswith('.ent')]
    
    f = open('pdb_to_remove.txt', 'w')
    for pdbfile in tqdm(pdbfiles[:], desc='Checking PDBs'): 
        # print(pdbfile)
        flag = contain_interfaces(os.path.join(path, pdbfile))

        if not flag: 
            # print(f'PDB: {pdbfile[:-4]} does not contain any interface.')
            f.write(f'{pdbfile}\n')
    f.close()

    print('Success!!')
    