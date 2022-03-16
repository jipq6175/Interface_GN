# prepare raw data

import os
import logging
logging.basicConfig(level="INFO")

from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.distance import *
from functools import partial

import shutil
from tqdm import tqdm


if __name__ == '__main__':

    # new_edge_funcs = {"edge_construction_functions": [partial(add_distance_threshold, long_interaction_threshold=5, threshold=10.)]}
    config = ProteinGraphConfig()

    datapath = '/Users/yen/pdb'
    target_path = '/Users/yen/Interface_GN/data/raw_data/'
    folders = os.listdir(datapath)
    folders.sort()
    n = len(folders)

    for i, folder in tqdm(enumerate(folders), desc='Analyzing PDBs'): 
        
        tmppath = os.path.join(datapath, folder)
        if not os.path.isdir(tmppath): continue
        
        for gzfile in os.listdir(tmppath): 
            if not gzfile.endswith('.ent.gz'): continue
            
            full_gzfile = os.path.join(tmppath, gzfile)
            
            # extract
            os.system(f'gzip -dk {full_gzfile}')
            
            try: 
                g = construct_graph(config=config, pdb_path=full_gzfile[:-3])

                # check 2 chains: 
                chains = set([g.nodes[node]['chain_id'] for node in g.nodes()])
                if len(chains) > 1: 
                    print(f'{i+1}/{n} folder {folder}: PDB: {os.path.basename(full_gzfile[:-3])} moved to raw_data .. ')
                    shutil.move(full_gzfile[:-3], os.path.join(target_path, os.path.basename(full_gzfile[:-3])))
                
                else:
                    print(f'{i+1}/{n} folder {folder}: PDB: {os.path.basename(full_gzfile[:-3])} has only 1 chain .. ')
                    os.remove(full_gzfile[:-3])
            
            except: 
                print(f'{i+1}/{n} folder {folder}: PDB: {os.path.basename(full_gzfile[:-3])} is not a protein ..')
                os.remove(full_gzfile[:-3])
