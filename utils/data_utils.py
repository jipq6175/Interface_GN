# utilities for handling the data

import os
import copy
import torch
from tqdm.auto import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.utils import add_self_loops


import logging
logging.getLogger('matplotlib').setLevel(level=logging.CRITICAL)
logging.getLogger('graphein').setLevel(level=logging.CRITICAL)


from graphein.protein.config import ProteinGraphConfig
from graphein.protein.graphs import construct_graph
from graphein.protein.edges.distance import *
from graphein.ml.conversion import GraphFormatConvertor
from graphein.protein.utils import three_to_one_with_mods
from functools import partial


# global parameters
AA = 'ACDEFGHIKLMNPQRSTVWY'
AA_ONE_HOT = {a:i for i, a in enumerate(AA)}
edge_funcs = {"edge_construction_functions": [add_peptide_bonds,
                                              add_hydrogen_bond_interactions,
                                              add_disulfide_interactions,
                                              add_ionic_interactions,
                                              add_aromatic_sulphur_interactions,
                                              add_cation_pi_interactions,
                                              partial(add_distance_threshold, long_interaction_threshold=5, threshold=8.0)]}
CONFIG = ProteinGraphConfig(**edge_funcs)



# compute the distance of edge
def tag_edge_distance(g): 
    G = copy.deepcopy(g)
    for u, v in G.edges(): 
        d = np.sqrt(((G.nodes[u]['coords'] - G.nodes[v]['coords']) ** 2).sum())
        G.edges[(u, v)]['distance'] = d
    return G


# get_edge_feature into an np array
def get_edge_feature(dct): 
    
    assert 'kind' in dct
    assert 'distance' in dct
    ftr = np.zeros(6)
    idxmap = {bond: i for i, bond in enumerate(['distance_threshold', 'hbond', 'ionic', 'peptide_bond', 'disulfide'])}
    for bond in dct['kind']: ftr[idxmap[bond]] = 1.
    ftr[5] = 1 / dct['distance']
    return ftr


# get interface nodes to mask out
def get_interface_nodes(g): 
    
    interface_nodes = set()
    for u, v in g.edges(): 
        if g.nodes[u]['chain_id'] == g.nodes[v]['chain_id']: continue
        interface_nodes.add(u)
        interface_nodes.add(v)
    return interface_nodes



# convert residue-residue to data in pyg
def rin_to_data(g, describe=True, task='AA'): 
    
    assert task in ['AA', 'IF']
    data = GraphFormatConvertor("nx", "pyg").convert_nx_to_pyg(g)
    
    data.x = torch.tensor(np.array([g.nodes[node]['meiler'].to_numpy() for node in g.nodes()]), dtype=torch.float)
    data.y = torch.tensor(np.array([AA_ONE_HOT[three_to_one_with_mods(g.nodes[node]['residue_name'])] if three_to_one_with_mods(g.nodes[node]['residue_name']) in AA_ONE_HOT else 0 for node in g.nodes()]))
    data.u = torch.tensor([[0]]) # whole graph embeddings
    
    data.edge_attr = torch.tensor(np.array([get_edge_feature(g.edges[edge]) for edge in g.edges()]), dtype=torch.float)
    data.num_edges = len(g.edges())

    interface_nodes = get_interface_nodes(g)
    assert len(interface_nodes) > 0
    data.node_mask = torch.tensor([node in interface_nodes for node in g.nodes()], dtype=torch.bool)
    # Side note: 
    # Some attribute names work and some don't keep the tensor.bool datatype
    # See: https://github.com/pyg-team/pytorch_geometric/discussions/4237
    
    # IMPORTANT !!
    # this time we might not want to use the self information 
    # because the residue prediction depends on the neighboring residues
    # we should only use the information of the neighbors
    if task == 'AA': assert not data.has_self_loops(), 'Data have self loops, please double check'
    elif task == 'IF': 
        edge_index, edge_attr = add_self_loops(data.edge_index, edge_attr=data.edge_attr, fill_value='mean')
        data.edge_index = edge_index
        data.edge_attr = edge_attr
        data.y = data.node_mask + 0.0 # to float
        data.node_mask = torch.ones(data.num_nodes, dtype=torch.bool)
    # print out the description
    if describe: 
        describe_data(data)
        describe_graph(g)
    
    return data
    

# describe data
def describe_data(data): 
    print(f'Data name: {data.name}')
    print(f'Interface Nodes: {data.node_mask.sum()}')
    print(f'Interface Ratio: {100 * data.node_mask.sum() / data.num_nodes: .2f} %')
    return None

# describe graph
def describe_graph(g): 
    r, degs = nx.radius(g), dict(nx.degree(g))
    deg_max = np.array(list(degs.values())).max()
    print(f'Graph radius: {r}')
    print(f'Graph max degree: {deg_max}')
    return None
    


# directly get the data from the pdb file
def construct_data_from_pdb(pdb_path, config=CONFIG, describe=False, task='AA'): 
    assert os.path.isfile(pdb_path)
    g = construct_graph(config=config, pdb_path=pdb_path)
    g = tag_edge_distance(g)
    return rin_to_data(g, describe=describe, task=task)



# create dataloader from a list of pdb files
def construct_dataloader_from_pdbs(pdb_list, config=CONFIG, batch_size=16, shuffle=True, task='AA'): 
    datalist = []
    for pdb in tqdm(pdb_list, desc='Constructing GNN Data Loader'): 
        try: datalist.append(construct_data_from_pdb(pdb, config=config, task=task))
        except: 
            print(f'Error encountered at {pdb}..')
            continue
    return DataLoader(datalist, batch_size=batch_size, shuffle=shuffle)
