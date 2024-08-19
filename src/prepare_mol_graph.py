import collections
import copy
import json
import numpy as np
import os
import random
import torch
import pandas as pd
import pickle

from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.data import Dataset
from tqdm import tqdm
from sklearn.metrics import pairwise_distances
import torch_geometric.transforms as T

import chemutils
from junction_graph import JunctionGraph, JunctionNode
from graph_positional_encoding import laplacian_positional_encoding

import re

from utils import *


def smarts2smiles(smarts, sanitize=True, canonical=True):
    t = re.sub(':\d*', '', smarts)
    mol = Chem.MolFromSmiles(t, sanitize=sanitize)
    return Chem.MolToSmiles(mol, canonical=canonical)

def get_distance_matrix(mol):
    try:
        mol_new = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_new, maxAttempts=5000)
        AllChem.UFFOptimizeMolecule(mol_new)
        mol = Chem.RemoveHs(mol_new)
    except:
        AllChem.Compute2DCoords(mol)
    conf = mol.GetConformer()
    pos_matrix = np.array([[conf.GetAtomPosition(k).x, conf.GetAtomPosition(k).y, conf.GetAtomPosition(k).z]
                           for k in range(mol.GetNumAtoms())])
    dist_matrix = pairwise_distances(pos_matrix)
    return dist_matrix


class RXNInfo:
    def __init__(self, mol):
        self.mol = mol
        self.atom_features = {}
        self.bond_features = {}
        self.getFeatures()

    def getFeatures(self):
        for atom in self.mol.GetAtoms():
            self.atom_features[atom.GetAtomMapNum()] = get_atom_feature(atom)
        for bond in self.mol.GetBonds():
            self.bond_features[(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())] = get_bond_features(bond)

    def get_atom_feature(self, atom_map_num):
        return self.atom_features.get(atom_map_num, None)

    def get_bond_feature(self, bond_begin_idx, bond_end_idx):
        feature = self.bond_features.get((bond_begin_idx, bond_end_idx), None)
        if not feature:
            feature = self.bond_features.get((bond_end_idx, bond_begin_idx), None)
        return feature


def mol_to_graph_data_obj(mol, synthon, bond_transformations, attach_indexes, rxn_info):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object -- product mol
    :param synthon: rdkit mol object of synthon
    :param bond_transformations: which bond change
    :param attach_indexes: extra attachment
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """

    # atoms
    atom_features_list = []
    atom_targets_list = []
    atom_symbols_list = []
    for atom in mol.GetAtoms():
        atom_features_list.append(rxn_info.get_atom_feature(atom.GetAtomMapNum()))
        atom_targets_list.append(atom.GetIdx() in attach_indexes)
        atom_symbols_list.append(atom.GetSymbol())
    x = torch.tensor(np.array(atom_features_list), dtype=torch.float32)
    atom_targets = torch.tensor(np.array(atom_targets_list), dtype=torch.float32)
    # TODO: atom_features_synthon_list
    atom_features_synthon_list = []
    for atom in synthon.GetAtoms():
        atom_features_synthon_list.append(get_atom_feature(atom))
    x_synthon = torch.tensor(np.array(atom_features_synthon_list), dtype=torch.float32)

    # bonds
    num_bond_features = 12  # bond type, bond direction
    edges_list = []
    edge_features_list = []
    edge_targets_list = []
    edge_transformations = []
    bondidx2atomidx = []
    adj_matrix = np.eye(mol.GetNumAtoms())
    if len(mol.GetBonds()) > 0:  # mol has bonds
        for bk, bond in enumerate(mol.GetBonds()):
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bondidx2atomidx.append((i, j))
            edge_feature = rxn_info.get_bond_feature(i, j)
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)
            adj_matrix[i, j] = adj_matrix[j, i] = 1
            if (i, j) in bond_transformations:
                edge_targets_list.append(bond_transformations[(i, j)])
                edge_targets_list.append(bond_transformations[(i, j)])
                # edge transformation, first integer indicates the bond index in rk mol object
                # second indicates the target bond type
                edge_transformations.append([bk, bond_transformations[(i, j)]])
            elif (j, i) in bond_transformations:
                edge_targets_list.append(bond_transformations[(j, i)])
                edge_targets_list.append(bond_transformations[(j, i)])
                edge_transformations.append([bk, bond_transformations[(j, i)]])
            else:
                edge_targets_list.append(int(bond.GetBondType()))
                edge_targets_list.append(int(bond.GetBondType()))

        assert len(edge_targets_list) == len(edges_list)
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)
        edge_attr = torch.tensor(np.array(edge_features_list), dtype=torch.bool)
        edge_targets = torch.tensor(np.array(edge_targets_list), dtype=torch.int8)

        adj_matrix = torch.tensor(adj_matrix, dtype=torch.long)
        la_pe = laplacian_positional_encoding(adj_matrix, 8)

    else:  # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.bool)
        edge_targets = torch.empty((0), dtype=torch.int8)
        adj_matrix = np.eye(mol.GetNumAtoms())
        adj_matrix = torch.tensor(adj_matrix, dtype=torch.long)
        la_pe = laplacian_positional_encoding(adj_matrix, 8)

    edge_synthons_list = []
    edge_features_synthons_list = []
    bondidx2atomidx_synthon = []
    if len(synthon.GetBonds()) > 0:  # mol has bonds
        adj_matrix_syn = np.eye(synthon.GetNumAtoms())
        for bk, bond in enumerate(synthon.GetBonds()):
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bondidx2atomidx_synthon.append((i, j))
            edge_feature = get_bond_features(bond)
            edge_synthons_list.append((i, j))
            edge_features_synthons_list.append(edge_feature)
            edge_synthons_list.append((j, i))
            edge_features_synthons_list.append(edge_feature)
            adj_matrix_syn[i, j] = adj_matrix_syn[j, i] = 1
        edge_index_synthon = torch.tensor(np.array(edge_synthons_list).T, dtype=torch.long)
        edge_attr_synthon = torch.tensor(np.array(edge_features_synthons_list), dtype=torch.bool)
        adj_matrix_syn = torch.tensor(adj_matrix_syn, dtype=torch.long)
        syn_pe = laplacian_positional_encoding(adj_matrix_syn, 8)
    else:  # mol has no bonds
        edge_index_synthon = torch.empty((2, 0), dtype=torch.long)
        edge_attr_synthon = torch.empty((0, num_bond_features), dtype=torch.bool)
        adj_matrix_syn = np.eye(synthon.GetNumAtoms())
        adj_matrix_syn = torch.tensor(adj_matrix_syn, dtype=torch.long)
        syn_pe = laplacian_positional_encoding(adj_matrix_syn, 8)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    data.atom_targets = atom_targets
    data.atom_len = len(mol.GetAtoms())
    data.atom_transformations = sorted(attach_indexes)
    data.atom_symbols = atom_symbols_list
    data.edge_targets = edge_targets
    data.edge_len = len(mol.GetBonds())
    data.edge_transformations = edge_transformations
    data.bondidx2atomidx = bondidx2atomidx
    data.pe = la_pe

    synthon_data = Data(x=x_synthon, edge_index=edge_index_synthon, edge_attr=edge_attr_synthon)
    synthon_data.pe = syn_pe

    return data, synthon_data


def decode_transformation(motif_vocab, gnn_data, transformation_path):
    """
    Decode reactants from starting product molecule and encoded transformation

    :param motif_vocab:
    :param gnn_data:
    :param transformation_path:
    :return: decoded reactants molecule
    """

    product = gnn_data.product
    p_mol = Chem.MolFromSmiles(product, sanitize=False)
    product_attach_indexes = gnn_data.atom_transformations
    bond_transformations = {}
    bonds = p_mol.GetBonds()
    for edge_transform in gnn_data.edge_transformations:
        bond_idx, new_bond_type = edge_transform
        beg = bonds[bond_idx].GetBeginAtomIdx()
        end = bonds[bond_idx].GetEndAtomIdx()
        bond_transformations[(beg, end)] = new_bond_type
        product_attach_indexes.extend([beg, end])

    synthon_mol = chemutils.apply_transform(p_mol, bond_transformations, product_attach_indexes)
    synthon = Chem.MolToSmiles(synthon_mol, kekuleSmiles=True, canonical=True)
    if synthon != gnn_data.junction_graph.synthon:
        print('synthon not equal:', gnn_data.junction_graph.synthon, synthon, )

    jgraph = JunctionGraph(synthon)
    jgraph.product = product
    total_root_attachments = 0
    for path in transformation_path:
        total_root_attachments += path[0]
    if total_root_attachments != len(jgraph.nodes[0].attachments):
        print('total_root_attachments not equal:', len(jgraph.nodes[0].attachments), total_root_attachments)

    res = jgraph.decode_transformation(motif_vocab, transformation_path)

    return res


class MoleculeDataset(Dataset):
    def __init__(self, root, split='train', transform=None):
        """
        Adapted from https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/loader.py
        :param root: directory of the dataset, containing a raw and processed1 dir.
            The raw dir should contain the file containing the smiles, and the
            processed1 dir can either empty or a previously processed1 file
        :param dataset: name of the dataset. Currently only implemented for USPTO50K
        """
        self.split = split
        self.root = os.path.join(root, split)
        #self.transform = transform
        super(MoleculeDataset, self).__init__(self.root)
        if os.path.isdir(self.processed_dir):
            files = [f for f in os.listdir(self.processed_dir) if f.endswith('.pkl')]
            files = sorted(files)
            self.process_data_files = [os.path.join(self.processed_dir, f) for f in files]

        self.indexed_motifs_json = os.path.join(self.root, 'indexed_motifs.json')
        if os.path.isfile(self.indexed_motifs_json) and split == 'train':
            with open(self.indexed_motifs_json) as f:
                self.indexed_motifs = json.load(f)
            max_attachments = 1
            motif_vocab = {}
            attachment_symbols = set()
            for symbol, motif in self.indexed_motifs.items():
                attachment_symbols.add(symbol)
                for mt, attachments in motif.items():
                    motif_vocab[mt] = attachments
                    for att in attachments:
                        max_attachments = max(max_attachments, len(att))
            print('max_attachments in encode_transformation:', max_attachments)

            keys = sorted(motif_vocab.keys())
            self.motif_vocab = collections.OrderedDict()
            for key in keys:
                self.motif_vocab[key] = motif_vocab[key]
                mol = Chem.MolFromSmiles(key, sanitize=False)
                symbols = []
                for atom in mol.GetAtoms():
                    if atom.GetAtomMapNum() in motif_vocab[key][0]:
                        symbols.append(atom.GetSymbol())
                self.motif_vocab[key].append(symbols)

            # prepare the motif masks for decoding
            self.motif_masks = {symbol: torch.zeros((1, 211), dtype=torch.float32) for symbol in attachment_symbols}
            for symbol, motif in self.indexed_motifs.items():
                for mt, attachments in motif.items():
                    assert mt in keys
                    idx = keys.index(mt)
                    self.motif_masks[symbol][0][idx] = 1

    def len(self):
        return len(self.processed_file_names)

    def _permute_sequence(self, rnn_input, rnn_target, shuffle=True):
        prev = 0
        inputs, targets = [], []
        for idx, transform in enumerate(rnn_input):
            if transform[0] == 6:
                inputs.append(rnn_input[prev:idx])
                targets.append(rnn_target[prev:idx])
                prev = idx
        inputs.append(rnn_input[prev:])
        targets.append(rnn_target[prev:])
        indexes = list(range(1, len(inputs)))
        if shuffle:
            random.shuffle(indexes)
        else:
            indexes.reverse()
        rnn_input = inputs[0]
        rnn_target = targets[0]
        for idx in indexes:
            rnn_input.extend(inputs[idx])
            rnn_target.extend(targets[idx])
        return rnn_input, rnn_target

    def get(self, idx):
        with open(self.processed_file_names[idx], 'rb') as f:
            precessed_rxn = pickle.load(f)
        gnn_data = precessed_rxn['gnn_data']
        gnn_data_synthon = precessed_rxn['gnn_data_synthon']
        gnn_data.junction_graph = precessed_rxn['junction_graph']
        return gnn_data, gnn_data_synthon

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        return file_name_list

    @property
    def processed_file_names(self):
        return self.process_data_files

    def process_data(self):
        input_path = self.raw_paths[0]
        print('input_path:', input_path)
        df = pd.read_csv(input_path)
        rxns = df['rxn_smiles'].tolist()
        types = df['class'].tolist()

        if not os.path.isdir(self.processed_dir):
            os.mkdir(self.processed_dir)

        skipped = 0
        cnt1, cnt2, cnt3, cnt4 = 0, 0, 0, 0
        self.process_data_files = []

        indexed_motifs = {}
        lg_smi_cano_dict = {}
        lg_smi_dict = {}
        cnt_lc = 0
        for k, rxn in tqdm(enumerate(rxns)):
            # if k < 298: continue
            reactant, product = rxn.strip().split('>>')
            r_mol = Chem.MolFromSmiles(reactant)
            # make sure all atoms have a mapping number
            max_mapnum = 0
            for atom in r_mol.GetAtoms():
                max_mapnum = max(max_mapnum, atom.GetAtomMapNum())
            for atom in r_mol.GetAtoms():
                if atom.GetAtomMapNum() == 0:
                    max_mapnum += 1
                    atom.SetAtomMapNum(max_mapnum)
            p_mol = Chem.MolFromSmiles(product)
            n_atom = p_mol.GetNumAtoms()

            if p_mol.GetNumHeavyAtoms() <= 1:
                continue

            # make the reactant kekulized in the same way as the product
            r_mol_kekulized, p_mol_kekulized = chemutils.align_kekule_pairs(r_mol, p_mol)
            try:
                r_smi = Chem.MolToSmiles(r_mol_kekulized, kekuleSmiles=True)
                p_smi = Chem.MolToSmiles(p_mol_kekulized, kekuleSmiles=True)
            except:
                skipped += 1
                continue
            if not chemutils.cycle_transform(mol=r_mol_kekulized):
                print('can not make the reactant kekulized in the same way as the product')
                Chem.Kekulize(r_mol)
                r_mol_kekulized = r_mol
            if not chemutils.cycle_transform(mol=r_mol_kekulized) or not chemutils.cycle_transform(mol=p_mol_kekulized):
                print('can not align kekule pair, skip')
                skipped += 1
                continue

            reactant = Chem.MolToSmiles(r_mol_kekulized, kekuleSmiles=True)
            product = Chem.MolToSmiles(p_mol_kekulized, kekuleSmiles=True)
            r_mol = Chem.MolFromSmiles(reactant, sanitize=False)
            p_mol = Chem.MolFromSmiles(product, sanitize=False)

            rxn_info = RXNInfo(p_mol)

            patomidx2mapnum = chemutils.get_atomidx2mapnum(p_mol)   # product index to mapping number
            pmapnum2atomidx = chemutils.get_mapnum2atomidx(p_mol)   # product mapping number to index
            rmapnum2atomidx = chemutils.get_mapnum2atomidx(r_mol)   # reaction mapping number to index

            # prepare bond transformation operations
            bond_target = []
            bond_transformations = {}
            product_attach_trans_indexes = set()
            for bk, bond in enumerate(p_mol.GetBonds()):
                beg = bond.GetBeginAtom()
                end = bond.GetEndAtom()
                reactant_beg = rmapnum2atomidx[beg.GetAtomMapNum()]
                reactant_end = rmapnum2atomidx[end.GetAtomMapNum()]
                reactant_bond = r_mol.GetBondBetweenAtoms(reactant_beg, reactant_end)
                bond_atom_index = (beg.GetIdx(), end.GetIdx())
                if not reactant_bond:
                    bond_transformations[bond_atom_index] = 0
                    product_attach_trans_indexes.add(bond_atom_index[0])
                    product_attach_trans_indexes.add(bond_atom_index[1])
                    bond_target.extend([0, 0])
                elif bond.GetBondType() != reactant_bond.GetBondType():
                    cnt3 += 1
                    bond_transformations[bond_atom_index] = int(reactant_bond.GetBondType())
                    product_attach_trans_indexes.add(bond_atom_index[0])
                    product_attach_trans_indexes.add(bond_atom_index[1])
                    bond_target.extend([
                        int(reactant_bond.GetBondType()),
                        int(reactant_bond.GetBondType())])
                else:
                    bond_target.extend([
                        int(reactant_bond.GetBondType()),
                        int(reactant_bond.GetBondType())])

            # attachment mapping numbers
            attachments = chemutils.get_attachments(p_mol, r_mol)
            product_attach_indexes = [pmapnum2atomidx[a] for a in attachments]  # normal (leaving group) to predict
            assert set(product_attach_indexes) >= set(product_attach_trans_indexes)
            product_attach_extra_indexes = set(product_attach_indexes) - set(
                product_attach_trans_indexes)  # self-bond to predict

            for bond in p_mol.GetBonds():
                if bond.GetBondType() == Chem.BondType.AROMATIC:
                    raise ('kekulized mol should not have aromatic bond!')
            for bond in r_mol.GetBonds():
                if bond.GetBondType() == Chem.BondType.AROMATIC:
                    raise ('kekulized mol should not have aromatic bond!')

            synthon_mol = chemutils.apply_transform(p_mol, bond_transformations,
                                                    product_attach_indexes)  # rooted synthon
            synthon = Chem.MolToSmiles(synthon_mol, kekuleSmiles=True, canonical=True)

            # extract graph features for gnn model
            gnn_data, gnn_data_synthon = mol_to_graph_data_obj(p_mol, synthon_mol, bond_transformations,
                                                               product_attach_extra_indexes, rxn_info)

            if gnn_data.atom_transformations == None:
                print(rxn)
            if gnn_data.edge_transformations == None:
                print(rxn)
            gnn_data.id = k
            gnn_data.type = types[k]
            gnn_data.product = product
            gnn_data.pmapnum2atomidx = pmapnum2atomidx
            gnn_data.patomidx2mapnum = patomidx2mapnum
            gnn_data_synthon.type = types[k]

            for atom in p_mol.GetAtoms():
                atom.SetNumExplicitHs(0)
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            for atom in r_mol.GetAtoms():
                atom.SetNumExplicitHs(0)
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            for bond in p_mol.GetBonds():
                bond.SetStereo(Chem.BondStereo.STEREOANY)
            for bond in r_mol.GetBonds():
                bond.SetStereo(Chem.BondStereo.STEREOANY)

            cnt_lc += 1
            visited = []
            lg_mol = Chem.Mol()
            # find the connected leaving group for each unvisited attachment atom
            new_attachments = []
            new_lg = Chem.Mol()
            for start_nummap in attachments:
                # two attachment atoms may connect the same leaving group
                if start_nummap in visited: continue
                cur_visited = chemutils.dfs_lg(r_mol, rmapnum2atomidx, pmapnum2atomidx, start_nummap, visited)
                lg = chemutils.get_sub_mol(r_mol, cur_visited, attachments)
                lg_mol = Chem.CombineMols(lg_mol, lg)
                lg_smi = Chem.MolToSmiles(lg_mol)
            assert set(attachments) & set(visited) == set(attachments)

            if lg_mol.GetNumAtoms() == 0:
                cnt_lc -= 1
            lg_smi = Chem.MolToSmiles(lg_mol, canonical=False, kekuleSmiles=True)
            lg_smi_cano = smarts2smiles(lg_smi, False, True)
            lg_smi_cano_dict.setdefault(lg_smi_cano, []).append(rxn)

            # build junction graph
            for atom in lg_mol.GetAtoms():
                # add 1000 to mapnum of attachments
                if atom.GetAtomMapNum() in attachments:
                    atom.SetAtomMapNum(atom.GetAtomMapNum() + 1000)
            lg_configuration = Chem.MolToSmiles(lg_mol, canonical=False)
            lg_smi_dict.setdefault(lg_configuration, []).append(rxn)

            jgraph = JunctionGraph(synthon, lg_configuration)
            jgraph.build_junction_graph(jgraph.synthon, jgraph.leaving_group)
            jgraph.reactant = reactant
            jgraph.product = product

            [atom.SetAtomMapNum(0) for atom in r_mol.GetAtoms()]
            Chem.SanitizeMol(r_mol)
            reactant_cano = Chem.MolToSmiles(r_mol, kekuleSmiles=False, canonical=True)
            jgraph.reactant_cano = reactant_cano
            path = jgraph.dfs_path()  # sequence for training

            gnn_data.synthon_attachment_indexes = []
            for vatt in jgraph.visited_attachments:
                if vatt[0] == 0:
                    gnn_data.synthon_attachment_indexes.append(pmapnum2atomidx[vatt[1]])
            gnn_data.synthon_attachment_idx2symbols = {}
            for mapnum, symbol in zip(jgraph.nodes[0].attachments, jgraph.nodes[0].attachment_atom_symbols):
                gnn_data.synthon_attachment_idx2symbols[gnn_data.pmapnum2atomidx[mapnum]] = symbol

            symbols = [p[0] for p in path]
            if 2 in symbols:
                print('loop found, and skip')
                skipped += 1
                cnt4 += 1
                continue

            # extract motif vocabulary, skip the root node
            for node in jgraph.nodes[1:]:
                for symbol in node.attachment_atom_symbols:
                    if symbol not in indexed_motifs:
                        indexed_motifs[symbol] = {}
                    if node.smiles not in indexed_motifs[symbol]:
                        indexed_motifs[symbol][node.smiles] = set()
                    indexed_motifs[symbol][node.smiles].add(tuple(sorted(node.attachments)))

            # try to reconstruct molecule from the traversal path
            smi_cano = jgraph.reconstruct_molcule_from_path()
            if smi_cano == reactant_cano:
                cnt1 += 1
            else:
                cnt2 += 1

            precessed_rxn = {
                'gnn_data': gnn_data,
                'gnn_data_synthon': gnn_data_synthon,
                'junction_graph': jgraph,
            }

            processed_data_file = os.path.join(self.processed_dir, '{}.pkl'.format(k))
            self.process_data_files.append(processed_data_file)
            with open(processed_data_file, 'wb') as f:
                pickle.dump(precessed_rxn, f, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.indexed_motifs_json.replace('indexed_motifs.json', 'lg_smis.json'), 'w', encoding='utf-8') as f:
            json.dump(lg_smi_cano_dict, f, indent=4, sort_keys=True)
        with open(self.indexed_motifs_json.replace('indexed_motifs.json', 'lg_smis_origin.json'), 'w',
                  encoding='utf-8') as f:
            json.dump(lg_smi_dict, f, indent=4, sort_keys=True)
        print(cnt_lc)

        print('skipped rxns:', skipped)
        print(cnt1, cnt2, cnt3, cnt4)
        sorted_indexed_motifs = collections.OrderedDict()
        for symbol, motif in indexed_motifs.items():
            for mt, attachments in motif.items():
                if len(attachments) > 1:
                    raise ValueError('motif has two configurations.')
                indexed_motifs[symbol][mt] = sorted(list(attachments))
            sorted_indexed_motifs[symbol] = collections.OrderedDict(indexed_motifs[symbol])
        self.indexed_motifs = sorted_indexed_motifs
        with open(self.indexed_motifs_json, 'w', encoding='utf-8') as f:
            json.dump(sorted_indexed_motifs, f, indent=4, sort_keys=True)

    def encode_transformation(self, motif_vocab):

        motifs = list(motif_vocab.keys())
        for idx, pfn in enumerate(tqdm(self.processed_file_names)):
            gnn_data, gnn_data_synthon = self.get(idx)

            # avoid pyg’s error
            if isinstance(gnn_data.edge_transformations, str):
                gnn_data.edge_transformations = string2list(gnn_data.edge_transformations)
            if isinstance(gnn_data.atom_transformations, str):
                gnn_data.atom_transformations = string2list(gnn_data.atom_transformations)
            if isinstance(gnn_data.pmapnum2atomidx, str):
                gnn_data.pmapnum2atomidx = string2dict(gnn_data.pmapnum2atomidx)
            if isinstance(gnn_data.patomidx2mapnum, str):
                gnn_data.patomidx2mapnum = string2dict(gnn_data.patomidx2mapnum)
            if isinstance(gnn_data.synthon_attachment_idx2symbols, str):
                gnn_data.synthon_attachment_idx2symbols = string2dict(gnn_data.synthon_attachment_idx2symbols)
            if isinstance(gnn_data.synthon_attachment_indexes, str):
                gnn_data.synthon_attachment_indexes = string2list(gnn_data.synthon_attachment_indexes)

            jgraph = gnn_data.junction_graph
            jgraph.build_transformation_path(motifs)

            # build rnn input and target sequence
            rnn_input, rnn_target = [], []
            # start of edge transformation
            rnn_input.append((0,))
            for et in gnn_data.edge_transformations:
                try:
                    edge_idx, new_type = et
                except:
                    print(gnn_data)
                    return
                rnn_target.append((1, edge_idx, new_type))
                rnn_input.append((1, edge_idx, new_type))
            # encode atom transformations as self-edge transformations
            for at in gnn_data.atom_transformations:
                rnn_target.append((1, gnn_data.edge_len + at, 0))
                rnn_input.append((1, gnn_data.edge_len + at, 0))
            rnn_target.append((4,))

            # start of motif generation
            for k, tp in enumerate(jgraph.transformation_path):
                start, motif_idx, attachment_idx, start_attachment = tp
                if start:
                    start_atom_idx = gnn_data.pmapnum2atomidx[start_attachment]
                    rnn_input.append((6, start_atom_idx))
                else:
                    assert rnn_target[-1][0] == 5
                    rnn_input.append(rnn_target[-1])
                # determine the target state
                if k == len(jgraph.transformation_path) - 1 or jgraph.transformation_path[k + 1][0]:
                    rnn_target.append((6, motif_idx, attachment_idx))
                else:
                    rnn_target.append((5, motif_idx, attachment_idx))

            gnn_data.rnn_input = rnn_input
            gnn_data.rnn_target = rnn_target
            with open(pfn, 'wb') as f:
                del gnn_data.junction_graph
                gnn_data.atom_transformations = list2string(gnn_data.atom_transformations)
                gnn_data.edge_transformations = list2string(gnn_data.edge_transformations)
                gnn_data.pmapnum2atomidx = dict2string(gnn_data.pmapnum2atomidx)
                gnn_data.patomidx2mapnum = dict2string(gnn_data.patomidx2mapnum)
                gnn_data.synthon_attachment_indexes = list2string(gnn_data.synthon_attachment_indexes)
                gnn_data.synthon_attachment_idx2symbols = dict2string(gnn_data.synthon_attachment_idx2symbols)
                pickle.dump({'gnn_data': gnn_data, 'junction_graph': jgraph, 'gnn_data_synthon': gnn_data_synthon}, f,
                            protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    for split in ['test', 'valid', 'train']:
        dataset_train = MoleculeDataset('data/USPTO50K', split)
        dataset_train.process_data()

    dataset_train = MoleculeDataset('data/USPTO50K', 'train')
    motif_vocabs = dataset_train.motif_vocab
    with open('data/USPTO50K/motif_vocab.pkl', 'wb') as f:
        pickle.dump(motif_vocabs, f)

    for split in ['valid', 'train', 'test']:
        dataset_test = MoleculeDataset('data/USPTO50K', split)
        dataset_test.encode_transformation(dataset_train.motif_vocab)
