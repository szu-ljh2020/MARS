import numpy as np
import torch
from torch_geometric.data import Data
from rdkit import Chem
from prepare_mol_graph import get_atom_feature, get_bond_features


def get_submol_by_edits(p_smi, transform, type=None):
    # edit is break bonds
    p_mol = Chem.MolFromSmiles(p_smi, sanitize=False)
    mol = Chem.RWMol(p_mol)
    bond_int_to_type = {
        1: Chem.rdchem.BondType.SINGLE,
        2: Chem.rdchem.BondType.DOUBLE,
        3: Chem.rdchem.BondType.TRIPLE,
    }
    bond_idx, new_bond_type = transform[0], transform[1]
    bonds = p_mol.GetBonds()
    beg = bonds[bond_idx].GetBeginAtomIdx()
    end = bonds[bond_idx].GetEndAtomIdx()
    bond = mol.GetBondBetweenAtoms(beg, end)
    assert bond
    mol.RemoveBond(beg, end)
    if new_bond_type > 0:
        mol.AddBond(beg, end, bond_int_to_type[new_bond_type])
    synthon = mol.GetMol()

    # get synthon feautures
    atom_features_synthon_list = []
    for atom in synthon.GetAtoms():
        atom_features_synthon_list.append(get_atom_feature(atom))
    x_synthon = torch.tensor(np.array(atom_features_synthon_list), dtype=torch.float32)

    num_bond_features = 12  # bond type, bond direction
    edge_synthons_list = []
    edge_features_synthons_list = []
    if len(synthon.GetBonds()) > 0:  # mol has bonds
        adj_matrix_syn = np.eye(synthon.GetNumAtoms())
        for bk, bond in enumerate(synthon.GetBonds()):
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            edge_feature = get_bond_features(bond)
            edge_synthons_list.append((i, j))
            edge_features_synthons_list.append(edge_feature)
            edge_synthons_list.append((j, i))
            edge_features_synthons_list.append(edge_feature)
            adj_matrix_syn[i, j] = adj_matrix_syn[j, i] = 1
        edge_index_synthon = torch.tensor(np.array(edge_synthons_list).T, dtype=torch.long)
        edge_attr_synthon = torch.tensor(np.array(edge_features_synthons_list), dtype=torch.bool)
    else:  # mol has no bonds
        edge_index_synthon = torch.empty((2, 0), dtype=torch.long)
        edge_attr_synthon = torch.empty((0, num_bond_features), dtype=torch.bool)
    # return x_synthon, edge_index_synthon, edge_attr_synthon
    synthon_data = Data(x=x_synthon, edge_index=edge_index_synthon, edge_attr=edge_attr_synthon, type=type)
    return synthon_data


# avoid error from different version of pyg
def dict2string(input_dict):
    """
    covert a dict to a str
    input: input_dict, a dict
    output: string, a str
    """
    if isinstance(input_dict, dict):
        string = ""
        for k, v in input_dict.items():
            string += str(k) + ":" + str(v) + ","
        string = string.strip(",")
        return string
    else:
        raise ValueError("Expect 'dict' but got '" + type(input_dict).__name__ + "'")


def string2dict(input_string):
    """
    covert a single string to a dict
    or covert a batch of string to a batch of dict
    input: input_string, a str or a list or a tuple
    output: a single dict or a batch of dict
    """
    if isinstance(input_string, str):
        output_dict = {}
        kv_list = input_string.split(",")
        for item in kv_list:
            kv = item.split(":")
            if kv[1].isdigit():
                output_dict[int(kv[0])] = int(kv[1])
            else:
                output_dict[int(kv[0])] = kv[1]
        return output_dict
    elif isinstance(input_string, list) or isinstance(input_string, tuple):
        output = []
        for input in input_string:
            output_dict = {}
            kv_list = input.split(",")
            for item in kv_list:
                kv = item.split(":")
                output_dict[int(kv[0])] = int(kv[1])
            output.append(output_dict)
        if isinstance(input_string, tuple):
            return tuple(output)
        else:
            return output
    else:
        raise ValueError("Expect 'str', 'list' or 'tuple' but got '" + type(input_string).__name__ + "'")


def list2string(input_list):
    """
        covert a list to a str
        input: input_list, a dict
        output: string, a str
    """
    if isinstance(input_list, list):
        if input_list:
            if isinstance(input_list[0], list):
                string = []
                for l in input_list:
                    string.append(','.join([str(i) for i in l]))
                output_string = '.'.join(string) + '.'
            else:
                output_string = ','.join([str(i) for i in input_list])
        else:
            output_string = '[]'
        return output_string
    else:
        raise ValueError("Expect 'list', but got '" + type(input_list).__name__ + "'")


def string2list(input_string):
    """
        covert a single string to a list
        or covert a batch of string to a batch of list
        input: input_string, a str or a list or a tuple
        output: a single list or a batch of list
    """
    if isinstance(input_string, str):
        if input_string == '[]':
            output_list = []
        elif input_string[-1] == '.':
            output_list = []
            sub_string = input_string.strip(".").split(".")
            for s in sub_string:
                output_list.append([int(i) for i in s.split(',')])
        else:
            output_list = list(map(int, input_string.split(',')))
        return output_list
    if isinstance(input_string, list) or isinstance(input_string, tuple):
        output = []
        for l in input_string:
            if isinstance(l, str):
                if l == '[]':
                    output_list = []
                elif l[-1] == '.':
                    output_list = []
                    sub_string = l.strip(".").split(".")
                    for s in sub_string:
                        output_list.append([int(i) for i in s.split(',')])
                else:
                    output_list = list(map(int, l.split(',')))
                output.append(output_list)
            else:
                raise ValueError("Expect 'str', but got '" + type(l).__name__ + "'")
        if isinstance(input_string, tuple):
            return tuple(output)
        else:
            return output
    else:
        raise ValueError("Expect 'str', but got '" + type(input_string).__name__ + "'")


