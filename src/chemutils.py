import re
import rdkit
import rdkit.Chem as Chem

BOND_TYPES = [None, Chem.rdchem.BondType.SINGLE, Chem.rdchem.BondType.DOUBLE, \
              Chem.rdchem.BondType.TRIPLE, Chem.rdchem.BondType.AROMATIC]
BOND_FLOAT_TO_TYPE = {
    0.0: BOND_TYPES[0],
    1.0: BOND_TYPES[1],
    2.0: BOND_TYPES[2],
    3.0: BOND_TYPES[3],
    1.5: BOND_TYPES[4],
}

idxfunc = lambda a: a.GetAtomMapNum() - 1


#  Get the dict maps each atom index to the mapping number.
def get_atomidx2mapnum(mol):
    atomidx2mapnum = {}
    for atom in mol.GetAtoms():
        assert atom.GetAtomMapNum() > 0
        atomidx2mapnum[atom.GetIdx()] = atom.GetAtomMapNum()
    return atomidx2mapnum


#  Get the dict maps each mapping number to the atom index .
def get_mapnum2atomidx(mol):
    mapnum2atomidx = {}
    for atom in mol.GetAtoms():
        assert atom.GetAtomMapNum() > 0
        mapnum2atomidx[atom.GetAtomMapNum()] = atom.GetIdx()
    return mapnum2atomidx


def get_mapnum(smarts_item):
    item = re.findall('(?<=:)\d+', smarts_item)
    item = list(map(int, item))
    return item


pattern = "(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\\\|\/|:|~|@|\?|>|\*|\$|\%[0-9]{2}|[0-9])"
regex = re.compile(pattern)


def smi_tokenizer(smi, regex=regex):
    tokens = [token for token in regex.findall(smi)]
    assert smi == ''.join(tokens)
    return tokens


def set_atommap(mol, num=0):
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(num)
    return mol


def get_mol(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None: Chem.Kekulize(mol)
    return mol


def get_smiles(mol):
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


def sanitize(mol, kekulize=True):
    try:
        smiles = get_smiles(mol) if kekulize else Chem.MolToSmiles(mol)
        mol = get_mol(smiles) if kekulize else Chem.MolFromSmiles(smiles)
    except:
        mol = None
    return mol


def is_aromatic_ring(mol):
    if mol.GetNumAtoms() == mol.GetNumBonds():
        aroma_bonds = [b for b in mol.GetBonds() if b.GetBondType() == Chem.rdchem.BondType.AROMATIC]
        return len(aroma_bonds) == mol.GetNumBonds()
    else:
        return False


def find_fragments(mol):
    new_mol = Chem.RWMol(mol)
    smi = Chem.MolToSmiles(mol)
    for bond in mol.GetBonds():
        # skip ring bond
        if bond.IsInRing(): continue
        a1 = bond.GetBeginAtom()
        a2 = bond.GetEndAtom()
        # both atoms are in two different rings
        if a1.IsInRing() and a2.IsInRing():
            new_idx1 = new_mol.AddAtom(a1)
            new_idx2 = new_mol.AddAtom(a2)
            new_mol.AddBond(new_idx1, new_idx2, bond.GetBondType())
            new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())
            new_mol.GetAtomWithIdx(new_idx1).SetAtomMapNum(a1.GetAtomMapNum() + 1000)
            new_mol.GetAtomWithIdx(new_idx2).SetAtomMapNum(a2.GetAtomMapNum() + 1000)
            new_mol.GetAtomWithIdx(a1.GetIdx()).SetAtomMapNum(a1.GetAtomMapNum() + 1000)
            new_mol.GetAtomWithIdx(a2.GetIdx()).SetAtomMapNum(a2.GetAtomMapNum() + 1000)

        # one atom is in ring, the other atom degree more than 1
        elif a1.IsInRing() and a2.GetDegree() > 1:
            new_idx = new_mol.AddAtom(a1)
            new_mol.AddBond(new_idx, a2.GetIdx(), bond.GetBondType())
            new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())
            new_mol.GetAtomWithIdx(new_idx).SetAtomMapNum(a1.GetAtomMapNum() + 1000)
            new_mol.GetAtomWithIdx(a1.GetIdx()).SetAtomMapNum(a1.GetAtomMapNum() + 1000)

        elif a2.IsInRing() and a1.GetDegree() > 1:
            new_idx = new_mol.AddAtom(a2)
            new_mol.AddBond(new_idx, a1.GetIdx(), bond.GetBondType())
            new_mol.RemoveBond(a1.GetIdx(), a2.GetIdx())
            new_mol.GetAtomWithIdx(new_idx).SetAtomMapNum(a2.GetAtomMapNum() + 1000)
            new_mol.GetAtomWithIdx(a2.GetIdx()).SetAtomMapNum(a2.GetAtomMapNum() + 1000)

    new_mol = new_mol.GetMol()
    new_smiles = Chem.MolToSmiles(new_mol, canonical=False)

    # manually fix the explicit Hs and chirality property issue
    tokens = smi_tokenizer(smi)
    atoms = [t for t in tokens if ':' in t]
    mapnum2token = {}
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    for idx, atom in enumerate(mol.GetAtoms()):
        mapnum2token[atom.GetAtomMapNum() % 1000] = atoms[idx]

    new_smiles_tokens = smi_tokenizer(new_smiles)
    new_smiles_atom_indexes = [i for i, t in enumerate(new_smiles_tokens) if ':' in t]
    new_mol = Chem.MolFromSmiles(new_smiles, sanitize=False)
    for idx, atom in enumerate(new_mol.GetAtoms()):
        origin_token = mapnum2token[atom.GetAtomMapNum() % 1000]
        origin_token1, origin_token2 = origin_token.split(':')
        new_token = new_smiles_tokens[new_smiles_atom_indexes[idx]]
        new_token1, new_token2 = new_token.split(':')
        if origin_token1 != new_token1:
            print('fix leaving group motif atoms:', origin_token, new_token)
            new_smiles_tokens[new_smiles_atom_indexes[idx]] = origin_token1 + ':' + new_token2
    fixed_smiles = ''.join(new_smiles_tokens)

    return fixed_smiles.split('.')

    # smiles = Chem.MolToSmiles(mol)
    # hopts = []
    # for fragment in new_smiles.split('.'):
    #     fmol = Chem.MolFromSmiles(fragment, sanitize=False)
    #     attachment_mapnum = [atom.GetAtomMapNum() % 1000 for atom in fmol.GetAtoms() if atom.GetAtomMapNum() > 1000]
    #     indices = set([atom.GetAtomMapNum() % 1000 - 1 for atom in fmol.GetAtoms()])
    #     fmol = get_sub_mol(mol, indices)
    #     fsmiles = Chem.MolToSmiles(fmol)
    #     for atom in fmol.GetAtoms():
    #         if atom.GetAtomMapNum() in attachment_mapnum:
    #             atom.SetAtomMapNum(1)
    #         else:
    #             atom.SetAtomMapNum(0)
    #     fsmiles = Chem.MolToSmiles(fmol)
    #     hopts.append((fsmiles, indices))

    # return hopts


def get_leaves(mol):
    leaf_atoms = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetDegree() == 1]

    clusters = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        if not bond.IsInRing():
            clusters.append(set([a1, a2]))

    rings = [set(x) for x in Chem.GetSymmSSSR(mol)]
    clusters.extend(rings)

    leaf_rings = []
    for r in rings:
        inters = [c for c in clusters if r != c and len(r & c) > 0]
        if len(inters) > 1: continue
        nodes = [i for i in r if mol.GetAtomWithIdx(i).GetDegree() == 2]
        leaf_rings.append(max(nodes))

    return leaf_atoms + leaf_rings


def atom_equal(a1, a2):
    return a1.GetSymbol() == a2.GetSymbol() and a1.GetFormalCharge() == a2.GetFormalCharge()


def bond_match(mol1, a1, b1, mol2, a2, b2):
    a1, b1 = mol1.GetAtomWithIdx(a1), mol1.GetAtomWithIdx(b1)
    a2, b2 = mol2.GetAtomWithIdx(a2), mol2.GetAtomWithIdx(b2)
    return atom_equal(a1, a2) and atom_equal(b1, b2)


def copy_atom(atom, atommap=True):
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    if atommap:
        new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom


# #mol must be RWMol object
# def get_sub_mol(mol, sub_atoms):
#     new_mol = Chem.RWMol()
#     atom_map = {}
#     for idx in sub_atoms:
#         atom = mol.GetAtomWithIdx(idx)
#         atom_map[idx] = new_mol.AddAtom(atom)
#
#     sub_atoms = set(sub_atoms)
#     for idx in sub_atoms:
#         a = mol.GetAtomWithIdx(idx)
#         for b in a.GetNeighbors():
#             if b.GetIdx() not in sub_atoms: continue
#             bond = mol.GetBondBetweenAtoms(a.GetIdx(), b.GetIdx())
#             bt = bond.GetBondType()
#             if a.GetIdx() < b.GetIdx(): #each bond is enumerated twice
#                 new_mol.AddBond(atom_map[a.GetIdx()], atom_map[b.GetIdx()], bt)
#
#     return new_mol.GetMol()


def get_sub_mol(mol, sub_atoms, attachments):
    '''
    get sub molecule with atoms whose mapping numbers are in sub_atoms

    :param mol:
    :param sub_atoms: mapping numbers of the sub-molecule
    :param attachments: mapping numbers of attachment atoms
    :return:
    '''
    r_smi = Chem.MolToSmiles(mol, canonical=False)

    mapnum2Hs = {}
    mapnums_to_keep = set(sub_atoms)
    new_mol = Chem.RWMol(mol)
    atoms_to_remove = []
    for atom in new_mol.GetAtoms():
        if atom.GetAtomMapNum() in mapnums_to_keep:
            mapnum2Hs[atom.GetAtomMapNum()] = atom.GetTotalNumHs()
        else:
            atoms_to_remove.append(atom.GetIdx())
    for idx in reversed(atoms_to_remove):
        new_mol.RemoveAtom(idx)

    # remove bond connects two attachment atoms one of which connects only attachment atoms
    bonds_to_remove = []
    for bond in new_mol.GetBonds():
        beg = bond.GetBeginAtom()
        end = bond.GetEndAtom()
        if beg.GetAtomMapNum() in attachments and end.GetAtomMapNum() in attachments:
            beg_nei = [nei.GetAtomMapNum() for nei in beg.GetNeighbors()]
            end_nei = [nei.GetAtomMapNum() for nei in end.GetNeighbors()]
            if set(beg_nei) <= attachments or set(end_nei) <= attachments:
                bonds_to_remove.append((beg.GetIdx(), end.GetIdx()))
    for bond in bonds_to_remove:
        new_mol.RemoveBond(bond[0], bond[1])

    smi = Chem.MolToSmiles(new_mol.GetMol(), canonical=True, kekuleSmiles=True)
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    for atom in mol.GetAtoms():
        # strange rdkit bug
        if atom.GetTotalNumHs() != mapnum2Hs[atom.GetAtomMapNum()]:
            atom.SetNumExplicitHs(mapnum2Hs[atom.GetAtomMapNum()])

    return mol


def copy_edit_mol(mol):
    new_mol = Chem.RWMol(Chem.MolFromSmiles(''))
    for atom in mol.GetAtoms():
        new_atom = copy_atom(atom)
        new_mol.AddAtom(new_atom)

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        bt = bond.GetBondType()
        new_mol.AddBond(a1, a2, bt)
        # if bt == Chem.rdchem.BondType.AROMATIC and not aromatic:
        #    bt = Chem.rdchem.BondType.SINGLE
    return new_mol


def get_clique_mol(mol, atoms):
    smiles = Chem.MolFragmentToSmiles(mol, atoms, kekuleSmiles=True)
    new_mol = Chem.MolFromSmiles(smiles, sanitize=False)
    new_mol = copy_edit_mol(new_mol).GetMol()
    smi1 = Chem.MolToSmiles(new_mol)
    new_mol = sanitize(new_mol)
    smi2 = Chem.MolToSmiles(new_mol)
    # if tmp_mol is not None: new_mol = tmp_mol
    return new_mol


def get_assm_cands(mol, atoms, inter_label, cluster, inter_size):
    atoms = list(set(atoms))
    mol = get_clique_mol(mol, atoms)
    atom_map = [idxfunc(atom) for atom in mol.GetAtoms()]
    mol = set_atommap(mol)
    rank = Chem.CanonicalRankAtoms(mol, breakTies=False)
    rank = {x: y for x, y in zip(atom_map, rank)}

    pos, icls = zip(*inter_label)
    if inter_size == 1:
        cands = [pos[0]] + [x for x in cluster if rank[x] != rank[pos[0]]]

    elif icls[0] == icls[1]:  # symmetric case
        shift = cluster[inter_size - 1:] + cluster[:inter_size - 1]
        cands = zip(cluster, shift)
        cands = [pos] + [(x, y) for x, y in cands if
                         (rank[min(x, y)], rank[max(x, y)]) != (rank[min(pos)], rank[max(pos)])]
    else:
        shift = cluster[inter_size - 1:] + cluster[:inter_size - 1]
        cands = zip(cluster + shift, shift + cluster)
        cands = [pos] + [(x, y) for x, y in cands if (rank[x], rank[y]) != (rank[pos[0]], rank[pos[1]])]

    return cands


def get_inter_label(mol, atoms, inter_atoms, atom_cls):
    new_mol = get_clique_mol(mol, atoms)
    if new_mol.GetNumBonds() == 0:
        inter_atom = list(inter_atoms)[0]
        for a in new_mol.GetAtoms():
            a.SetAtomMapNum(0)
        return new_mol, [(inter_atom, Chem.MolToSmiles(new_mol))]

    inter_label = []
    for a in new_mol.GetAtoms():
        idx = idxfunc(a)
        if idx in inter_atoms and is_anchor(a, inter_atoms):
            inter_label.append((idx, get_anchor_smiles(new_mol, idx)))

    for a in new_mol.GetAtoms():
        idx = idxfunc(a)
        if idx in inter_atoms:
            a.SetAtomMapNum(1)
        elif len(atom_cls[idx]) > 1:
            a.SetAtomMapNum(2)
        else:
            a.SetAtomMapNum(0)

    return new_mol, inter_label


def is_anchor(atom, inter_atoms):
    for a in atom.GetNeighbors():
        if idxfunc(a) not in inter_atoms:
            return True
    return False


def get_anchor_smiles(mol, anchor, idxfunc=idxfunc):
    copy_mol = Chem.Mol(mol)
    for a in copy_mol.GetAtoms():
        idx = idxfunc(a)
        if idx == anchor:
            a.SetAtomMapNum(1)
        else:
            a.SetAtomMapNum(0)

    return get_smiles(copy_mol)


def get_bond_info(mol):
    if mol is None:
        return {}
    bond_info = {}
    for bond in mol.GetBonds():
        a_start = bond.GetBeginAtom().GetAtomMapNum()
        a_end = bond.GetEndAtom().GetAtomMapNum()
        key_pair = sorted([a_start, a_end])
        bond_info[tuple(key_pair)] = [bond.GetBondTypeAsDouble(), bond.GetIdx()]
    return bond_info


def align_kekule_pairs(reac_mol, prod_mol):
    """
    Aligns kekule pairs to ensure unchanged bonds have same bond order in previously aromatic rings.
    将凯库勒对对齐，以确保未改变的键在先前的芳香环中具有相同的键序
    """
    prod_prev = get_bond_info(prod_mol)
    Chem.Kekulize(prod_mol)
    prod_new = get_bond_info(prod_mol)
    reac_prev = get_bond_info(reac_mol)
    Chem.Kekulize(reac_mol)
    reac_new = get_bond_info(reac_mol)
    for bond in prod_new:
        if bond in reac_new and (prod_prev[bond][0] == reac_prev[bond][0]):
            reac_new[bond][0] = prod_new[bond][0]

    reac_mol = Chem.RWMol(reac_mol)
    amap_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in reac_mol.GetAtoms()}
    for bond in reac_new:
        idx1, idx2 = amap_idx[bond[0]], amap_idx[bond[1]]
        bo = reac_new[bond][0]
        reac_mol.RemoveBond(idx1, idx2)
        reac_mol.AddBond(idx1, idx2, BOND_FLOAT_TO_TYPE[bo])

    return reac_mol.GetMol(), prod_mol


def cycle_transform(smiles=None, mol=None, kekuleSmiles=True):
    assert smiles is not None or mol is not None
    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            return None
        smi = Chem.MolToSmiles(mol, kekuleSmiles=kekuleSmiles)
        return smi
    else:
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=kekuleSmiles)
        m = Chem.MolFromSmiles(smiles)
        if not m:
            return None
        # print(smiles)
        return m


def get_attachments(p_mol, r_mol):                              # 找attachment原子
    # p_smi = Chem.MolToSmiles(p_mol, kekuleSmiles=True)
    # r_smi = Chem.MolToSmiles(r_mol, kekuleSmiles=True)
    r_mapnum2idx = get_mapnum2atomidx(r_mol)
    attachments = set()
    for atom in p_mol.GetAtoms():
        mapnum = atom.GetAtomMapNum()
        for nei_atom in atom.GetNeighbors():
            # make sure each bond enumerate only once
            if mapnum > nei_atom.GetAtomMapNum():
                continue
            p_bond = p_mol.GetBondBetweenAtoms(atom.GetIdx(), nei_atom.GetIdx())
            r_bond = r_mol.GetBondBetweenAtoms(r_mapnum2idx[mapnum], r_mapnum2idx[nei_atom.GetAtomMapNum()])
            if (r_bond and r_bond.GetBondType() != p_bond.GetBondType()) or not r_bond:     # 如果反应物和产物对应的原子间的化学键不同，则认为是attachment原子
                attachments.add(atom.GetAtomMapNum())
                attachments.add(nei_atom.GetAtomMapNum())

        p_atom_numHs = atom.GetTotalNumHs()
        r_atom_numHs = r_mol.GetAtomWithIdx(r_mapnum2idx[mapnum]).GetTotalNumHs()
        p_atom_charge = atom.GetFormalCharge()
        r_atom_charge = r_mol.GetAtomWithIdx(r_mapnum2idx[mapnum]).GetFormalCharge()
        if r_atom_numHs != p_atom_numHs or r_atom_charge != p_atom_charge:      # 如果反应物和产物对应的原子上的氢原子数量发生变化，或者化学价发生了变化，也认为是attachment原子
            attachments.add(atom.GetAtomMapNum())

    return attachments


def apply_transform(mol, transform, attachments=None):
    mol = Chem.RWMol(mol)       # 将分子设置为可编辑
    smi = Chem.MolToSmiles(mol, canonical=False)
    bond_int_to_type = {
        1: Chem.rdchem.BondType.SINGLE,
        2: Chem.rdchem.BondType.DOUBLE,
        3: Chem.rdchem.BondType.TRIPLE,
    }
    for idx, atom in enumerate(mol.GetAtoms()):
        if attachments and idx in attachments:
            atom.SetAtomMapNum(1000 + atom.GetAtomMapNum())
    smi = Chem.MolToSmiles(mol, canonical=False)
    for trans, bond_type in transform.items():
        bond = mol.GetBondBetweenAtoms(trans[0], trans[1])
        assert bond
        mol.RemoveBond(trans[0], trans[1])      # 把发生键变化的键去除
        if bond_type > 0:
            mol.AddBond(trans[0], trans[1], bond_int_to_type[bond_type])    # 如果还有键，就添加新的键

    return mol.GetMol()


def dfs_lg(r_mol, r_mapnum2idx, p_mapnum2idx, start_mapnum, visited):
    visited.append(start_mapnum)
    cur_visited = [start_mapnum]
    r_atom = r_mol.GetAtomWithIdx(r_mapnum2idx[start_mapnum])
    children = []
    for atom in r_atom.GetNeighbors():
        mapnum = atom.GetAtomMapNum()
        # start from a reactant only atom connects an unvisited product atom
        # for corner case: two product atoms connect a leaving group, DFS will reach another product atom
        # mark the end product atom as visited
        if start_mapnum not in p_mapnum2idx and mapnum not in visited and mapnum in p_mapnum2idx:
            visited.append(mapnum)
            cur_visited.append(mapnum)
            continue
        # find an unvisited reactant only atom
        if mapnum not in visited and mapnum not in p_mapnum2idx:
            children.append(mapnum)

    children = sorted(children)
    for mapnum in children:
        if mapnum in visited: continue
        cur_visited.extend(dfs_lg(r_mol, r_mapnum2idx, p_mapnum2idx, mapnum, visited))

    return cur_visited
