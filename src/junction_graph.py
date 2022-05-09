
from rdkit import Chem
from itertools import permutations

import chemutils


def dfs(jgraph, path=[], visited_nodes=[], visited_attachments=[], to_visit_attachments=[]):
    if not to_visit_attachments: return
    # next is a pair of (graph node index, attachment)
    current = to_visit_attachments.pop()
    if current in visited_attachments:
        return

    neighbors = jgraph.nodes[current[0]].neighbors.values()
    found = False
    for neighbor in neighbors:
        attachment_pairs = jgraph.edges[neighbor]
        if len(attachment_pairs) == 2:
            print('two attachment_pairs indicates a loop:', attachment_pairs)
        for attachment_pair in attachment_pairs:
            if current[1] == attachment_pair[0]:
                found = True
                edge_path = (neighbor, attachment_pair)
    if not found:
        print('errr')
    assert found
    next_node = edge_path[0][1]

    visited_attachments.append(current)
    visited_attachments.append((edge_path[0][1], edge_path[1][1]))
    if next_node in visited_nodes:
        # loop found, initial 2 indicates this is a loop connection
        path.append((2, edge_path))
        print('*****************************\nloop found:', path[-1])
        return

    visited_nodes.append(next_node)
    attachments = jgraph.nodes[next_node].attachments
    for attach in attachments:
        if attach == edge_path[1][1]:
            continue
        to_visit_attachments.append((next_node, attach))

    # initial 1 indicates this is the forward visit
    path.append((1, edge_path))
    dfs(jgraph, path, visited_nodes, visited_attachments, to_visit_attachments)
    path.append((0, edge_path))


def dfs_reconstruct(motif_vocab, jgraph, path=[], visited_attachments=[], to_visit_attachments=[]):
    if not to_visit_attachments: return
    current = to_visit_attachments.pop()
    if current in visited_attachments:
        return

    motifs = list(motif_vocab.keys())
    start, motif_idx, attachment_idx, start_attachment  = path.pop(0)

    if motif_idx == len(motifs):
        return

    motif = motifs[motif_idx]
    # for uspto_50k, there is only one motif configuration
    motif_attachments = motif_vocab[motif][0]
    assert attachment_idx < len(motif_attachments)
    mol = Chem.MolFromSmiles(motif, sanitize=False)
    motif_attachments_global = motif_attachments.copy()
    cnt_updated = 0
    for atom in mol.GetAtoms():
        jgraph.max_mapnum += 1
        if atom.GetAtomMapNum() in motif_attachments:
            idx = motif_attachments.index(atom.GetAtomMapNum())
            motif_attachments_global[idx] = jgraph.max_mapnum
            cnt_updated += 1
        atom.SetAtomMapNum(jgraph.max_mapnum)
    smi0 = Chem.MolToSmiles(mol)
    if cnt_updated != len(motif_attachments):
        print('errrr motif_attachments update:', motif_attachments, motif_attachments_global)

    cur_attachment = motif_attachments_global[attachment_idx]
    mol_cur = Chem.CombineMols(jgraph.mol_cur, mol)
    smi1 = Chem.MolToSmiles(mol_cur)
    # connect atom (current, cur_attachment)
    mol_cur = Chem.RWMol(mol_cur)
    mapnum_to_idx = chemutils.get_mapnum2atomidx(mol_cur)
    if current not in mapnum_to_idx:
        print('eee current not found:', current, mapnum_to_idx)
    if cur_attachment not in mapnum_to_idx:
        print('eee cur_attachment not found:', cur_attachment, mapnum_to_idx)
    atom1 = mol_cur.GetAtomWithIdx(mapnum_to_idx[current])
    atom2 = mol_cur.GetAtomWithIdx(mapnum_to_idx[cur_attachment])
    for nei in atom2.GetNeighbors():
        bond = mol_cur.GetBondBetweenAtoms(mapnum_to_idx[cur_attachment], mapnum_to_idx[nei.GetAtomMapNum()])
        # skip if already added the bond
        if not mol_cur.GetBondBetweenAtoms(mapnum_to_idx[current], mapnum_to_idx[nei.GetAtomMapNum()]):
            mol_cur.AddBond(mapnum_to_idx[current], mapnum_to_idx[nei.GetAtomMapNum()], bond.GetBondType())

    smi2 = Chem.MolToSmiles(mol_cur)
    if atom1.GetTotalNumHs() != atom2.GetTotalNumHs():
        atom1.SetNumExplicitHs(atom2.GetNumExplicitHs())
    if atom1.GetFormalCharge() != atom2.GetFormalCharge():
        atom1.SetFormalCharge(atom2.GetFormalCharge())
    smi3 = Chem.MolToSmiles(mol_cur)
    mol_cur.RemoveAtom(mapnum_to_idx[cur_attachment])
    smi4 = Chem.MolToSmiles(mol_cur)

    jgraph.mol_cur = mol_cur.GetMol()

    visited_attachments.append(current)
    visited_attachments.append(cur_attachment)
    for attach in motif_attachments_global:
        if attach == cur_attachment:
            continue
        to_visit_attachments.append(attach)

    dfs_reconstruct(motif_vocab, jgraph, path, visited_attachments, to_visit_attachments)


class JunctionNode(object):
    def __init__(self, mol, smiles, attachments, index=0):
        self.mol = mol
        self.smiles = smiles
        self.attachments = sorted(attachments)
        self.attachment_atom_symbols = []
        for attach in self.attachments:
            for atom in mol.GetAtoms():
                if atom.GetAtomMapNum() == attach:
                    self.attachment_atom_symbols.append(atom.GetSymbol())
        self.index = index
        self.neighbors = {}

class JunctionGraph(object):
    def __init__(self, synthon, leaving_group=None):
        '''
        rooted graph structure presents the molecule connection

        :param synthon: synthon is the graph root
        :param leaving_group: leaving group is the complementary of synthon
        '''
        self.synthon = synthon
        self.leaving_group = leaving_group
        self.edges = {}

        # build junction graph root node
        self.nodes = []
        mol_syn = Chem.MolFromSmiles(synthon, sanitize=False)
        attachments_syn = []
        motif_original_to_cano = {}
        for atom in mol_syn.GetAtoms():
            if atom.GetAtomMapNum() > 1000:
                atom.SetAtomMapNum(atom.GetAtomMapNum() % 1000)
                attachments_syn.append(atom.GetAtomMapNum())
            motif_original_to_cano[atom.GetAtomMapNum()] = atom.GetAtomMapNum()
        # for debug
        smi_syn = Chem.MolToSmiles(mol_syn)
        root = JunctionNode(mol_syn, smi_syn, attachments_syn)
        root.original_smiles = smi_syn
        root.mapnum_original_to_cano = motif_original_to_cano
        root.mapnum_cano_to_original = motif_original_to_cano
        self.nodes.append(root)

    def build_junction_graph(self, synthon, leaving_group):
        '''
        build junction graph of the reactant, root node is synthon and leaving group is divided into motifs

        :param synthon:
        :param leaving_group:
        :return:
        '''
        assert self.synthon == synthon

        if len(leaving_group):

            # split lg into motifs that are in motif_vocab
            mol_lg = Chem.MolFromSmiles(leaving_group, sanitize=False)
            for bond in mol_lg.GetBonds():
                if bond.GetBondType == Chem.BondType.AROMATIC:
                    raise ('Unkekulized bond')

            # leaving group is divided into motifs
            smis = chemutils.find_fragments(mol_lg)
            for smi in smis:
                # must make sanitize=True to find match for all smiles
                mol = Chem.MolFromSmiles(smi, sanitize=True)
                attachments = []
                for atom in mol.GetAtoms():
                    if atom.GetAtomMapNum() > 1000:
                        attachments.append(atom.GetIdx())
                    atom.SetAtomMapNum(1)
                smi_cano = Chem.MolToSmiles(mol)
                mol_cano = Chem.MolFromSmiles(smi_cano)
                # use canonical ordering
                match = mol_cano.GetSubstructMatch(mol, useChirality=True)
                assert len(match) == len(mol.GetAtoms())

                motif_cano_to_original = {}
                motif_original_to_cano = {}
                motif_attachments = []
                mol = Chem.MolFromSmiles(smi, sanitize=False)

                for atom, idx in zip(mol.GetAtoms(), match):
                    mapnum = atom.GetAtomMapNum() % 1000
                    mapnum_new = idx + 1
                    if atom.GetIdx() in attachments:
                        mapnum_new += 1000
                        motif_attachments.append(mapnum_new)
                    motif_original_to_cano[mapnum] = mapnum_new
                    motif_cano_to_original[mapnum_new] = mapnum
                    atom.SetAtomMapNum(mapnum_new)

                smi1 = Chem.MolToSmiles(mol)
                motif_in_cano = Chem.MolToSmiles(mol)
                mol_cano = Chem.MolFromSmiles(motif_in_cano, sanitize=False)

                # 构建一个node list，第一个node是root
                node = JunctionNode(mol_cano, motif_in_cano, motif_attachments)
                node.original_smiles = smi
                node.mapnum_original_to_cano = motif_original_to_cano
                node.mapnum_cano_to_original = motif_cano_to_original
                self.nodes.append(node)

            # start to build graph connections
            for idx, node in enumerate(self.nodes):
                node.index = idx
            self.edges = {}
            for idx in range(len(self.nodes)):
                attachments = self.nodes[idx].attachments
                attachments_ori = [self.nodes[idx].mapnum_cano_to_original[i] for i in attachments]
                for k in range(idx + 1, len(self.nodes)):
                    # skip node itself
                    if k == idx: continue
                    k_attachments = self.nodes[k].attachments
                    k_attachments_ori = [self.nodes[k].mapnum_cano_to_original[i] for i in k_attachments]
                    if set(attachments_ori) & set(k_attachments_ori):
                        common = list(set(attachments_ori) & set(k_attachments_ori))
                        # at most two common attachment atoms
                        assert len(common) <= 2
                        common_in_cur = [self.nodes[idx].mapnum_original_to_cano[i] for i in common]
                        common_in_neig = [self.nodes[k].mapnum_original_to_cano[i] for i in common]
                        # update edge information
                        self.edges[tuple((idx, k))] = [(c, n) for c, n in zip(common_in_cur, common_in_neig)]
                        self.edges[tuple((k, idx))] = [(n, c) for n, c in zip(common_in_neig, common_in_cur)]
                        # update neighbor information
                        self.nodes[idx].neighbors[k] = tuple((idx, k))
                        self.nodes[k].neighbors[idx] = tuple((k, idx))

    def dfs_path(self):
        '''
        return the dfs traversal path of the rooted junction graph

        :return:
        '''
        assert self.nodes
        # assert self.edges
        if self.edges:
            mol = Chem.MolFromSmiles(self.leaving_group)
            attach = 0
            for atom in mol.GetAtoms():
                attach += atom.GetAtomMapNum() > 1000
            if attach >= 2 and '.' not in self.leaving_group:
                print('found loop')

            path = []
            visited_attachments = []
            visited_nodes = [0]
            # each to_visit is a pair of (node index, attachments)
            to_visit_attachments = [(0, attach) for attach in self.nodes[0].attachments]
            while len(to_visit_attachments) > 0:
                dfs(self, path=path, visited_nodes=visited_nodes,
                    visited_attachments=visited_attachments,
                    to_visit_attachments=to_visit_attachments)

            self.traversal_path = path
            self.visited_attachments = visited_attachments
            assert len(visited_nodes) == len(self.nodes)
        else:
            path = []
            visited_attachments = []
            self.traversal_path = path
            self.visited_attachments = visited_attachments

        return path

    def reconstruct_molcule_from_path(self):
        mapnum = 0
        mol = Chem.Mol()
        mapnum_local_to_global = []
        mapnum_global_to_local = []
        for k, node in enumerate(self.nodes):
            mapnum_local_to_global.append({})
            mapnum_global_to_local.append({})
            node_mol = Chem.RWMol(node.mol)
            if k == 0:
                for atom in node_mol.GetAtoms():
                    mapnum = max(mapnum, atom.GetAtomMapNum())
                    mapnum_local_to_global[-1][atom.GetAtomMapNum()] = atom.GetAtomMapNum()
                    mapnum_global_to_local[-1][atom.GetAtomMapNum()] = atom.GetAtomMapNum()
            else:
                for atom in node_mol.GetAtoms():
                    mapnum_local = atom.GetAtomMapNum()
                    mapnum += 1
                    atom.SetAtomMapNum(mapnum)
                    mapnum_local_to_global[-1][mapnum_local] = atom.GetAtomMapNum()
                    mapnum_global_to_local[-1][atom.GetAtomMapNum()] = mapnum_local
            mol = Chem.CombineMols(mol, node_mol)

        # for debug
        smi = Chem.MolToSmiles(mol, canonical=False)
        mapnum_to_idx = {}
        for atom in mol.GetAtoms():
            mapnum_to_idx[atom.GetAtomMapNum()] = atom.GetIdx()
        mol_cur = Chem.RWMol(mol)
        atoms_to_remove = []
        for path in self.traversal_path:
            direction, (edge, attachments) = path
            if direction == 0:
                continue
            elif direction == 1:
                mapnum1 = mapnum_local_to_global[edge[0]][attachments[0]]
                mapnum2 = mapnum_local_to_global[edge[1]][attachments[1]]
            elif direction == 2:
                mapnum1 = mapnum_local_to_global[edge[1]][attachments[1]]
                mapnum2 = mapnum_local_to_global[edge[0]][attachments[0]]

            # connect attachment atoms
            atom1 = mol_cur.GetAtomWithIdx(mapnum_to_idx[mapnum1])
            atom2 = mol_cur.GetAtomWithIdx(mapnum_to_idx[mapnum2])
            for nei in atom2.GetNeighbors():
                bond = mol_cur.GetBondBetweenAtoms(mapnum_to_idx[mapnum2], mapnum_to_idx[nei.GetAtomMapNum()])
                # skip if already added the bond
                if not mol_cur.GetBondBetweenAtoms(mapnum_to_idx[mapnum1], mapnum_to_idx[nei.GetAtomMapNum()]):
                    mol_cur.AddBond(mapnum_to_idx[mapnum1], mapnum_to_idx[nei.GetAtomMapNum()], bond.GetBondType())

            smi = Chem.MolToSmiles(mol_cur)
            if edge[0] == 0:
                if atom1.GetTotalNumHs() != atom2.GetTotalNumHs():
                    atom1.SetNumExplicitHs(atom2.GetNumExplicitHs())
                if atom1.GetFormalCharge() != atom2.GetFormalCharge():
                    atom1.SetFormalCharge(atom2.GetFormalCharge())

            atoms_to_remove.append(mapnum_to_idx[mapnum2])

        smi = Chem.MolToSmiles(mol_cur)
        atoms_to_remove = sorted(atoms_to_remove, reverse=True)
        for idx in atoms_to_remove:
            mol_cur.RemoveAtom(idx)
        smi = Chem.MolToSmiles(mol_cur)
        [atom.SetAtomMapNum(0) for atom in mol_cur.GetAtoms()]
        try:
            mol_cur = mol_cur.GetMol()
            for atom in mol_cur.GetAtoms():
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
                atom.SetNumExplicitHs(0)
                # atom.SetAtomMapNum(0)
            for bond in mol_cur.GetBonds():
                bond.SetStereo(Chem.BondStereo.STEREOANY)
            Chem.SanitizeMol(mol_cur)
            # Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ADJUSTHS)
            smi_cano = Chem.MolToSmiles(mol_cur, kekuleSmiles=False, canonical=True)
            # smi_cano = Chem.CanonSmiles(smi_cano)
        except:
            smi_cano = None

        if smi_cano != self.reactant_cano:
            print('reconstruct molecule fail.')

        # reactant_mol = Chem.MolFromSmiles(self.reactant)
        # Chem.Kekulize(reactant_mol)
        # for atom in reactant_mol.GetAtoms():
        #     atom.SetAtomMapNum(0)
        #     atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        #     # atom.SetNumExplicitHs(0)
        # Chem.SanitizeMol(reactant_mol)
        # reactant_smi = Chem.MolToSmiles(reactant_mol, kekuleSmiles=False, canonical=True)

        # Flag = False
        # reactants = smi_cano.split('.')
        # for i in permutations(reactants, len(reactants)):
        #     tmp = '.'.join(i)
        #     if tmp == self.reactant_cano:
        #         Flag = True
        #         smi_cano = tmp
        #         break
        # if not Flag:
        #     print('reconstruct molecule fail.')

        return smi_cano

    def build_transformation_path(self, motifs):
        # skip the root node
        for node in self.nodes[1:]:
            if node.smiles in motifs:
                node.motif_idx = motifs.index(node.smiles)
            else:
                print('motif_vocab does not have motif', node.smiles)
                node.motif_idx = len(motifs)
        self.transformation_path = []
        for path in self.traversal_path:
            if path[0] == 0:
                continue
            elif path[0] == 1:
                (cur_node, next_node), attachments = path[1]
                motif_idx = self.nodes[next_node].motif_idx
                config_attachments = sorted(list(self.nodes[next_node].attachments))
                assert attachments[1] in config_attachments
                attach_idx = config_attachments.index(attachments[1])
                tp = (cur_node == 0, motif_idx, attach_idx, attachments[0])
                self.transformation_path.append(tp)
            else:
                raise ValueError('wrong junction graph traversal path indicator')

    def decode_transformation(self, motif_vocab, transformation_path):
        """
        Decode reactants from starting product molecule and encoded transformation

        :param motif_vocab:
        :param transformation_path:
        :return: decoded reactants molecule
        """
        assert len(self.nodes) == 1

        max_mapnum = 0
        for atom in self.nodes[0].mol.GetAtoms():
            max_mapnum = max(max_mapnum, atom.GetAtomMapNum())
        self.max_mapnum = max_mapnum
        self.mol_cur = self.nodes[0].mol
        to_visit_attachments = list(self.nodes[0].attachments)
        visited_attachments = []
        path = list(transformation_path)
        while len(to_visit_attachments) > 0:
            dfs_reconstruct(
                motif_vocab, self, path=path,
                visited_attachments=visited_attachments,
                to_visit_attachments=to_visit_attachments)

        smi1 = Chem.MolToSmiles(self.mol_cur)
        for atom in self.mol_cur.GetAtoms():
            atom.SetAtomMapNum(0)
        smi2 = Chem.MolToSmiles(self.mol_cur)
        try:
            Chem.SanitizeMol(self.mol_cur)
        except:
            pass
        cur_smi = Chem.MolToSmiles(self.mol_cur)
        return cur_smi


if __name__ == "__main__":
    reactant = '[NH2:3][C:4]1=[CH:5][CH:6]=[C:7]([O:8][C:9]2=[CH:10][CH:11]=[N:12][C:13]3=[C:17]2[CH:16]=[CH:15][NH:14]3)[C:18]([F:19])=[CH:20]1.[O:1]=[C:2]([C:21]([F:22])([F:23])[F:24])[O:27][C:26](=[O:25])[C:28]([F:29])([F:30])[F:31]'
    reactant_cano = 'Nc1ccc(Oc2ccnc3[nH]ccc23)c(F)c1.O=C(OC(=O)C(F)(F)F)C(F)(F)F'

    product = '[O:1]=[C:2]([NH:3][C:4]1=[CH:5][CH:6]=[C:7]([O:8][C:9]2=[CH:10][CH:11]=[N:12][C:13]3=[C:17]2[CH:16]=[CH:15][NH:14]3)[C:18]([F:19])=[CH:20]1)[C:21]([F:22])([F:23])[F:24]'
    synthon = '[C:4]1([NH:1003])=[CH:5][CH:6]=[C:7]([O:8][C:9]2=[CH:10][CH:11]=[N:12][C:13]3=[C:17]2[CH:16]=[CH:15][NH:14]3)[C:18]([F:19])=[CH:20]1.[O:1]=[C:1002][C:21]([F:22])([F:23])[F:24]'
    lg_configuration = '[C:1002][O:27][C:26](=[O:25])[C:28]([F:29])([F:30])[F:31].[NH2:1003]'

    jgraph = JunctionGraph(synthon, lg_configuration)
    jgraph.build_junction_graph(jgraph.synthon, jgraph.leaving_group)
    jgraph.reactant = reactant
    jgraph.product = product
    jgraph.reactant_cano = reactant_cano

    path = jgraph.dfs_path()
    print(path)

    jgraph.reconstruct_molcule_from_path()
