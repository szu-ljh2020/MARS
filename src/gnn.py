# this file is adapted from https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/model.py

import torch
from copy import deepcopy
from torch import nn
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool, GlobalAttention, Set2Set
from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.data.batch import Batch
from MAT import MATNet_aug as MAT_aug

from gnn_zoo import GNN
from beam_search_node import BeamSearchNode, PriorityQueue
from loss_function import Mish, FocalLoss
from utils import get_submol_by_edits, string2dict, string2list


class RNN_model(torch.nn.Module):
    def __init__(self, num_layer, gnn_num_layer, emb_dim, atom_feat_dim, bond_feat_dim, JK="last", drop_ratio=0,
                 graph_pooling="mean",
                 gnn_type="gin", pe=False, pe_dim=8):
        super(RNN_model, self).__init__()
        embedding_dim = emb_dim  # 512
        hidden_size = emb_dim  # 512
        gnn_feat_dim = emb_dim  # 512
        n_layers = num_layer  # 6
        self.rnn = nn.GRU(input_size=embedding_dim,
                          hidden_size=hidden_size,
                          num_layers=n_layers,
                          batch_first=True)
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.mish = Mish()  # x * tanh(softplus(x))

        # 8 states, 0: start, 1: bond transformation, 2: start of atom transformation
        # 3: continue atom transformation, 4: transition to motif generation
        # 6: motif generation start from synthon attachment
        # 5: continue motif generation from non-synthon, 7: padding
        self.embedding_state = nn.Embedding(num_embeddings=8, embedding_dim=embedding_dim)
        self.embedding_atom = nn.Linear(in_features=gnn_feat_dim, out_features=embedding_dim)
        self.embedding_bond = nn.Linear(in_features=gnn_feat_dim, out_features=embedding_dim)
        self.embedding_bond_type = nn.Embedding(num_embeddings=4, embedding_dim=embedding_dim)
        self.embedding_motif = nn.Embedding(num_embeddings=300, embedding_dim=embedding_dim)
        self.embedding_motif_attach_idx = nn.Embedding(num_embeddings=4, embedding_dim=embedding_dim)  # 这个是啥
        self.embedding_mol = nn.Sequential(
            nn.Linear(in_features=gnn_feat_dim, out_features=hidden_size * n_layers),
            Mish(),
            nn.Dropout(p=0.3)
        )

        # concatenate RNN output and predicted bond presentation
        self.bond_change = nn.Sequential(
            nn.Linear(in_features=gnn_feat_dim + hidden_size + embedding_dim, out_features=hidden_size),
            Mish(),
            nn.Dropout(0.4),
            nn.Linear(in_features=hidden_size, out_features=1)
        )
        self.MLP_edge_type = nn.Sequential(
            nn.Linear(in_features=gnn_feat_dim + hidden_size + embedding_dim, out_features=hidden_size),
            Mish(),
            nn.Linear(in_features=hidden_size, out_features=4)
        )
        self.MLP_atom = nn.parameter.Parameter(torch.randn(gnn_feat_dim + hidden_size, embedding_dim))

        self.MLP_motif = nn.Sequential(
            nn.Linear(in_features=gnn_feat_dim + hidden_size, out_features=hidden_size),
            Mish(),
            nn.Dropout(0.4),
            nn.Linear(in_features=hidden_size, out_features=211)
        )
        self.MLP_motif_attach_idx = nn.Sequential(
            nn.Linear(in_features=gnn_feat_dim + hidden_size + embedding_dim, out_features=hidden_size),
            Mish(),
            nn.Linear(in_features=hidden_size, out_features=4)
        )

        self.bce = nn.BCEWithLogitsLoss(reduction="sum")
        self.ce = nn.CrossEntropyLoss(reduction='sum')

        self.gnn = GNN_graphpred(gnn_num_layer, emb_dim, atom_feat_dim, bond_feat_dim,
                                 JK=JK, drop_ratio=drop_ratio,
                                 graph_pooling=graph_pooling, gnn_type=gnn_type, pe=pe, pe_dim=pe_dim)

    def embed_transform(self, nodes, edges, transform, synthon):
        # when state not in [1, 5, 6] synthon is None while inference, other wise synthon is NOT None!
        transform = torch.LongTensor(transform).to(nodes.device)
        state = transform[0].unsqueeze(0)
        state_emb = self.embedding_state(state)
        if state in [0, 7]:
            pass
        elif state == 1:
            # use synthon graph embedding
            state_emb += synthon
            _, bond_idx, new_type = transform
            edge_type_emb = self.embedding_bond_type(new_type)
            edge_emb = self.embedding_bond(edges[bond_idx])
            state_emb += edge_type_emb + edge_emb
        elif state in [6]:
            # use synthon substructure graph embedding
            state_emb += synthon
            # TODO: use intermediate graph embedding and previous lg embedding
            _, atom_idx = transform
            atom_emb = self.embedding_atom(nodes[atom_idx])
            state_emb += atom_emb
        elif state == 5:
            # use synthon substructure graph embedding
            state_emb += synthon
            # TODO: use intermediate graph embedding and previous lg embedding
            _, motif_idx, attachment_idx = transform
            motif_emb = self.embedding_motif(motif_idx)
            motif_attach_idx_emb = self.embedding_motif_attach_idx(attachment_idx)
            state_emb += motif_emb + motif_attach_idx_emb
        else:
            raise ValueError('unknown rnn transformation state: {}'.format(state))

        return state_emb

    def embedding(self, nodes_list, edges_list, inputs, synthons=None, synthon_pe=None, typed=False):
        if synthons is not None:
            if synthon_pe is not None:
                _, _, _, synthon_graph_embedding = self.gnn(synthons, pe=synthon_pe, typed=typed)
            else:
                _, _, _, synthon_graph_embedding = self.gnn(synthons, typed=typed)
        else:
            synthon_graph_embedding = None
        # if not isinstance(inputs[0][0], list):
        #     inputs = [inputs]
        # pad inputs and targets
        input_len = [len(inp) for inp in inputs]
        max_len = max(input_len)
        inputs_embedding = []
        # the k-th sample in a batch
        for k, input in enumerate(inputs):
            nodes = nodes_list[k]
            edges = edges_list[k]
            input_embedding = []
            if synthon_graph_embedding is not None:
                synthon = synthon_graph_embedding[k]
            else:
                synthon = None
            for transform in input:
                state_emb = self.embed_transform(nodes, edges, transform, synthon)
                input_embedding.append(state_emb)

            while len(input_embedding) < max_len:
                input_embedding.append(self.embedding_state(torch.tensor([7]).to(nodes.device)))
            input_embedding = torch.cat(input_embedding, dim=0)
            inputs_embedding.append(input_embedding)

        inputs_embedding = torch.stack(inputs_embedding, dim=0)
        return inputs_embedding

    def split_node_edge(self, nodes, self_edges, edges, atom_len, bond_len):
        # arrange nodes and edges for each molecule separately
        '''
        cause the nodes, edges of different mols in a batch are concatenated, so this function
        aims to split those nodes and edges into different mols, and get the mean pooling of nodes as
        the mol embedding.
        Args:
            nodes:
            self_edges:
            edges:
            atom_len:
            bond_len:
        Returns:
        '''
        mols_list, nodes_list, self_edges_list, edges_list = [], [], [], []
        atom_pos = 0
        for alen in atom_len:
            nodes_cur = nodes[atom_pos:(atom_pos + alen)]
            self_edges_list.append(self_edges[atom_pos:(atom_pos + alen)])
            mols_list.append(nodes_cur.mean(dim=0, keepdim=True))
            nodes_list.append(nodes_cur)
            atom_pos += alen
        assert atom_pos == nodes.size(0)
        bond_pos = 0
        for blen in bond_len:
            edges_list.append(edges[bond_pos:(bond_pos + blen)])
            bond_pos += blen
        assert bond_pos == edges.size(0)

        return mols_list, nodes_list, self_edges_list, edges_list

    def rnn_training(self, typed=False):
        mols_list, nodes_list, self_edges_list, edges_list = self.split_node_edge(
            self.data['nodes'],
            self.data['self_edges'],
            self.data['edges'],
            self.data['atom_len'],
            self.data['bond_len'],
        )

        edges_list = [torch.cat((edge, self_edge), dim=0) for edge, self_edge in zip(edges_list, self_edges_list)]

        mols_embedding = torch.cat(mols_list, dim=0)
        hidden = self.embedding_mol(mols_embedding)
        hidden = hidden.view(self.n_layers, -1, self.hidden_size)
        inputs_embedding = self.embedding(nodes_list, edges_list, self.data['inputs'], self.data['synthon'],
                                          self.data['synthon_pe'], typed=typed)  # input(state, transform) embedding

        # TODO： use node_representation, edge_representation to predict bond idx or atom idx instead of the whole mol_representation
        output, hidden = self.rnn(inputs_embedding, hidden)


        target_pred_res = []
        loss_list = [0 for _ in self.data['targets']]
        loss = 0
        for k, target in enumerate(self.data['targets']):  # 遍历batch中的所有分子
            edges = edges_list[k]
            target_pred = []
            for j, transform in enumerate(target):  # 每一个分子有一个rnn_target
                rnn_vec = output[k, j].unsqueeze(0)
                # concatenate mol embedding with the rnn output vector
                rnn_vec = torch.cat((mols_list[k], rnn_vec), dim=1)
                state_pred = self.MLP_state(rnn_vec)
                transform = torch.LongTensor(transform).to(rnn_vec.device).view(-1, 1)
                state = transform[0]
                loss_list[k] += self.ce(state_pred, state)

                state_pred = torch.argmax(state_pred, dim=1)
                state = transform[0].item()
                if state in [4, 7]:
                    target_pred.append((state_pred[0].item(),))
                    pass
                elif state == 1:
                    _, bond_idx, new_type = transform
                    edges_emb = self.embedding_bond(edges)
                    bond_idx_pred = self.bond_change(
                        torch.cat((rnn_vec.repeat((edges_emb.shape[0], 1)), edges_emb), dim=1))
                    label = torch.zeros((edges_emb.shape[0], 1), device=edges_emb.device)
                    label[bond_idx] = 1
                    loss_list[k] += self.bce(bond_idx_pred, label)

                    bond_emb = edges_emb[bond_idx]
                    edge_type_pred = self.MLP_edge_type(torch.cat((rnn_vec, bond_emb), dim=1))
                    loss_list[k] += self.ce(edge_type_pred, new_type)

                    bond_idx_pred = torch.argmax(bond_idx_pred, dim=0)
                    edge_type_pred = torch.argmax(edge_type_pred, dim=1)
                    target_pred.append((state_pred[0].item(), bond_idx_pred[0].item(), edge_type_pred[0].item()))

                elif state in [5, 6]:
                    _, motif_idx, attachment_idx = transform
                    motif_pred = self.MLP_motif(rnn_vec)
                    loss_list[k] += self.ce(motif_pred, motif_idx)

                    motif_emb = self.embedding_motif(motif_idx)
                    attachment_idx_pred = self.MLP_motif_attach_idx(torch.cat((rnn_vec, motif_emb), dim=1))
                    loss_list[k] += self.ce(attachment_idx_pred, attachment_idx)

                    motif_pred = torch.argmax(motif_pred, dim=1)
                    attachment_idx_pred = torch.argmax(attachment_idx_pred, dim=1)
                    target_pred.append((state_pred[0].item(), motif_pred[0].item(), attachment_idx_pred[0].item()))

                else:
                    raise ValueError('unknown rnn transformation state: {}'.format(state))

            second_phase_idx = target.index([4]) + 1
            target_res = [tuple(t) == tp for t, tp in zip(target, target_pred)]
            target_phase1_res = target_res[:second_phase_idx]
            target_phase2_res = target_res[second_phase_idx:]
            target_pred_res.append((False not in target_phase1_res, False not in target_phase2_res))
            loss += loss_list[k]

        return loss, target_pred_res

    def forward(self, batch, typed=False, motif_vocab=None, motif_masks=None, beam_size=1, epoch=0, device=None):

        # TODO: new steps
        '''
        step 1: predict state1 via gnn using product as input                      ---> product node representation, product mol representation, reaction center predictions
        step 2: apply reaction centers(gt or predict) on product to get synthon    ---> synthon graph(preprocess when teacher-forcing)
        step 3: apply gnn on synthon                                               ---> synthon node representation, synthon mol representation
        '''
        # if training mode
        if self.training or beam_size == 1:
            batch, batch_synthon = batch
            batch.patomidx2mapnum = string2dict(batch.patomidx2mapnum)
            batch.synthon_attachment_indexes = string2list(batch.synthon_attachment_indexes)
            batch.edge_transformations = string2list(batch.edge_transformations)
            batch.atom_transformations = string2list(batch.atom_transformations)
            # TODO： random sign flip pe
            # TODO: add pe to node feature
            # 1. get init hidden state for rnn
            node_representation, self_edge_representation, edge_representation, graph_representation = self.gnn(
                batch, typed, motif_vocab=motif_vocab, motif_masks=motif_masks, epoch=epoch, pe=batch.pe)

            self.data = {
                'nodes': node_representation,
                'self_edges': self_edge_representation,
                'edges': edge_representation,
                'inputs': batch.rnn_input,
                'targets': batch.rnn_target,
                'atom_len': batch.atom_len,
                'bond_len': batch.edge_len,
                'bondidx2atomidx': batch.bondidx2atomidx,
                'attachments_list': batch.synthon_attachment_indexes,
                'atom_symbols': batch.atom_symbols,
                'atomidx2mapnums': batch.patomidx2mapnum,
                'motif_vocab': motif_vocab,
                'motif_masks': motif_masks,
                'synthon': batch_synthon,
                'synthon_pe': batch_synthon.pe
            }
            # 2. run gnn model
            loss, pred_res = self.rnn_training(typed)
        else:
            batch.patomidx2mapnum = string2dict(batch.patomidx2mapnum)
            batch.atom_transformations = string2list(batch.atom_transformations)
            batch.edge_transformations = string2list(batch.edge_transformations)
            batch.synthon_attachment_indexes = string2list(batch.synthon_attachment_indexes)

            # 1. get init hidden state for rnn
            node_representation, self_edge_representation, edge_representation, graph_representation = self.gnn(
                batch, typed, motif_vocab=motif_vocab, motif_masks=motif_masks, epoch=epoch, pe=batch.pe)
            # 2
            output = self.beam_deocde(
                beam_size, node_representation, self_edge_representation, edge_representation, batch.rnn_input,
                batch.rnn_target, batch.atom_len, batch.edge_len, batch.bondidx2atomidx,
                batch.synthon_attachment_indexes, batch.atom_symbols, batch.patomidx2mapnum, motif_vocab,
                motif_masks, batch.product, device, typed)
            if output != None:
                logprob, edge_transformations, atom_transformations, transformation_paths, targets = output

                rank = -1
                for k in range(len(edge_transformations)):
                    etk = set(edge_transformations[k])
                    etkgt = set([tuple(et) for et in batch.edge_transformations[0]])
                    if set(edge_transformations[k]) != set([tuple(et) for et in batch.edge_transformations[0]]):
                        continue
                    atk = set(atom_transformations[k])
                    atkgt = set(batch.atom_transformations[0])
                    if set(atom_transformations[k]) != set(batch.atom_transformations[0]):
                        continue
                    tp = transformation_paths[k]
                    tpgt = batch.junction_graph[0].transformation_path
                    if transformation_paths[k] == batch.junction_graph[0].transformation_path:
                        rank = k
                        break

                return rank, logprob, edge_transformations, atom_transformations, transformation_paths, targets
            else:
                return None
        return loss, pred_res

    def beam_deocde(self, beam_size, nodes, self_edges, edges, inputs, targets, atom_len, bond_len, bondidx2atomidx,
                    attachments_list, atom_symbols, atomidx2mapnums, motif_vocab, motif_masks, products, device,
                    typed=False):

        # beam search currently only works for batch size 1
        assert len(atom_len) == 1
        mols_list, nodes_list, self_edges_list, edges_list = self.split_node_edge(nodes, self_edges, edges, atom_len,
                                                                                  bond_len)
        edges_list = [torch.cat((edge, self_edge), dim=0) for edge, self_edge in zip(edges_list, self_edges_list)]
        mols_embedding = torch.cat(mols_list, dim=0)
        hidden = self.embedding_mol(mols_embedding).view(self.n_layers, -1, self.hidden_size)

        beam_nodes = [BeamSearchNode(hidden[:, 0], 0.0, [[0, ]], products)]
        # transform path length
        for i in range(12):  # 最多做12步transform
            if not beam_nodes:
                break
            batch_size = len(beam_nodes)
            inputs_cur = [node.input_next for node in beam_nodes]
            synthons = Batch.from_data_list([node.synthon.to(device) for node in beam_nodes]).to(
                device)  # 将多个synthon合成一个大graph
            hidden = [node.h for node in beam_nodes]
            hidden = torch.stack(hidden, dim=1)
            inputs_embedding = self.embedding(nodes_list * batch_size, edges_list * batch_size, inputs_cur, synthons,
                                              typed)
            output, hidden = self.rnn(inputs_embedding, hidden)
            # concatenate mol embedding with the rnn output vector
            output = torch.cat((mols_embedding.repeat(batch_size, 1), output.squeeze(1)), dim=1)
            state_next = self.MLP_state(output)
            state_next = torch.log_softmax(state_next, dim=1)

            candidates = PriorityQueue(max_size=beam_size)
            # batch search
            for k, input_cur in enumerate(inputs_cur):
                state = input_cur[0][0]
                rnn_vec = output[k].unsqueeze(0)
                state_next_cur = state_next[k]
                if state in [0, 1]:  # 0: start // 1: bond transformation
                    edges_emb = self.embedding_bond(edges_list[0])
                    bond_idx_pred = self.bond_change(
                        torch.cat((rnn_vec.repeat((edges_emb.shape[0], 1)), edges_emb), dim=1))
                    bond_idx_pred = torch.log(nn.Sigmoid()(bond_idx_pred))
                    values, indexes = bond_idx_pred.topk(k=min(beam_size, bond_idx_pred.size(0)), dim=0)
                    values = values.squeeze().tolist()
                    indexes = indexes.squeeze().tolist()
                    if indexes is None or values is None:
                        return None
                    for val, bond_idx in zip(values, indexes):
                        if bond_idx < bond_len[0]:
                            bond_emb = edges_emb[bond_idx].unsqueeze(0)
                            edge_type_pred = self.MLP_edge_type(torch.cat((rnn_vec, bond_emb), dim=1))
                            edge_type_pred = torch.log_softmax(edge_type_pred, dim=1)
                            vals, idxes = edge_type_pred.topk(k=min(beam_size, edge_type_pred.size(1)), dim=1)
                            vals = vals.squeeze().tolist()
                            idxes = idxes.squeeze().tolist()
                            for v, edge_type in zip(vals, idxes):
                                node = deepcopy(beam_nodes[k])
                                node.logp += val + v + state_next_cur[1].item()
                                node.h = hidden[:, k]
                                node.targets_predict.append((1, bond_idx, edge_type))
                                node.input_next = [(1, bond_idx, edge_type)]
                                node.edge_transformation.append((bond_idx, edge_type))
                                edge = bondidx2atomidx[0][bond_idx]
                                node.attachments_list.extend(edge)
                                node.synthon = get_submol_by_edits(node.p_smi[0], [bond_idx, edge_type])
                                candidates.add((node.logp, node))
                        else:
                            atom_idx = (bond_idx - bond_len[0]).item()
                            node = deepcopy(beam_nodes[k])
                            node.logp += val + state_next_cur[1].item()
                            node.h = hidden[:, k]
                            node.targets_predict.append((1, bond_idx, 0))
                            node.input_next = [(1, bond_idx, 0)]
                            node.atom_transformation.append(atom_idx)
                            node.attachments_list.append(atom_idx)
                            candidates.add((node.logp, node))

                    # if state_next_cur == 4
                    if len(beam_nodes[k].attachments_list) > 0:
                        node = deepcopy(beam_nodes[k])
                        node.logp += state_next_cur[4].item()
                        node.h = hidden[:, k]
                        node.targets_predict.append((4,))
                        node.attachments_list = sorted(list(set(beam_nodes[k].attachments_list)), reverse=True)
                        node.input_next = [(6, node.attachments_list.pop(0))]
                        candidates.add((node.logp, node))
                        if len(beam_nodes[k].attachments_list) > 1:
                            node = deepcopy(node)
                            node.attachments_list = sorted(list(set(beam_nodes[k].attachments_list)), reverse=False)
                            node.input_next = [(6, node.attachments_list.pop(0))]
                            candidates.add((node.logp, node))

                    candidates.fit_size()

                elif state in [5, 6]:  # motif generation
                    motif_pred = self.MLP_motif(rnn_vec)
                    motif_pred = torch.log_softmax(motif_pred, dim=1)
                    # motif mask
                    if state == 6:
                        atom_idx = input_cur[0][1]
                        symbol = atom_symbols[0][atom_idx]
                    else:
                        _, motif_idx, attachment_idx = input_cur[0]
                        motif = list(motif_vocab.keys())[motif_idx]
                        motif = motif_vocab[motif]
                        symbol = motif[1][attachment_idx]

                    # skip if unknown attachment
                    if symbol not in motif_masks:
                        continue

                    motif_mask = motif_masks[symbol].squeeze().tolist()
                    values, indexes = motif_pred.topk(k=motif_pred.size(1), dim=1)
                    values = values.squeeze().tolist()
                    indexes = indexes.squeeze()
                    for val, motif_idx_pred in zip(values, indexes):
                        if motif_mask[motif_idx_pred.item()] == 0: continue
                        motif_emb = self.embedding_motif(motif_idx_pred).unsqueeze(0)
                        attachment_idx_pred = self.MLP_motif_attach_idx(torch.cat((rnn_vec, motif_emb), dim=1))
                        attachment_idx_pred = torch.log_softmax(attachment_idx_pred, dim=1).squeeze().tolist()
                        motif_idx_pred = motif_idx_pred.item()
                        motif = list(motif_vocab.keys())[motif_idx_pred]
                        motif = motif_vocab[motif]
                        for att_idx in range(len(motif[1])):
                            node = deepcopy(beam_nodes[k])
                            node.logp += val + attachment_idx_pred[att_idx]
                            node.h = hidden[:, k]
                            # attachment mask
                            if state == 6:
                                atom_idx = input_cur[0][1]
                                mapnum = atomidx2mapnums[0][atom_idx]
                            else:
                                mapnum = motif[0][att_idx]

                            node.transformation_paths.append((state == 6, motif_idx_pred, att_idx, mapnum))
                            node.active_attachments += len(motif[1]) - 1
                            if node.active_attachments == 0:
                                if len(node.attachments_list) == 0:
                                    node.input_next = [(7,)]
                                else:
                                    node.input_next = [(6, node.attachments_list.pop(0))]
                                node.targets_predict.append((6, motif_idx_pred, att_idx))
                                node.logp += state_next_cur[6].item()
                            else:
                                node.input_next = [(5, motif_idx_pred, att_idx)]
                                node.targets_predict.append((5, motif_idx_pred, att_idx))
                                node.logp += state_next_cur[5].item()
                            candidates.add((node.logp, node))

                    candidates.fit_size()

                elif state == 7:
                    beam_nodes[k].input_next = [(7,)]
                    candidates.add((beam_nodes[k].logp, beam_nodes[k]))
                else:
                    raise ValueError("unknown decoder state: {}".format(state))

            candidates.fit_size()
            beam_nodes = [val[1] for val in candidates.values]

        logprob, edge_transformations, atom_transformations, transformation_paths, targets = [], [], [], [], []
        for bnode in beam_nodes:
            # skip unfinished decoding sequence
            if bnode.active_attachments > 0 or bnode.input_next[0][0] < 7:
                continue
            logprob.append(bnode.logp)
            edge_transformations.append(bnode.edge_transformation)
            atom_transformations.append(bnode.atom_transformation)
            transformation_paths.append(bnode.transformation_paths)
            targets.append(bnode.targets_predict)
            for idx, transform in enumerate(bnode.transformation_paths):
                if not transform[0]:
                    print('\ntransformation_path 1:', bnode.transformation_paths)

        return logprob, edge_transformations, atom_transformations, transformation_paths, targets

    def from_pretrained(self, model_file, device=0):
        if torch.cuda.is_available():
            self.load_state_dict(torch.load(model_file, map_location='cuda:{}'.format(device)))
        else:
            self.load_state_dict(torch.load(model_file, map_location='cpu'))


class GNN_graphpred(torch.nn.Module):
    """
    Extension of GIN to incorporate edge information by concatenation.

    Args:
        num_layer (int): the number of GNN layers
        emb_dim (int): dimensionality of embeddings
        num_tasks (int): number of tasks in multi-task learning scenario
        drop_ratio (float): dropout rate
        JK (str): last, concat, max or sum.
        graph_pooling (str): sum, mean, max, attention, set2set
        gnn_type: gin, gcn, graphsage, gat

    See https://arxiv.org/abs/1810.00826
    JK-net: https://arxiv.org/abs/1806.03536
    """

    def __init__(self, num_layer, emb_dim, atom_feat_dim, bond_feat_dim, JK="last", drop_ratio=0, graph_pooling="mean",
                 gnn_type="gin", pe=False, pe_dim=8):
        super(GNN_graphpred, self).__init__()
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.JK = JK
        self.emb_dim = emb_dim

        if self.num_layer < 2:
            raise ValueError("Number of GNN layers must be greater than 1.")
        self.gnn_type = gnn_type
        if self.gnn_type == 'MAT_aug':
            self.gnn = MAT_aug(atom_feat_dim, num_layer, emb_dim, h=8, dropout=0.1,
                               lambda_attention=0.3, lambda_distance=0.3, trainable_lambda=False,
                               N_dense=2, leaky_relu_slope=0.0, aggregation_type='none',
                               dense_output_nonlinearity='relu', distance_matrix_kernel='softmax',
                               use_edge_features=True, n_output=1,
                               control_edges=False, integrated_distances=False,
                               scale_norm=False, init_type='uniform', use_adapter=False, n_generator_layers=1)
        else:
            self.gnn = GNN(num_layer, emb_dim, atom_feat_dim, bond_feat_dim, JK, drop_ratio, gnn_type=gnn_type, pe=pe,
                           pe_dim=pe_dim)

        # Different kind of graph pooling
        if graph_pooling == "sum":
            self.pool = global_add_pool
        elif graph_pooling == "mean":
            self.pool = global_mean_pool
        elif graph_pooling == "max":
            self.pool = global_max_pool
        elif graph_pooling == "attention":
            if self.JK == "concat":
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear((self.num_layer + 1) * emb_dim, 1))
            else:
                self.pool = GlobalAttention(gate_nn=torch.nn.Linear(emb_dim, 1))
        else:
            raise ValueError("Invalid graph pooling type.")

        multiply = self.num_layer + 1 if self.JK == "concat" else 1
        self.node_fusion_linear = torch.nn.Sequential(
            torch.nn.Linear(2 * self.emb_dim * multiply, self.emb_dim),
            Mish(),
        )
        self.edge_fusion_linear = torch.nn.Sequential(
            torch.nn.Linear(2 * self.emb_dim, self.emb_dim),
            Mish(),
        )
        self.graph_pooling_linear = torch.nn.Linear(self.emb_dim * multiply, self.emb_dim)
        # RNN model for transformation prediction
        # self.rnn_model = RNN_model(embedding_dim=emb_dim, hidden_size=emb_dim, gnn_feat_dim=emb_dim)

    def forward(self, batch, typed=False, motif_vocab=None, motif_masks=None, beam_size=1, epoch=0, pe=None):
        node_feat = batch.x
        if typed:
            type_feat = batch.type[batch.batch]
            type_feat_onehot = torch.eye(10, dtype=torch.float32).to(node_feat.device)[type_feat - 1]
            node_feat = torch.cat((batch.x, type_feat_onehot), dim=1)
        # print(batch)
        # print('...................................................................................')
        # print(node_feat.size())

        # get node representations by gnn embedding
        if self.gnn_type is 'MAT_aug':
            adj_matrix = dense_to_sparse(batch.edge_index)
            # print(adj_matrix.size())
            batch_mask = torch.sum(torch.abs(node_feat), dim=-1) != 0
            node_representation = self.gnn(node_feat, batch_mask, adj_matrix, adj_matrix, batch.edge_attr)
        else:
            node_representation = self.gnn(node_feat, batch.edge_index, batch.edge_attr, batch.batch, pe)
        graph_representation = self.pool(node_representation, batch.batch)
        # distribute graph representation to its nodes
        graph_to_node = graph_representation[batch.batch]
        # concatenate node presentation and graph representation
        node_representation = torch.cat((node_representation, graph_to_node), dim=1)
        node_representation = self.node_fusion_linear(node_representation)
        # only need one edge for each bond
        edge_representation = self.edge_fusion_linear(
            torch.cat((node_representation[batch.edge_index[0]],
                       node_representation[batch.edge_index[1]]), dim=1))
        edge_representation = edge_representation.view(-1, 2, edge_representation.size(-1)).mean(dim=1)
        self_edge_representation = self.edge_fusion_linear(torch.cat((node_representation, node_representation), dim=1))
        graph_representation = self.graph_pooling_linear(graph_representation)
        return node_representation, self_edge_representation, edge_representation, graph_representation


class BaseModel(torch.nn.Module):
    def __init__(self, num_layer, emb_dim, atom_feat_dim, bond_feat_dim, JK="last", drop_ratio=0, graph_pooling="mean",
                 gnn_type="gin"):
        super(BaseModel, self).__init__()
        self.gnn = GNN_graphpred(num_layer, emb_dim, atom_feat_dim, bond_feat_dim, JK, drop_ratio, graph_pooling,
                                 gnn_type)
        self.rnn = RNN_model(embedding_dim=emb_dim, hidden_size=emb_dim, gnn_feat_dim=emb_dim)

    def from_pretrained(self, model_file, device=0):
        self.load_state_dict(torch.load(model_file, map_location='cuda:{}'.format(device)))

    def forward(self, batch, typed=False, motif_vocab=None, motif_masks=None, beam_size=1, epoch=0):
        # 1. get init hidden state for rnn
        node_representation, self_edge_representation, edge_representation, graph_representation = self.gnn(
            batch, typed, motif_vocab=motif_vocab, motif_masks=motif_masks, epoch=epoch)
        # 2.
        pass


if __name__ == "__main__":
    pass
