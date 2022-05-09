# this file is adapted from https://github.com/snap-stanford/pretrain-gnns/blob/master/chem/finetune.py

import argparse
import json
import numpy as np
import math
import os
import shutil
import torch
torch.cuda.current_device()
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch_geometric
import time

from collections import Counter
from tqdm import tqdm
from torch_geometric.data import DataLoader
from tensorboardX import SummaryWriter

from gnn import GNN, GNN_graphpred, RNN_model
from prepare_mol_graph import MoleculeDataset

from cyclic_lr import *


def train(args, model, device, loader, motif_vocab, motif_masks, optimizer=None, train=True, epoch=1):
    if train:
        model.train()
    else:
        model.eval()

    loss_list = []
    pred_mol_list = []
    pred_phase1_list = []
    pred_phase2_list = []

    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch[0].to(device), batch[1].to(device)
        loss, pred_res = model(batch, args.typed, motif_vocab=motif_vocab, motif_masks=motif_masks, epoch=epoch)
        loss_list.append(loss.item())

        if model.training:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        for (pred_phase1, pred_phase2) in pred_res:
            pred_phase1_list.append(pred_phase1)
            pred_phase2_list.append(pred_phase2)
            pred_mol_list.append(pred_phase2 and pred_phase2)

    loss = np.mean(loss_list)
    mol_acc = np.mean(pred_mol_list)
    phase1_acc = np.mean(pred_phase1_list)
    phase2_acc = np.mean(pred_phase2_list)

    return loss, mol_acc, phase1_acc, phase2_acc


def eval_decoding(args, model, device, dataset, motif_vocab, motif_masks, k):
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    pred_ranks = []
    pred_results = {}
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        with torch.no_grad():
            batch = batch[0].to(device)
            output = model(
                batch, args.typed, motif_vocab, motif_masks, args.beam_size, device=device)
            if output is not None:
                rank, logprob, edge_trans, atom_trans, trans_paths, targets = output
            else:
                continue

        pred_ranks.append(rank)
        pred_results[batch.id[0].item()] = {
            'rank': rank,
            'product': batch.product[0],
            'reactant': batch.junction_graph[0].reactant,
            'logprob': logprob,
            'edge_transform': edge_trans,
            'atom_transform': atom_trans,
            'transform_path': trans_paths,
            'targets': targets,
            'edge_transform_gt': batch.edge_transformations[0],
            'atom_transform_gt': batch.atom_transformations[0],
            'transform_path_gt': batch.junction_graph[0].transformation_path,
            'targets_gt': batch.rnn_target[0],
        }

    res_file = os.path.join(args.filename, '{}_results_{}_{}.txt'.format(args.input_model_file, args.test_set, k))
    np.savetxt(res_file, pred_ranks, delimiter="\n", fmt="%d")
    beam_res_file = os.path.join(args.filename, '{}_beam_result_{}_{}.json'.format(args.input_model_file, args.test_set, k))
    with open(beam_res_file, 'w') as f:
        json.dump(pred_results, f)


def eval(args, model, device, test_dataset, motif_vocab, motif_masks):
    model.eval()

    # test_dataset.process_data_files = test_dataset.process_data_files[:1000]
    res_file = os.path.join(args.filename, '{}_results_{}_{}.txt'.format(args.input_model_file, args.test_set, 0))
    eval_decoding(args, model, device, test_dataset, motif_vocab, motif_masks, 0)
    pred_ranks = np.loadtxt(res_file).tolist()

    cnt = Counter()
    for rank in pred_ranks:
        cnt[rank] += 1
    print(cnt)

    with open(os.path.join(args.filename, '{}_top-{}.txt'.format(args.input_model_file, args.beam_size)), 'w') as f:
        sum = 0
        for rank in range(args.beam_size):
            cnt[rank] += sum
            sum = cnt[rank]
            message = 'Top-{} acc: {}'.format(rank + 1, cnt[rank] / len(pred_ranks))
            print(message)
            f.write(message + '\n')

    return cnt[0] / len(pred_ranks)


def eval_multi_process(args, model, device, test_dataset, motif_vocab, motif_masks):
    model.eval()

    data_chunks = []
    # test_dataset.process_data_files = test_dataset.process_data_files[:1000]
    chunk_size = len(test_dataset.process_data_files) // args.num_processes + 1
    for i in range(0, len(test_dataset.process_data_files), chunk_size):
        data_chunks.append(test_dataset.process_data_files[i:i + chunk_size])

    mp.set_start_method('spawn', force=True)
    model.share_memory()
    processes = []
    results = []
    for k, data_files in enumerate(data_chunks):
        # if k in [4, 5, 6, 7]:
        #     continue
        test_dataset.process_data_files = data_files
        res_file = os.path.join(args.filename, '{}_results_{}_{}.txt'.format(args.input_model_file, args.test_set, k))
        results.append(res_file)
        p = mp.Process(
            target=eval_decoding,
            args=(args, model, device, test_dataset, motif_vocab, motif_masks, k)
        )
        # We first train the model across `num_processes` processes
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    pred_ranks = []
    for file in results:
        pred = np.loadtxt(file).tolist()
        if isinstance(pred, list):
            pred_ranks.extend(pred)
        else:
            pred_ranks.append(pred)
    cnt = Counter()
    for rank in pred_ranks:
        cnt[rank] += 1
    print(cnt)

    with open(os.path.join(args.filename, '{}_top-{}-{}.txt'.format(args.input_model_file, args.test_set, args.beam_size)), 'w') as f:
        sum = 0
        for rank in range(args.beam_size):
            cnt[rank] += sum
            sum = cnt[rank]
            message = 'Top-{} acc: {}'.format(rank + 1, cnt[rank] / len(pred_ranks))
            print(message)
            f.write(message + '\n')

    return cnt[0] / len(pred_ranks)


def train_multiprocess(rank, args, model, device, train_dataset, valid_dataset, test_dataset, motif_vocab, motif_masks):
    logfile = os.path.join(args.filename, 'log_rank{}.csv'.format(rank))
    logfile = open(logfile, 'w', buffering=1)
    logfile.write('epoch,train_loss,train_phase1_acc,train_phase2_acc,valid_loss,valid_phase1_acc,valid_phase2_acc,'
                  'test_loss,test_phase1_acc,test_phase2_acc\n')
    # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
    epochs = args.epochs // args.num_processes
    output_model_file = os.path.join(args.filename, 'model_{}.pt'.format(rank))
    print('output_model_file:', output_model_file)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    cycle_inter = args.cyc_inner
    cycle_inter = cycle_inter // args.num_processes
    optimizer = optim.Adam([{'params': model.parameters()}], lr=args.lr, weight_decay=args.decay)
    scheduler = CosineAnnealingLR_with_Restart(optimizer,
                                               T_max=cycle_inter,
                                               T_mult=1,
                                               model=model,
                                               out_dir=output_model_file,
                                               take_snapshot=False,
                                               eta_min=1e-5)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    max_test_phase2_acc = 0.0
    for epoch in range(1, epochs + 1):
        print("====rank and epoch: ", rank, epoch)
        # val_res = train(args, model, device, val_loader, motif_vocab, motif_masks, optimizer, train=False)

        train_res = train(args, model, device, train_loader, motif_vocab, motif_masks, optimizer, epoch=epoch)
        loss, mol_acc, phase1_acc, phase2_acc = train_res
        # print(
        #     "rank: %d epoch: %d train_loss: %f mol_acc: %f transform_acc: %f phase1_acc: %f phase2_acc: %f phase1_transform_acc: %f phase2_transform_acc: %f" % (
        #     rank, epoch, *train_res))
        print("rank: %d epoch: %d train_loss: %f mol_acc: %f phase1_acc: %f phase2_acc: %f " % (
            rank, epoch, loss, mol_acc, phase1_acc, phase2_acc))

        scheduler.step()
        print("====Evaluation")
        if args.eval_train:
            train_loss, train_mol_acc, train_transform_acc = train(args, model, device, train_loader, motif_vocab,
                                                                   motif_masks, optimizer, train=False, epoch=epoch)
            print("train_loss: %f mol_acc: %f transform_acc: %f" % (train_loss, train_mol_acc, train_transform_acc))
        else:
            print("omit the training accuracy computation")
        torch.save(model.state_dict(), output_model_file)
        output_model_file = os.path.join(args.filename, 'model_{}_{}.pt'.format(rank, epoch))
        torch.save(model.state_dict(), output_model_file)
        val_res = train(args, model, device, val_loader, motif_vocab, motif_masks, optimizer, train=False, epoch=epoch)

        test_res = train(args, model, device, test_loader, motif_vocab, motif_masks, optimizer, train=False,
                         epoch=epoch)

        if test_res[3] > max_test_phase2_acc:
            max_test_phase2_acc = test_res[3]
            output_model_file = os.path.join(args.filename, 'model_{}_max.pt'.format(rank))
            torch.save(model.state_dict(), output_model_file)
        # print(
        #     "rank: %d epoch: %d val_loss: %f mol_acc: %f transform_acc: %f phase1_acc: %f phase2_acc: %f phase1_transform_acc: %f phase2_transform_acc: %f" % (
        #     rank, epoch, *val_res))
        loss, mol_acc, phase1_acc, phase2_acc = val_res
        print("rank: %d epoch: %d val_loss: %f mol_acc: %f phase1_acc: %f phase2_acc: %f " % (
            rank, epoch, loss, mol_acc, phase1_acc, phase2_acc))
        # print(
        #     "rank: %d epoch: %d test_loss: %f mol_acc: %f transform_acc: %f phase1_acc: %f phase2_acc: %f phase1_transform_acc: %f phase2_transform_acc: %f" % (
        #     rank, epoch, *test_res))
        loss, mol_acc, phase1_acc, phase2_acc = test_res
        print("rank: %d epoch: %d test_loss: %f mol_acc: %f phase1_acc: %f phase2_acc: %f " % (
            rank, epoch, loss, mol_acc, phase1_acc, phase2_acc))
        logfile.write(
            str(epoch) + ',' + str(train_res[0]) + ',' + str(train_res[2]) + ',' + str(train_res[3]) + ','
            + str(val_res[0]) + ',' + str(val_res[2]) + ',' + str(val_res[3]) + ','
            + str(test_res[0]) + ',' + str(test_res[2]) + ',' + str(test_res[3]) + '\n')
        logfile.flush()
        print("max_phase2 test acc: ", max_test_phase2_acc)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
    parser.add_argument('--root_dir', type=str, default='',
                        help='root dir')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.0003,
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--lr_scale', type=float, default=1,
                        help='relative learning rate for the feature extraction layer (default: 1)')
    parser.add_argument('--decay', type=float, default=0,
                        help='weight decay (default: 0)')
    parser.add_argument('--num_layer', type=int, default=4,
                        help='number of GNN message passing layers (default: 4).')
    parser.add_argument('--gnn_num_layer', type=int, default=6,
                        help='number of GNN message passing layers (default: 6).')
    parser.add_argument('--emb_dim', type=int, default=512,
                        help='embedding dimensions (default: 300)')
    parser.add_argument('--dropout_ratio', type=float, default=0.5,
                        help='dropout ratio (default: 0.5)')
    parser.add_argument('--graph_pooling', type=str, default="attention",
                        help='graph level pooling (sum, mean, max, set2set, attention)')
    parser.add_argument('--JK', type=str, default="concat",
                        help='how the node features across layers are combined. last, sum, max or concat')
    parser.add_argument('--gnn_type', type=str, default="transformer")
    parser.add_argument('--dataset', type=str, default='data3/USPTO50K',
                        help='root directory of dataset.')
    parser.add_argument('--atom_feat_dim', type=int, default=45, help="atom feature dimension.")
    parser.add_argument('--bond_feat_dim', type=int, default=12, help="bond feature dimension.")
    parser.add_argument('--process_data', action='store_true', default=False,
                        help='if process data to prepare molecule graph data')
    parser.add_argument('--typed', action='store_true', default=False, help='if given reaction types')
    parser.add_argument('--test_only', action='store_true', default=False, help='only evaluate on test data')
    parser.add_argument('--test_set', type=str, default="test")
    parser.add_argument('--multiprocess', action='store_true', default=False, help='train a model with multi process')
    parser.add_argument('--num_processes', type=int, default=4, help='number of processes for multi-process training')
    parser.add_argument('--input_model_file', type=str, default='',
                        help='filename to read the model (if there is any)')
    parser.add_argument('--filename', type=str, default='addsynall_gtn_d512_bz128_cyc200_e400_20211229', help='output filename')
    parser.add_argument('--seed', type=int, default=42, help="Seed for splitting the dataset.")
    parser.add_argument('--runseed', type=int, default=0, help="Seed for minibatch selection, random initialization.")
    parser.add_argument('--eval_train', action='store_true', default=False, help='evaluating training or not')
    parser.add_argument('--num_workers', type=int, default=0, help='number of workers for dataset loading')
    parser.add_argument('--beam_size', type=int, default=10, help='beam search size for rnn decoding')
    parser.add_argument('--cyc_inner', type=int, default=10, help='cyclic train epoches inner')
    parser.add_argument('--pe', action='store_true', default=False, help='use positional encoding')
    parser.add_argument('--pe_dim', type=int, default=8, help='input of positional encoding')

    args = parser.parse_args()

    torch.manual_seed(args.runseed)
    np.random.seed(args.runseed)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.runseed)
    # device = torch.device("cpu")

    # set up dataset
    train_dataset = MoleculeDataset(args.dataset, split='train')
    valid_dataset = MoleculeDataset(args.dataset, split='valid')
    test_dataset = MoleculeDataset(args.dataset, split='test')
    if args.process_data:
        train_dataset.process_data()
        valid_dataset.process_data()
        test_dataset.process_data()
        train_dataset.encode_transformation(train_dataset.motif_vocab)
        valid_dataset.encode_transformation(train_dataset.motif_vocab)
        test_dataset.encode_transformation(train_dataset.motif_vocab)

    print(train_dataset[0])

    if args.typed:
        args.atom_feat_dim += 10
        args.filename = os.path.join('typed', args.filename)

    # set up model
    model = RNN_model(args.num_layer, args.gnn_num_layer, args.emb_dim, args.atom_feat_dim, args.bond_feat_dim,
                      JK=args.JK, drop_ratio=args.dropout_ratio,
                      graph_pooling=args.graph_pooling, gnn_type=args.gnn_type, pe=args.pe, pe_dim=args.pe_dim)
    model.to(device)

    dataset = os.path.basename(args.dataset)
    args.filename = os.path.join(args.root_dir, 'runs', dataset, args.filename)
    os.makedirs(args.filename, exist_ok=True)
    print('log filename:', args.filename)
    if not args.input_model_file == "":
        input_model_file = os.path.join(args.filename, args.input_model_file)
        model.from_pretrained(input_model_file, args.device)
        print("load model from:", input_model_file)

    # if args.input_model_file:
    #     import collections
    #     nets = []
    #     for i in range(-5, 5):
    #         num = args.input_model_file
    #         input_model_file = os.path.join(args.filename, "model_e{}.pt".format(int(num[7:][:-3]) + i))
    #         net = RNN_model(args.num_layer, args.gnn_num_layer, args.emb_dim, args.atom_feat_dim, args.bond_feat_dim,
    #                         JK=args.JK, drop_ratio=args.dropout_ratio,
    #                         graph_pooling=args.graph_pooling, gnn_type=args.gnn_type,
    #                         pe=args.pe, pe_dim=args.pe_dim)
    #         net.from_pretrained(input_model_file)
    #         nets.append(net)
    #     worker_state_dict = [x.state_dict() for x in nets]
    #     weight_keys = list(worker_state_dict[0].keys())
    #     print(worker_state_dict[0].keys())
    #     fed_state_dict = collections.OrderedDict()
    #     for key in weight_keys:
    #         key_sum = 0
    #         for i in range(len(nets)):
    #             key_sum = key_sum + worker_state_dict[i][key]
    #         fed_state_dict[key] = torch.true_divide(key_sum, len(nets))
    #     model.load_state_dict(fed_state_dict)

    motif_vocab = train_dataset.motif_vocab
    motif_masks = train_dataset.motif_masks
    if args.test_only:
        t1 = time.time()
        if args.test_set == 'test':
            print("evaluate on test data only")
            acc = eval_multi_process(args, model, device, test_dataset, motif_vocab, motif_masks)
            # acc = eval(args, model, device, test_dataset, motif_vocab, motif_masks)
        elif args.test_set == 'valid':
            print("evaluate on valid data only")
            acc = eval_multi_process(args, model, device, valid_dataset, motif_vocab, motif_masks)
        elif args.test_set == 'train':
            print("evaluate on train data only")
            acc = eval_multi_process(args, model, device, train_dataset, motif_vocab, motif_masks)
        # acc = eval(args, model, device, test_dataset, motif_vocab, motif_masks)
        t2 = time.time()
        print("prediction acc:", acc)
        print(t2 - t1)
        exit(1)

    if args.multiprocess:
        # if 0:
        # ctx = mp.get_context("spawn")
        # ctx.set_start_method("spawn")
        mp.set_start_method('spawn', force=True)
        model.share_memory()  # gradients are allocated lazily, so they are not shared here
        processes = []
        for rank in range(args.num_processes):
            p = mp.Process(
                target=train_multiprocess,
                args=(rank, args, model, device, train_dataset, valid_dataset, test_dataset, motif_vocab, motif_masks)
            )
            # We first train the model across `num_processes` processes
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    else:
        logfile = os.path.join(args.filename, 'log.csv')
        logfile = open(logfile, 'w', buffering=1)
        logfile.write('epoch,train_loss,train_phase1_acc,train_phase2_acc,valid_loss,valid_phase1_acc,valid_phase2_acc,'
                      'test_loss,test_phase1_acc,test_phase2_acc\n')
        output_model_file = os.path.join(args.filename, 'model_e{}.pt')
        print('output_model_file:', output_model_file)
        # set up optimizer
        # optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

        cycle_inter = args.cyc_inner
        optimizer = optim.Adam([{'params': model.parameters()}], lr=args.lr, weight_decay=args.decay)
        scheduler = CosineAnnealingLR_with_Restart(optimizer,
                                                   T_max=cycle_inter,
                                                   T_mult=1,
                                                   model=model,
                                                   out_dir=output_model_file,
                                                   take_snapshot=False,
                                                   eta_min=1e-5)

        print(optimizer)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

        for epoch in range(1, args.epochs + 1):
            print("====epoch " + str(epoch))
            # lr = optimizer.param_groups[0]['lr']
            train_loss, train_mol_acc, train_phase1_acc, train_phase2_acc = train(args, model, device, train_loader, motif_vocab, motif_masks,
                                                          optimizer, train=True, epoch=epoch)
            scheduler.step()
            print('loss: {}, mol_acc: {}, phase1_acc: {}, phase2_acc: {}'.format(float(train_loss / args.batch_size), train_mol_acc,
                                                                                 train_phase1_acc, train_phase2_acc))
            torch.save(model.state_dict(), output_model_file.format(epoch))

            val_res = train(args, model, device, val_loader, motif_vocab, motif_masks, optimizer, train=False,
                            epoch=epoch)
            test_res = train(args, model, device, test_loader, motif_vocab, motif_masks, optimizer, train=False,
                             epoch=epoch)
            loss, mol_acc, phase1_acc, phase2_acc = val_res
            print("epoch: %d val_loss: %f mol_acc: %f phase1_acc: %f phase2_acc: %f " % (epoch, loss, mol_acc, phase1_acc, phase2_acc))
            loss, mol_acc, phase1_acc, phase2_acc = test_res
            print("epoch: %d test_loss: %f mol_acc: %f phase1_acc: %f phase2_acc: %f " % (epoch, loss, mol_acc, phase1_acc, phase2_acc))

            logfile.write(str(epoch) + ',' + str(float(train_loss / args.batch_size)) + ',' + str(train_phase1_acc) + ',' + str(
                train_phase2_acc) + ','
                          + str(val_res[0]) + ',' + str(val_res[2]) + ',' + str(val_res[3]) + ','
                          + str(test_res[0]) + ',' + str(test_res[2]) + ',' + str(test_res[3]) + '\n')
            logfile.flush()


if __name__ == "__main__":
    main()
