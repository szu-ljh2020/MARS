import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def draw_log_pic(f, is_multiprocess, filename, rank=None):
    if rank is not None: filename += '_' + str(rank)
    lines = [line.strip() for line in f.readlines()]
    epoch = len(lines)
    loss = [[None] * epoch for _ in range(3)]
    phase1_acc = [[None] * epoch for _ in range(3)]
    phase2_acc = [[None] * epoch for _ in range(3)]

    for ep, line in enumerate(lines):
        items = line.strip(',')
        anchor = 1
        if is_multiprocess:
            anchor = 2
        for i in range(3):
            loss[i][ep]       = float(items[3 * i + anchor])
            phase1_acc[i][ep] = float(items[3 * i + anchor + 1])
            phase2_acc[i][ep] = float(items[3 * i + anchor + 2])

    # red dashes, blue squares and green triangles
    idx = list(range(epoch))

    plt.figure(figsize=(15, 4))
    plt.subplot(131)
    plt.plot(idx, loss[0], 'r--', label='train')
    plt.plot(idx, loss[1], 'bs', label='valid')
    plt.plot(idx, loss[2], 'g^', label='test')
    plt.title('loss')
    plt.legend()

    plt.subplot(132)
    plt.plot(idx, phase1_acc[0], 'r--', idx, phase1_acc[1], 'bs', idx, phase1_acc[2], 'g^')
    plt.title('phase1_acc')

    plt.subplot(133)
    plt.plot(idx, phase2_acc[0], 'r--', idx, phase2_acc[1], 'bs', idx, phase2_acc[2], 'g^')
    plt.title('phase2_acc')

    plt.savefig('{}.png'.format(filename))

#plt.show()

parser = argparse.ArgumentParser(description='PyTorch implementation of pre-training of graph neural networks')
parser.add_argument('--root_dir', type=str, default='', help='root dir')
parser.add_argument('--filename', type=str, default='debug', help='output filename')
parser.add_argument('--multiprocess', action='store_true', default=False, help='train a model with multi process')
parser.add_argument('--num_processes', type=int, default=4, help='number of processes for multi-process training')
parser.add_argument('--typed', action='store_true', default=False, help='if given reaction types')
args = parser.parse_args()

if args.typed:
    args.filename = os.path.join('typed', args.filename)

if argparse.multiprocess:
    for rank in args.num_processes:
        logfile_dir = os.path.join(args.filename, 'log_rank{}.csv'.format(rank))
        logfile = open(logfile_dir, 'r')
        draw_log_pic(logfile, True, args.filename, rank)
else:
    logfile_dir = os.path.join(args.filename, 'log.csv')
    logfile = open(logfile_dir, 'r')
    draw_log_pic(logfile, False, args.filename)