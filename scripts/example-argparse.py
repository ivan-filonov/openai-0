# -*- coding: utf-8 -*-
import argparse
#%%
def parse_args():
    parser = argparse.ArgumentParser(description='kaggle avito model')
    parser.add_argument('--fold-from', '-f', default=1, type=int, help='fold to start processing from')
    parser.add_argument('--fold-to', '-t', default=16, type=int, help='last fold to process')

    parser.add_argument('--reuse', '-r', dest='reuse', default=True, action='store_true', help='allow reusing of fold predictions')
    parser.add_argument('--no-reuse', '-n', dest='reuse', action='store_false', help='do not allow reusing of fold predictions')

    parser.add_argument('--reuse-state', '-rs', dest='reuse_state', default=True, action='store_true', help='allow reusing of prepared data')
    parser.add_argument('--no-reuse-state', '-ns', dest='reuse_state', action='store_false', help='do not allow reusing of prepared data')

    parser.add_argument('--gpu', dest='gpu', nargs='+', default=[0], help='GPU to use')
    parser.add_argument('--cpu', dest='gpu', action='store_const', const=[], help='use CPU only')

    return parser.parse_args()

#%% MAIN
if '__main__' == __name__:
    print('DEBUG =', DEBUG)

    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(x) for x in args.gpu)
    assert 1 <= args.fold_from  <= 16
    assert args.fold_from  <= args.fold_to <= 16

