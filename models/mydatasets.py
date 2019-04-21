import re
import os
import random
import tarfile
import urllib
from torchtext import data
from torchtext.data import Field


def ir_dataset(TEXT, args):
    tv_datafields = [("query", TEXT), ("doc", TEXT)]
    trn, vld = data.TabularDataset.splits(
                   path=args.data_path, # the root directory where the data lies
                   train='train.tsv', validation="dev.tsv",
                   format='tsv',
                   skip_header=True, 
                   fields=tv_datafields)
     
    query_datafield = [("query", TEXT)]
    doc_datafield = [("doc", TEXT)]


    query = data.TabularDataset(
               path=os.path.join(args.data_path, 'query.tsv'),
               format='tsv',
               skip_header=True,
               fields=query_datafield)

    doc = data.TabularDataset(
               path=os.path.join(args.data_path, 'test.tsv'),
               format='tsv',
               skip_header=True,
               fields=doc_datafield)

    TEXT.build_vocab(trn, vld)

    # iterator
    train_iter, val_iter = data.BucketIterator.splits(
         (trn, vld), 
         batch_sizes= (args.batch_size, args.batch_size),
         sort_key=lambda x: len(x.doc), 
         sort_within_batch=False,
         sort=False, 
         shuffle = True,
         repeat=False
    )

    query_iter = data.Iterator(query, batch_size=1, shuffle = False, sort_within_batch=False, repeat=False)
    doc_iter = data.Iterator(doc, batch_size=args.batch_size * 4, shuffle = False, sort_within_batch=False, repeat=False)

    return train_iter, val_iter, query_iter, doc_iter