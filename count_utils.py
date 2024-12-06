import numpy
import os
import pickle
import random

from test_utils import social_test, test_model

def test_count_model(args, key, datasets, present_words, trans_from_en, coocs, vocab, row_words, ctx_words):
    trans_pmi_vecs = build_ppmi_vecs(coocs, vocab, row_words, ctx_words, smoothing=False)
    model = {k : v for k, v in trans_pmi_vecs.items()}
    curr_vocab = [w for w in trans_pmi_vecs.keys()]
    test_model(args, key, model, curr_vocab, datasets, present_words, trans_from_en)

def test_frequency_model(args, key, datasets, present_words, trans_from_en, freqs, vocab, row_words):
    trans_pmi_vecs = build_frequency_vecs(args, freqs, row_words, )
    model = {k : v for k, v in trans_pmi_vecs.items()}
    curr_vocab = [w for w in trans_pmi_vecs.keys()]
    test_model(args, key, model, curr_vocab, datasets, present_words, trans_from_en)

def test_coocs_model(args, key, datasets, present_words, trans_from_en, coocs, vocab, row_words):
    trans_pmi_vecs = build_coocs_vecs(args, coocs, row_words, vocab)
    model = {k : v for k, v in trans_pmi_vecs.items()}
    #curr_vocab = [w for w in trans_pmi_vecs.keys()]
    test_model(args, key, model, row_words, datasets, present_words, trans_from_en)

def read_mitchell_25dims(lang):
    dimensions = list()
    with open(os.path.join('data', 'fmri', 'mitchell', 'mitchell_dimensions_{}.tsv'.format(lang))) as i:
        for l in i:
            line = l.strip().split()
            assert len(line) >= 2
            dimensions.append(line)
    assert len(dimensions) == 25

    return dimensions

def build_coocs_vecs(args, coocs, row_words, vocab):
    pmi_mtrx = numpy.array(
                             [
                              [coocs[vocab[w]][vocab[w_two]] if vocab[w_two] in coocs[vocab[w]].keys() else 0 for w_two in row_words]
                              for w in row_words])
    assert pmi_mtrx.shape[0] == len(row_words)
    assert pmi_mtrx.shape[1] == len(row_words)
    trans_pmi_vecs = {w : {w_two : pmi_mtrx[w_i][w_two_i] for w_two_i, w_two in enumerate(row_words)} for w_i, w in enumerate(row_words)}
    for v in trans_pmi_vecs.values():
        for v_two in v.values():
            assert not numpy.isnan(v_two)

    return trans_pmi_vecs

def build_frequency_vecs(args, freqs, row_words, ):
    pmi_mtrx = numpy.array(
                             [
                              freqs[w]
                              for w in row_words]
                              )
    assert pmi_mtrx.shape[0] == len(row_words)
    trans_pmi_vecs = {w : pmi_mtrx[w_i] for w_i, w in enumerate(row_words)}
    for v in trans_pmi_vecs.values():
        assert not numpy.isnan(v)

    return trans_pmi_vecs

def build_ppmi_vecs(coocs, vocab, row_words, col_words, smoothing=False, power=1.):
    pmi_mtrx = numpy.array(
                             [
                              [coocs[vocab[w]][vocab[w_two]] if vocab[w_two] in coocs[vocab[w]].keys() else 0 for w_two in col_words]
                              for w in row_words])
    assert pmi_mtrx.shape[0] == len(row_words)
    assert pmi_mtrx.shape[1] == len(col_words)
    if power != 1.:
        pmi_mtrx = numpy.power(pmi_mtrx, power)
    #matrix_[matrix_ != 0] = np.array(1.0/matrix_[matrix_ != 0])
    axis_one_sum = pmi_mtrx.sum(axis=1)
    #axis_one_mtrx = numpy.divide(1, axis_one_sum, where=axis_one_sum!=0).reshape(-1, 1)
    axis_one_mtrx = numpy.array([1/val if val!=0 else val for val in axis_one_sum]).reshape(-1, 1)
    assert True not in numpy.isnan(axis_one_mtrx)
    axis_zero_sum = pmi_mtrx.sum(axis=0)
    #axis_zero_mtrx = numpy.divide(1, axis_zero_sum, where=axis_zero_sum!=0).reshape(1, -1)
    axis_zero_mtrx = numpy.array([1/val if val!=0 else val for val in axis_zero_sum]).reshape(1, -1)
    assert True not in numpy.isnan(axis_one_mtrx)
    ### raising to 0.75 as suggested in Levy & Goldberg 2015
    if smoothing:
        total_sum = numpy.power(pmi_mtrx, 0.75).sum()
    else:
        total_sum = pmi_mtrx.sum()
    #trans_pmi_mtrx = numpy.multiply(numpy.multiply(numpy.multiply(pmi_mtrx,1/pmi_mtrx.sum(axis=1).reshape(-1, 1)), 1/pmi_mtrx.sum(axis=0).reshape(1, -1)), pmi_mtrx.sum())
    trans_pmi_mtrx = numpy.multiply(
                                    numpy.multiply(
                                                   numpy.multiply(
                                                                  pmi_mtrx,axis_one_mtrx), 
                                                   axis_zero_mtrx), 
                                    total_sum)
    trans_pmi_mtrx[trans_pmi_mtrx<1.] = 1
    assert True not in numpy.isnan(trans_pmi_mtrx.flatten())
    ### checking for nans
    trans_pmi_vecs = {w : numpy.log2(trans_pmi_mtrx[w_i]) for w_i, w in enumerate(row_words)}
    for v in trans_pmi_vecs.values():
        assert True not in numpy.isnan(v)

    return trans_pmi_vecs

def load_count_coocs(args):
    print(args.model)
    if args.lang == 'en':
        if 'bnc' in args.model:
            min_count = 10
        elif 'cc100' in args.model:
            min_count = 500
        else:
            min_count = 10
    else:
        if 'cc100' in args.model:
            if args.lang == 'it':
                min_count = 10
            else:
                min_count = 100
        else:
            min_count = 10
    #print(min_count)
    f = args.model.split('-')[0]
    base_folder = os.path.join(
                            '/',
                            'data',
                            'tu_bruera',
                            'counts',
                           args.lang, 
                           f,
                           )
    with open(os.path.join(
                            base_folder,
                           '{}_{}_uncased_word_freqs.pkl'.format(
                                                                 args.lang, 
                                                                 f
                                                                 ),
                           ), 'rb') as i:
        freqs = pickle.load(i)
    vocab_file = os.path.join(
                            base_folder,
                           '{}_{}_uncased_vocab_min_{}.pkl'.format(
                                                                   args.lang, 
                                                                   #args.model, 
                                                                   f,
                                                                   min_count
                                                                   ),
                           )
    if 'tagged_' in args.model:
        vocab_file = vocab_file.replace('.pkl', '_no-entities.pkl')
    with open(vocab_file, 'rb') as i:
        vocab = pickle.load(i)
    print('total size of the corpus: {:,} tokens'.format(sum(freqs.values())))
    print('total size of the vocabulary: {:,} words'.format(max(vocab.values())))
    if 'fwd' not in args.model and 'surprisal' not in args.model:
        coocs_file = os.path.join(base_folder,
                      '{}_{}_coocs_uncased_min_{}_win_20.pkl'.format(
                                                                         args.lang,
                                                                         #args.model, 
                                                                         f,
                                                                         min_count
                                                                         ),
                           )
    else:
        coocs_file = os.path.join(base_folder,
                      '{}_{}_forward-coocs_uncased_min_{}_win_20.pkl'.format(
                                                                         args.lang,
                                                                         #args.model, 
                                                                         f,
                                                                         min_count
                                                                         )
                      )
    if 'tagged_' in args.model:
        coocs_file = coocs_file.replace('.pkl', '_no-entities.pkl')
    with open(coocs_file, 'rb') as i:
        coocs = pickle.load(i)
    return vocab, coocs, freqs
