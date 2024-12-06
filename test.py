from tqdm import tqdm

from psycholing_norms_loaders import load_lancaster_en_de_it
from count_utils import build_ppmi_vecs, read_mitchell_25dims, load_count_coocs, test_count_model, test_coocs_model, test_frequency_model
from test_utils import args, check_present_words, load_dataset, load_static_model, load_context_model, load_context_surpr, rt, test_model

args = args()
lancaster_ratings, trans_from_en = load_lancaster_en_de_it(args)

rows, datasets = load_dataset(args, trans_from_en)

### for static models, we only test once
static_models = [
                 'fasttext',
                 'fasttext_aligned',
                 'conceptnet',
                 ]
top_freqs = [
                          100, 
                          200, 
                          500, 
                          750,
                          1000, 
                          2500, 
                          5000, 
                          7500,
                          10000, 
                          12500, 
                          15000, 
                          17500,
                          20000, 
                          25000,
                          30000,
                          35000,
                          40000,
                          45000,
                          50000,
                          60000,
                          70000,
                          80000,
                          90000,
                          100000,
                          150000,
                          200000,
                          250000,
                          300000,
                          350000,
                          400000,
                          450000,
                          500000,
                          #550000,
                          #600000,
                          #650000,
                          #700000,
                          #750000,
                          #800000,
                          #850000,
                          #900000,
                          #950000,
                          #1000000,
                          ]
#top_freqs = [200000]
if args.model == 'response_times':
    model = dict()
    vocab = [w for w in rows]
    present_words = [w for w in rows]
    rt(
       args, 
       args.model,
       model, 
       vocab, 
       datasets, 
       present_words,
       trans_from_en,
       )
elif args.model == 'word_length':
    model = {k : [len(k)] for k in rows}
    vocab = [w for w in rows]
    present_words = [w for w in rows]
    test_model(
               args, 
               args.model,
               model, 
               vocab, 
               datasets, 
               present_words,
               trans_from_en,
               )
elif args.model in static_models:
    model, vocab = load_static_model(args)
    present_words = check_present_words(args, rows, vocab)
    test_model(
               args, 
               args.model,
               model, 
               vocab, 
               datasets, 
               present_words,
               trans_from_en,
               )
elif 'xlm' in args.model or 'xglm' in args.model or 'llama' in args.model or 'pt' in args.model:
    if 'surpr' not in args.model:
        model, vocab = load_context_model(args)
    else:
        model, vocab = load_context_surpr(args)
    present_words = check_present_words(args, rows, vocab)
    test_model(
               args, 
               args.model,
               model, 
               vocab, 
               datasets, 
               present_words,
               trans_from_en,
               )
### for count models, we test with a lot of different possibilities
else:
    vocab, coocs, freqs = load_count_coocs(args)
    ### keeping row words that are actually available
    row_words = [w for w in rows if w in vocab.keys() and vocab[w] in coocs.keys() and vocab[w]!=0]
    present_words = check_present_words(args, row_words, list(vocab.keys()))
    if 'abs-prob' in args.model:
        test_frequency_model(args, args.model, datasets, present_words, trans_from_en, freqs, vocab, row_words,)
        pass
    elif 'surprisal' in args.model or 'cond-prob' in args.model:
        test_coocs_model(args, args.model, datasets, present_words, trans_from_en, coocs, vocab, row_words,)
    else:
        '''
        #
        ### mitchell hand-picked dimensions
        #
        for row_mode in [
                         '', 
                         #'_rowincol',
                         ]:
            key = 'ppmi_{}_mitchell{}_words'.format(args.model, row_mode)
            if row_mode == 'rowincol':
                ctx_words = set([w for ws in read_mitchell_25dims(args.lang) for w in ws] + row_words)
            else:
                ctx_words = [w for ws in read_mitchell(args.lang) for w in ws]
            test_count_model(args, key, datasets, trans_from_en, coocs, vocab, row_words, ctx_words)
        #
        ### lancaster
        #
        filt_ratings = {w : freqs[w] for w in lancaster_ratings[args.lang].keys() if w in vocab.keys() and vocab[w] in coocs.keys() and vocab[w]!=0}
        sorted_ratings = [w[0] for w in sorted(filt_ratings.items(), key=lambda item: item[1], reverse=True)]
        filt_perc = {w : v['minkowski3'] for w, v in lancaster_ratings[args.lang].items() if w in vocab.keys() and w in freqs.keys() and vocab[w]!=0}
        sorted_perc = [w[0] for w in sorted(filt_perc.items(), key=lambda item: item[1], reverse=True)]
        for freq in tqdm([
                          100, 
                          200, 
                          500, 
                          750,
                          1000, 
                          2500, 
                          5000, 
                          7500,
                          10000, 
                          12500, 
                          15000, 
                          17500,
                          20000, 
                          25000,
                          ]):
            for row_mode in [
                             '', 
                             #'_rowincol',
                             ]:
                for selection_mode in [
                                       'top', 
                                       #'random', 
                                       #'hi-perceptual', 
                                       #'lo-perceptual',
                                       ]: 
                    key = 'ppmi_{}_lancaster_freq_{}{}_{}_words'.format(args.model, selection_mode, row_mode, freq)
                    if selection_mode == 'top':
                        if row_mode == 'rowincol':
                            ctx_words = set([w for w in sorted_ratings[:freq]]+row_words)
                        else:
                            ctx_words = [w for w in sorted_ratings[:freq]]
                    elif selection_mode == 'hi-perceptual':
                        if row_mode == 'rowincol':
                            ctx_words = set([w for w in sorted_perc[:freq]]+row_words)
                        else:
                            ctx_words = [w for w in sorted_perc[:freq]]
                    elif selection_mode == 'lo-perceptual':
                        if row_mode == 'rowincol':
                            ctx_words = set([w for w in sorted_perc[-freq:]]+row_words)
                        else:
                            ctx_words = [w for w in sorted_perc[-freq:]]
                    else:
                        random.seed(12)
                        idxs = random.sample(range(len(sorted_ratings)), k=min(freq, len(sorted_ratings)))
                        if row_mode == 'rowincol':
                            ctx_words = set([sorted_ratings[i] for i in idxs]+row_words)
                        else:
                            ctx_words = [sorted_ratings[i] for i in idxs]
                    test_count_model(args, key, datasets, present_words, trans_from_en, coocs, vocab, row_words, ctx_words)
        '''
        #
        ### top-n frequencies
        #
        filt_freqs = {w : f for w, f in freqs.items() if w in vocab.keys() and vocab[w] in coocs.keys() and vocab[w]!=0}
        sorted_freqs = [w[0] for w in sorted(filt_freqs.items(), key=lambda item: item[1], reverse=True)]
        for freq in tqdm(
                         top_freqs
                          ):
            if freq > max(vocab.values()):
                print('too many words requested, skipping!')
                continue
            for row_mode in [
                             '_', 
                             #'_rowincol',
                             ]:
                for selection_mode in [
                                       'top', 
                                       #'random',
                                       ]: 
                    key = 'ppmi_{}_abs_freq_{}{}_{}_words'.format(args.model, selection_mode, row_mode, freq)
                    if selection_mode == 'top':
                        if row_mode == 'rowincol':
                            ctx_words = set([w for w in sorted_freqs[:freq]]+row_words)
                        else:
                            ctx_words = [w for w in sorted_freqs[:freq]]
                    else:
                        random.seed(12)
                        idxs = random.sample(range(len(sorted_freqs)), k=min(freq, len(sorted_freqs)))
                        if row_mode == 'rowincol':
                            ctx_words = set([sorted_freqs[i] for i in idxs]+row_words)
                        else:
                            ctx_words = [sorted_freqs[i] for i in idxs]
                    ### using the basic required vocab for all tests as a basis set of words
                    test_count_model(args, key, datasets, present_words, trans_from_en, coocs, vocab, row_words, ctx_words)
