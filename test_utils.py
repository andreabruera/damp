import argparse
import fasttext
#import gensim
import numpy
import os
import pickle
import random
import scipy
import sklearn

from scipy import spatial
from sklearn import linear_model, metrics
from tqdm import tqdm

from fmri_loaders import read_abstract_ipc, read_fern, read_fern_areas, read_fern_categories, read_mitchell2008
from meeg_loaders import read_dirani_n400, read_kaneshiro_n400
from behav_loaders import read_italian_behav, read_italian_anew, read_italian_mouse, read_italian_deafhearing, read_italian_blindsighted, read_german_behav, read_picnaming_seven
from tms_loaders import read_it_social_quantity_tms, read_it_distr_learn_tms, read_de_pmtg_production_tms, read_de_sound_act_tms, read_de_sem_phon_tms
from simrel_norms_loaders import read_men, read_simlex, read_ws353
from utf_utils import transform_german_word, transform_italian_word

def social_test(args, model, present_words):
    trust = True
    if args.dataset == 'it_social-quantity' and 'abs-prob' not in args.model and 'surp' not in args.model:
        ws = {'sociale' : list(), 'quantità' : list()}
        with open('data/tms/it_tms_social-quant.tsv') as i:
            for l in i:
                line = l.strip().split('\t')
                if line[4][:4] == line[6][:4]:
                    if line[5] in present_words:
                        ws[line[4]].append(line[5])
        sims = {k : numpy.average([1-scipy.spatial.distance.cosine(model[k], model[v_two]) for v_two in v]) for k, v in ws.items()}
        for k, v in ws.items():
            for k_two, v_two in ws.items():
                if k == k_two:
                    continue
                sims[(k, k_two)] = numpy.average([1-scipy.spatial.distance.cosine(model[k], model[v]) for v in v_two])
        if sims['quantità']<sims[('quantità', 'sociale')] or sims['quantità']<sims[('sociale', 'quantità')]:
            #raise RuntimeError('untrustworthy model')
            trust = False
        if sims['sociale']<sims[('sociale', 'quantità')] or sims['sociale']<sims[('quantità', 'sociale')]:
            #raise RuntimeError('untrustworthy model')
            trust = False
    return trust

def check_dataset_words(args, dataset_name, dataset, present_words, trans_from_en, trans_words):
    #print('checking if words appear in the dictionary...')
    #import pdb; pdb.set_trace()
    missing_words = set()
    test_sims = list()
    if type(dataset) != list:
        dataset = [(k, v) for k, v in dataset.items()]
    for ws, val in dataset:
        ### prototypes
        if type(ws[0]) != tuple:
            first_word = set([ws[0]])
        else:
            for w in ws[0]:
                assert type(w) == str
            first_word = ws[0]
        marker = True
        w_ones = list()
        w_twos = list()
        ### hard-coded translations
        if 'fern' in dataset_name and args.lang in ['de', 'it']:
            ### word one
            for word in first_word:
                try:
                    candidates = trans_from_en[args.lang][word]
                    for c in candidates:
                        try:
                            present_words.index(c)
                            w_ones.append(c)
                        except ValueError:
                            #print(c)
                            pass
                except KeyError:
                    #print(ws[0])
                    pass
            ### word two 
            try:
                candidates = trans_from_en[args.lang][ws[1]]
                for c in candidates:
                    try:
                        present_words.index(c)
                        w_twos.append(c)
                    except ValueError:
                        #print(c)
                        pass
            except KeyError:
                #print(ws[1])
                pass
        else:
            #if args.lang in ['de', 'it']:
            if args.lang in ['de']:
                ### first words
                for word in first_word:
                    if '_' in word:
                        word = word.split('_')[1]
                    try:
                        w_ones.extend(trans_words[word])
                    except KeyError:
                        if args.lang == 'de':
                            curr_words = transform_german_word(word)
                        if args.lang == 'it':
                            curr_words = transform_italian_word(word)
                        #print(curr_words)
                        trans_words[word] = list()
                        for w in curr_words:
                            if '_' in w:
                                w = w.split('_')[1]
                            try:
                                present_words.index(w)
                                trans_words[word].append(w)
                            except ValueError:
                                #print(w)
                                continue
                        w_ones.extend(trans_words[word])
                ### second words
                #print(ws[1])
                try:
                    w_twos.extend(trans_words[ws[1]])
                except KeyError:
                    if args.lang == 'de':
                        curr_words = transform_german_word(ws[1])
                    if args.lang == 'it':
                        curr_words = transform_italian_word(ws[1])
                    trans_words[ws[1]] = list()
                    for w in curr_words:
                        if '_' in w:
                            w = w.split('_')[1]
                        try:
                            present_words.index(w)
                            trans_words[ws[1]].append(w)
                        except ValueError:
                            continue
                    w_twos.extend(trans_words[ws[1]])
            else:
                for word in first_word:
                    if '_' in word:
                        word = word.split('_')[1]
                    for w in [word.lower(), word.capitalize()]:
                        try:
                            present_words.index(w)
                        except ValueError:
                            continue
                        w_ones.append(w)
                for w in [ws[1].lower(), ws[1].capitalize()]:
                    if '_' in w:
                        w = w.split('_')[1]
                    try:
                        present_words.index(w)
                    except ValueError:
                        continue
                    w_twos.append(w)
        if len(w_ones)<1:
            missing_words = missing_words.union(first_word)
        if len(w_twos)<1:
            missing_words.add(ws[1])
        if len(w_ones)<1 or len(w_twos)<1:
            marker = False
        if marker:
            test_sims.append((
                              list(set(w_ones)), list(set(w_twos)), val))
            #print(set(w_ones))
            #print(set(w_twos))
    return test_sims, missing_words, trans_words

def compute_corr(args, model_sims, test_sims, to_be_removed):
    #test_sims, missing_words = check_dataset_words(args, dataset_name, dataset, present_words, trans_from_en,)
    assert len(test_sims) > 0
    real = list()
    pred = list()
    for w_ones, w_twos, v in test_sims:
        joint_key = ('_'.join(w_ones), '_'.join(w_twos))
        marker = False
        for k in joint_key:
            if k in to_be_removed:
                marker = True
        if marker:
            continue
        real.append(v)
        pred.append(model_sims[joint_key])
    if args.approach == 'correlation' or ('social' not in args.dataset and 'distr' not in args.dataset and 'sem-phon' not in args.dataset and 'pmtg' not in args.dataset and 'sound' not in args.dataset):
        corr = scipy.stats.spearmanr(real, pred).statistic
        #corr = scipy.stats.pearsonr(real, pred).statistic
        return corr
    elif args.approach == 'rsa':
        ### RSA
        ### squaring the matrix
        sq_real = list()
        sq_pred = list()
        for i in range(len(real)):
            for i_two in range(len(real)):
                if i_two <= i:
                    continue
                sq_real.append(abs(real[i]-real[i_two]))
                sq_pred.append(abs(pred[i]-pred[i_two]))
        corr = scipy.stats.spearmanr(sq_real, sq_pred).statistic
        return corr
    ### different out...
    elif args.approach == 'rsa_encoding':
        #rescale_real = [(p-min(real))/(max(real)-min(real)) for p in real]
        #rescale_pred = [(p-min(pred))/(max(pred)-min(pred)) for p in pred]
        #assert max(rescale_pred) == 1.
        #assert min(rescale_pred) == 0.
        r_avg = numpy.average(real)
        r_std = numpy.std(real)
        rescale_real = [float((p-r_avg)/r_std) for p in real]
        p_avg = numpy.average(pred)
        p_std = numpy.std(pred)
        rescale_pred = [float((p-p_avg)/p_std) for p in pred]
        rng = numpy.random.default_rng()
        splits = rng.choice(
                             range(len(real)),
                             replace=True,
                             size=(
                                   ### 20% test
                                   4,
                                   ### 10 repetitions
                                   10
                                   ),
                             )
        ins = list()
        outs = list()
        for split in splits:
            train_real = [rescale_real[_] for _ in range(len(real)) if _ not in split]
            train_pred = [rescale_pred[_] for _ in range(len(rescale_pred)) if _ not in split]
            #train_pred = random.sample(train_pred, k=len(train_pred))
            spl_i = list()
            spl_o = list()
            for test_idx in split:
                test_item_real = rescale_real[test_idx]
                test_item_pred = rescale_pred[test_idx]
                #denom = [float(1-abs(test_item_pred-pr)) for pr in train_pred]
                denom = [-abs(test_item_pred-pr) for pr in train_pred]
                assert len(denom) == len(train_pred)
                #assert max(denom) <= 1.
                #assert min(denom) >= -1.
                o = sum([tr*d for d, tr in zip(denom, train_real)])/sum(denom)
                if str(spl_i) == 'nan':
                    import pdb; pdb.set_trace()
                spl_i.append(test_item_real)
                spl_o.append(o)
                if str(spl_o) == 'nan':
                    import pdb; pdb.set_trace()
            ins.append(spl_i)
            outs.append(spl_o)
        return ins, outs

def write_res(args, case, dataset_name, corr, trust=True):
    corpus_fold = case.split('_')[1] if 'ppmi' in case else case
    details = '_'.join(case.split('_')[2:]) if 'ppmi' in case else case
    out_folder = os.path.join(
                              'test_results',
                              args.approach,
                              args.stat_approach,
                              args.evaluation,
                              args.lang, 
                              corpus_fold, 
                              details,
                              )
    os.makedirs(out_folder, exist_ok=True)
    out_f = os.path.join(out_folder, '{}.tsv'.format(dataset_name))
    if trust == False:
        out_f = out_f.replace('.tsv', '_DONTRUST.tsv')
    with open(out_f, 'w') as o:
        o.write('{}\t{}\t{}\t'.format(args.lang, case, dataset_name))
        for c in corr:
            o.write('{}\t'.format(c))
    print(out_f)

def check_present_words(args, rows, vocab):
    present_words = list()
    for w in rows:
        ### for fasttext in german we only use uppercase!
        if w[0].isupper() == False and args.lang=='de':
            if args.model=='fasttext':
                #or 'lm' in args.model or 'llama' in args.model:
                continue
        try:
            vocab.index(w)
        except ValueError:
            continue
        present_words.append(w)
    return present_words

def rt(args, case, model, vocab, datasets, present_words, trans_from_en):
    if args.stat_approach != 'simple':
        datasets = bootstrapper(args, datasets, )
    else:
        datasets = {k : [v] for k, v in datasets.items()}
    for dataset_name, dataset in datasets.items():
        corr = list()
        ### bootstrapping/iterations should be hard-coded now...
        for iter_dataset in tqdm(dataset):
            iter_corrs = list()
            for s, s_data in iter_dataset.items():
                curr_corr = numpy.average([v[1] for v in s_data])
                if curr_corr == None:
                    print('error with {}'.format([args.lang, case, dataset_name]))
                    continue
                iter_corrs.append(curr_corr)
            if args.stat_approach == 'simple':
                corr.extend(iter_corrs)
            else:
                iter_corr = numpy.average(iter_corrs)
                corr.append(iter_corr)

        write_res(args, case, dataset_name, corr, )

def test_model(args, case, model, vocab, datasets, present_words, trans_from_en):
    trust = social_test(args, model, present_words)
    if args.stat_approach != 'simple':
        datasets = bootstrapper(args, datasets, )
    else:
        datasets = {k : [v] for k, v in datasets.items()}
    ### time shortcut 1:
    ### we pre-load all sim tests
    print('now pre-checking words...')
    all_sims_data = dict()
    to_be_computed = set()
    with tqdm() as counter:
        trans_words = dict()
        for dataset_name, dataset in datasets.items():
            assert type(dataset) == list
            assert len(dataset) in [1, 10, 1000]
            all_sims_data[dataset_name] = list()
            missing_words = set()
            for iter_dataset in dataset:
                iter_dict = dict()
                for s, s_data in iter_dataset.items():
                    test_sims, new_miss, trans_words = check_dataset_words(
                                                              args, 
                                                              dataset_name, 
                                                              s_data, 
                                                              present_words, 
                                                              trans_from_en,
                                                              trans_words
                                                              )
                    #print(trans_words)
                    #all_sims_data[dataset_name][iter_dataset].append(test_sims)
                    iter_dict[s] = test_sims
                    missing_words = missing_words.union(new_miss)
                    for w_ones, w_twos, v in test_sims:
                        key = ('_'.join(w_ones), '_'.join(w_twos))
                        to_be_computed.add(key)
                all_sims_data[dataset_name].append(iter_dict)
                counter.update(1)
    ### time-shortcut 2:
    ### we pre-compute all pairwise similarities
    ws_vecs = dict()
    ws_sims = dict()
    to_be_removed = list()
    print('now pre-computing model dissimilarities...')
    #print(to_be_computed)
    with tqdm() as counter:
        ### probabilities / frequencies
        if 'cond-prob' in args.model or 'surprisal' in args.model:
            for joint_ones, joint_twos in to_be_computed:
                ### first words
                w_ones = tuple(joint_ones.split('_'))
                ### second words
                w_twos = tuple(joint_twos.split('_'))
                cooc = list()
                for one in w_ones:
                    for two in w_twos:
                        try:
                            cooc.append(model[one][two])
                        except KeyError:
                            continue
                        ### for most tms cases, surprisal is directional,
                        ### so one -> two
                        if 'distr' in args.dataset:
                            pass
                        elif 'social' in args.dataset:
                            pass
                        elif 'sem' in args.dataset:
                            pass
                        #elif 'act' in args.dataset:
                        #    pass
                        elif 'prod' in args.dataset:
                            pass
                        else:
                            ### if not directional, one->two and two->one
                            try:
                                cooc.append(model[two][one])
                            except KeyError:
                                continue


                #print(cooc)
                ### negative!
                if 'pt' not in args.model and 'llam' not in args.model and 'lm' not in args.model:
                    if 'log10' in args.model or 'surprisal' in args.model:
                        ### we smooth by adding 1 to all counts
                        sim = -numpy.log2(sum(cooc)+1)
                        if str(sim) == 'nan':
                            #sim = 0
                            raise RuntimeError()
                    else:
                        sim = -sum(cooc)
                else:
                    sim = numpy.nanmean(cooc)
                    if str(sim) == 'nan':
                        sim = 0
                ws_sims[(joint_ones, joint_twos)] = sim
                counter.update(1)
        else:
            for joint_ones, joint_twos in to_be_computed:
                ### first words
                w_ones = tuple(joint_ones.split('_'))
                #print(w_ones)
                try:
                    vecs_ones = ws_vecs[w_ones]
                except KeyError:
                    if 'abs-prob' in args.model:
                        ws_vecs[w_ones] = numpy.sum([model[w] for w in w_ones], axis=0)
                    else:
                        ws_vecs[w_ones] = numpy.average([model[w] for w in w_ones], axis=0)
                    vecs_ones = ws_vecs[w_ones]
                ### second words
                w_twos = tuple(joint_twos.split('_'))
                #print(w_twos)
                try:
                    vecs_twos = ws_vecs[w_twos]
                except KeyError:
                    if 'abs-prob' in args.model:
                        ws_vecs[w_twos] = numpy.sum([model[w] for w in w_twos], axis=0)
                    else:
                        ws_vecs[w_twos] = numpy.average([model[w] for w in w_twos], axis=0)
                    vecs_twos = ws_vecs[w_twos]
                ### for length, we sum
                if  'length' in args.model:
                    sim = sum([vecs_ones, vecs_twos])
                elif 'abs-prob' in args.model:
                    ### negative!
                    if 'log' in args.model:
                        #print('using smoothed log')
                        if 'social' in args.dataset or \
                           'sound' in args.dataset or \
                           'pmtg' in args.dataset or \
                           'sem-phon' in args.dataset:
                            #print(w_twos)
                            sim = -numpy.log10(vecs_twos)
                        else:
                            sim = -numpy.log10(sum([vecs_ones, vecs_twos]))
                    else:
                        ### negative!
                        if 'social' in args.dataset:
                            #print(w_twos)
                            sim = -vecs_twos
                        else:
                            sim = -sum([vecs_ones, vecs_twos])
                else:
                    sim = scipy.spatial.distance.cosine(vecs_ones, vecs_twos)
                if 'social' in args.dataset:
                    if 'abs-prob' in args.model:
                        #print('checking')
                        if 'log' in args.model:
                            assert sim == -numpy.log10(ws_vecs[w_twos])
                        else:
                            assert sim == -ws_vecs[w_twos]
                if str(sim) == 'nan':
                    if sum(vecs_ones) == 0:
                        to_be_removed.append(joint_ones)
                    if sum(vecs_ones) == 1:
                        to_be_removed.append(joint_twos)
                    #to_be_removed.append((joint_ones, joint_twos))
                    continue
                ws_sims[(joint_ones, joint_twos)] = sim
                counter.update(1)
    if 'social' in args.dataset:
        if 'abs-prob' in args.model:
            #print('checking')
            ts = set([v[1] for v in ws_sims.keys()])
            for t in ts:
                poss = set([v for k, v in ws_sims.items() if t in k])
                assert len(list(poss)) == 1
    print('impossible to compute similarities in these cases:')
    print(set([w for w in to_be_removed]))
    ### now we can run correlations
    #for dataset_name, dataset in datasets.items(): for dataset_name, dataset
    for dataset_name, dataset in all_sims_data.items():
        corr = list()
        ### bootstrapping/iterations should be hard-coded now...
        for iter_dataset in tqdm(dataset):
            iter_corrs = list()
            for s, s_data in iter_dataset.items():
                curr_corr = compute_corr(args, ws_sims, s_data, to_be_removed)
                if curr_corr == None:
                    print('error with {}'.format([args.lang, case, dataset_name]))
                    continue
                iter_corrs.append(curr_corr)
            if args.approach in ['rsa', 'correlation']:
                if args.stat_approach == 'simple':
                    corr.extend(iter_corrs)
                else:
                    iter_corr = numpy.average(iter_corrs)
                    corr.append(iter_corr)
            elif args.approach == 'rsa_encoding':
                #print('encoding')
                iter_corrs = numpy.array(iter_corrs)
                if args.stat_approach == 'bootstrap':
                    assert iter_corrs.shape == (20, 2, 4, 10)
                ins = iter_corrs[:, 0, :, :]
                outs = iter_corrs[:, 1, :, :]
                enc_corrs = list()
                for _ in range(ins.shape[-1]):
                    for __ in range(outs.shape[-2]):
                        #c = scipy.stats.spearmanr(ins[:, __, _], outs[:, __, _]).statistic
                        if args.evaluation == 'r2':
                            if __ > 0:
                                continue
                            #import pdb; pdb.set_trace()
                            r2s = list()
                            for ___ in range(outs.shape[0]):
                                #c = sklearn.metrics.r2_score(ins[___, :, _], outs[___, :, _])
                                c = sklearn.metrics.mean_squared_error(ins[___, :, _], outs[___, :, _])
                                r2s.append(c)
                            c = numpy.average(r2s)
                        elif args.evaluation == 'squared_error':
                            c = sklearn.metrics.mean_squared_error(ins[:, __, _], outs[:, __, _])
                        elif args.evaluation == 'spearman':
                            c = scipy.stats.spearmanr(ins[:, __, _], outs[:, __, _]).statistic
                        else:
                            raise RuntimeError()
                        enc_corrs.append(c)
                corr.append(numpy.average(enc_corrs))

        print('\n')
        print('{} model'.format(case))
        print('correlation with {} dataset:'.format(dataset_name))
        print(numpy.nanmean(corr))
        if len(missing_words) > 0:
            print('missing words: {}'.format(missing_words))
        write_res(args, case, dataset_name, corr, trust=trust)
        #return results

def bootstrapper(args, full_data, ):
    ### bootstrapping with b=block_size of the original data
    ### Politis et al 1999, Bootstrapping page 198 "Indeed, for b too close to n 
    ### all subsample statistics (On,b,i or On,b,t) will be almost equal to On, 
    ### resulting in the subsampling distribution being too tight and in 
    ### undercoverage of subsampling confidence intervals [...]
    ###  On the other hand, if b is too small, the intervals can undercover or overcover depending on the state of nature. 
    ### This leaves a number of b-values in
    ### the "right range" where we would expect almost correct results, at least
    ### for large sample sizes. Hence, in this range, the confidence intervals should
    ### be "stable" when considered as a function of the block size. This idea is
    ### exploited by computing subsampling intervals for a large number of block
    ### sizes b and then looking for a region where the intervals do not change
    ### see figure in page 191
    ### we do not estimate it, but use values used in their simulation (page 208)
    if args.stat_approach == 'residualize':
        proportions = [
                       0.5
                       ]
    else:
        if 'behav' in args.dataset:
            proportions = [
                           4/256,
                           8/256,
                           16/256,
                           32/256,
                           64/256,
                           ]
        else:
            ### Riccardo De Bin, Silke Janitza, Willi Sauerbrei, Anne-Laure Boulesteix, 
            ### Subsampling Versus Bootstrapping in Resampling-Based Model Selection for 
            ### Multivariable Regression, Biometrics, Volume 72, Issue 1, 
            ### March 2016, Pages 272–280
            proportions = [
                    0.632
                    #0.5
                    ]
    ### labels
    labels = list(full_data.keys())
    all_subjects = {k : list(v.keys()) for k, v in full_data.items()}
    all_trials = {k : [len(vs) for vs in v.values()] for k, v in full_data.items()}
    #if 'all' in subjects and args.stat_approach ==  'residualize':
    #    raise RuntimeError('single subject not implemented')
    '''
    ### subjects
    all_subjects = [list(v.keys()) for v in full_data.values()]
    # checking
    n_subjects = set([len(v) for v in all_subjects])
    #assert len(n_subjects) == 1
    n_subjects = list(n_subjects)[0]
    subjects = list(set([val for v in all_subjects for val in v]))
    #subjects = [s if s!='all' else 0 for s in subjects]
    if 'social' not in args.dataset:
        assert len(subjects) == n_subjects
        n_subjects = min([len(v) for v in all_subjects])
    '''
    ### for tms we fix n=20
    tms_datasets = [
                   'it_distr-learn',
                   'it_social-quantity',
                   'de_sem-phon', 
                   'de_sound-act',
                   'de_pmtg-prod',
                   ]
    behav_datasets = [
                'de_behav',
                'it_behav',
                'it_mouse',
                'it_deafhearing',
                'it_blindsighted',
                'picture-naming-seven',
                'it_anew-lexical-decision',
                'it_anew-word-naming',
                'it_anew',
            ]
    #if args.dataset not in tms_datasets and args.dataset not in behav_datasets:
    #    n_iter_sub = max(1, int(min(list(all_subjects.values()))*random.choice(proportions)))
    #else:
    n_iter_sub = 20
    n_iter_trials = 20
    for k, v in all_subjects.items():
        if len(v) < n_iter_sub:
            print('max number of subjects for {}: {}'.format(k, len(v)))
    for k, vs in all_trials.items():
        smaller_ns = [_ for _ in vs if _ <n_iter_trials]
        if len(smaller_ns) > 0:
            print('insufficient number of trials for {} of subjects: {}'.format(len(smaller_ns)/len(vs), smaller_ns))
    ### here we create 1000
    boot_data = {l : list() for l in labels}
    if args.stat_approach == 'residualize':
        print('residualizing...')
    for _ in tqdm(range(1000)):
        iter_subs = {l : random.sample(subjects, k=min(n_iter_sub, len(subjects))) for l, subjects in all_subjects.items()}
        ### for tms we fix ns=15,20
        if args.dataset not in tms_datasets and args.dataset not in behav_datasets:
            iter_data_idxs = {l : 
                               {s : random.sample(
                                                 range(len(full_data[l][s])), 
                                                 k=int(len(full_data[l][s].keys())*random.choice(proportions))
                                                 ) for s in iter_subs[l]}
                                                 for l in labels}
        else:
            iter_data_idxs = {l : 
                               {s : random.sample(
                                                 range(len(full_data[l][s])), 
                                                 k=min(n_iter_trials, len(full_data[l][s])),
                                                 ) for s in iter_subs[l]}
                                                 for l in labels}
        iter_data = {l : {s : [(full_data[l][s][k][0], full_data[l][s][k][1]) for k in iter_data_idxs[l][s]] for s in iter_subs[l]} for l in labels}
        ### residualization
        if args.stat_approach == 'residualize':
            struct_train_data = {l : {s : [(full_data[l][s][k][0], full_data[l][s][k][1]) for k in range(len(full_data[l][s])) if k not in iter_data_idxs[l][s]] for s in iter_subs[l]} for l in labels}
            flat_train_data = [(l, s, k, rt) for l, l_res in struct_train_data.items() for s, s_res in l_res.items() for k, rt in s_res]
            flat_test_data = [(l, s, k, rt) for l, l_res in iter_data.items() for s, s_res in l_res.items() for k, rt in s_res]
            model = sklearn.linear_model.LinearRegression()
            ### we remove tms/sham -> t[0] per subject -> t[1]
            model.fit(
                      ### input
                      [
                       [
                        ### word(s) length
                        len(''.join(t[2])),
                        ### tms/sham-vertex
                        #labels.index(t[0]), 
                        ### subject
                        #t[1], 
                        ] for t in flat_train_data],
                      ###target
                      [
                       [
                        ### rt
                        t[3]
                        ] for t in flat_train_data],
                      )
            preds = model.predict(
                                  [[
                                    #labels.index(t[0]),t[1], i
                                    len(''.join(t[2]))] for t in flat_test_data]
                                  )
            #print(preds)
            #print([t[3] for t in flat_test_data])
            residuals = [(real[0], real[1], real[2], real[3]-pred[0]) for real, pred in zip(flat_test_data, preds)]
            for l, s, k, r in residuals:
                try:
                    iter_data[l][s].append((k, r))
                except KeyError:
                    iter_data[l][s] = [(k, r)]
        for l, l_data in iter_data.items():
            boot_data[l].append(l_data)
    return boot_data

def check_args(args):
    ### checking language if in first part of the name
    if '_' in args.dataset:
        assert args.dataset.split('_')[0] == args.lang
    if args.dataset == 'men' and args.lang != 'en':
        raise RuntimeError()
    if 'lexical' in args.dataset or 'naming' in args.dataset or 'behav' in args.dataset:
        if args.stat_approach == 'residualize':
            raise RuntimeError()
    if args.approach != 'rsa_encoding':
        if args.evaluation != 'spearman':
            raise RuntimeError()

def load_dataset(args, trans_from_en):
    if args.dataset == 'en_men':
        data, vocab = read_men(args)
    if '999' in args.dataset:
        data, vocab = read_simlex(args)
    if '353' in args.dataset:
        data, vocab = read_ws353(args)
    if 'fern' in args.dataset:
        if 'areas' in args.dataset:
            if 'all' in args.dataset:
                data, vocab = read_fern_areas(args, trans_from_en)
            elif 'categories' in args.dataset:
                raise RuntimeError('to be implemented')
        else:
            if 'all' in args.dataset:
                data, vocab = read_fern(args, trans_from_en)
            elif 'categories' in args.dataset:
                data, vocab = read_fern_categories(args, trans_from_en)
    if 'dirani' in args.dataset:
        data, vocab = read_dirani_n400(args)
    if 'kane' in args.dataset:
        data, vocab = read_kaneshiro_n400(args)
    if 'mitchell' in args.dataset:
        data, vocab = read_mitchell2008(args)
    if 'abstract' in args.dataset:
        data, vocab = read_abstract_ipc(args)
    if 'seven' in args.dataset:
        data, vocab = read_picnaming_seven(args)
    if 'de_behav' in args.dataset:
        data, vocab = read_german_behav(args)
    if 'it_behav' in args.dataset:
        data, vocab = read_italian_behav(args)
    if 'it_anew' in args.dataset:
        data, vocab = read_italian_anew(args)
    if 'it_deafhearing' in args.dataset:
        data, vocab = read_italian_deafhearing(args)
    if 'it_blindsighted' in args.dataset:
        data, vocab = read_italian_blindsighted(args)
    if 'mouse' in args.dataset:
        data, vocab = read_italian_mouse(args)
    if 'sem-phon' in args.dataset:
        data, vocab = read_de_sem_phon_tms(args)
    if 'sound-act' in args.dataset:
        data, vocab, prototypes = read_de_sound_act_tms(args)
    if 'pmtg-prod' in args.dataset:
        data, vocab = read_de_pmtg_production_tms(args)
    if 'distr-learn' in args.dataset:
        data, vocab = read_it_distr_learn_tms(args)
    if 'social-quantity' in args.dataset:
        data, vocab = read_it_social_quantity_tms(args)
    return vocab, data

def load_static_model(args):
    print('loading {}'.format(args.model))
    base_folder = os.path.join(
                                '/',
                                'data',
                                'u_bruera_software',
                                #'tu_bruera',
                                'word_vectors', 
                                args.lang, 
                                )
    if args.model == 'fasttext':
        model = fasttext.load_model(
                                    os.path.join(
                                        base_folder,
                                        'cc.{}.300.bin'.format(args.lang)
                                        )
                                    )
        vocab = model.words
    elif args.model == 'conceptnet':
        with open(
                os.path.join(
                    base_folder,
                   'conceptnet_{}.pkl'.format(args.lang)
                   ), 'rb') as i:
            model = pickle.load(i)
        vocab = model.keys()
    elif args.model == 'fasttext_aligned':
        with open(
                  os.path.join(
                            base_folder,
                           'ft_{}_aligned.pkl'.format(args.lang)
                           ), 'rb') as i:
            model = pickle.load(i)
        vocab = model.keys()
    model = {w : model[w] for w in vocab}
    vocab = [w for w in vocab]

    return model, vocab

def load_context_surpr(args):
    print('loading {}'.format(args.model))
    model = args.model.split('_')[0]
    base_folder = os.path.join(
                                'collect_word_sentences',
                                'llm_surprisals',
                                args.lang, 
                                model,
                                )
    assert os.path.exists(base_folder)
    vocab = set()
    model = dict()
    for f in os.listdir(base_folder):
        if 'tsv' not in f:
            continue
        if args.dataset not in f:
            continue
        with open(os.path.join(base_folder, f)) as i:
            for l_i, l in enumerate(i):
                if l_i == 0:
                    continue
                line = l.strip().split('\t')
                word_one = line[0].strip()
                word_two = line[1].strip()
                s = float(line[2])
                #s = float(line[3])
                try:
                    model[word_one][word_two] = s
                except KeyError:
                    model[word_one] = {word_two : s}
                vocab.add(word_one)
                vocab.add(word_two)
    vocab = list(vocab)

    return model, vocab

def load_context_model(args):
    print('loading {}'.format(args.model))
    model = args.model.split('_')[0]
    if '_best' not in args.model:
        layer = [int(args.model.split('-')[-1])]
    else:
        layer = [8, 9, 10, 11, 12]

    base_folder = os.path.join(
                                'collect_word_sentences',
                                'llm_vectors',
                                args.lang, 
                                'wac',
                                model,
                                )
    assert os.path.exists(base_folder)
    vocab = list()
    model = dict()
    for f in os.listdir(base_folder):
        if 'tsv' not in f:
            continue
        with open(os.path.join(base_folder, f)) as i:
            for l_i, l in enumerate(i):
                if l_i == 0:
                    continue
                line = l.strip().split('\t')
                word = line[0]
                #if args.lang == 'de':
                #    if word[0].isupper() == False:
                #        continue
                l = int(line[1])
                if l in layer:
                    if word in model.keys():
                        model[word] = numpy.average([model[word].copy(), numpy.array(line[2:], dtype=numpy.float64)], axis=0)
                    else:
                        model[word] = numpy.array(line[2:], dtype=numpy.float64)
                    vocab.append(word)

    return model, vocab

def args():
    parser = argparse.ArgumentParser()
    corpora_choices = ['word_length', 'levenshtein']
    llms = [
         'gpt2',
         'gpt2-small',
         'minervapt-350m',
         'minervapt-1b',
         'minervapt-3b',
         'llama-1b',
         'llama-3b',
         'xglm-7.5b',
         'xglm-1.7b',
         'xglm-564m',
         'xglm-2.9b',
         'xglm-4.5b',
         'xlm-roberta-large',
         'xlm-roberta-xl',
         'xlm-roberta-xxl',
         ]
    for llm in llms:
        if '1b' in llm:
            m = 16
        elif '3b' in llm:
            m = 28
        elif 'erta-xl' in llm:
            m = 36
        elif '2.9' in llm:
            m = 48
        elif '4.5' in llm:
            m = 48
        elif '7.5' in llm:
            m = 48
        elif 'erpt' in llm:
            m = 36
        elif 'iner' in llm:
            m = 16
        else:
            m = 24
        for l in range(m):
            corpora_choices.append('{}_layer-{}'.format(llm, l))
        corpora_choices.append('{}_surprisal'.format(llm))
        corpora_choices.append('{}_best'.format(llm))
    for corpus in [
                   'bnc',
                   'wac',
                   'tagged_wiki',
                   'opensubs',
                   'wac',
                   'cc100',
                   'tagged_leipzig',
                   'tagged_gutenberg',
                   ]:
        corpora_choices.append('{}-ppmi-vecs'.format(corpus))
        for mode in [
                     'neg-raw-abs-prob',
                     'neg-log10-abs-prob',
                     'neg-sym-raw-cond-prob',
                     'neg-fwd-raw-cond-prob',
                     'neg-sym-log10-cond-prob',
                     'surprisal',
                     ]:
            corpora_choices.append('{}-{}'.format(corpus, mode))
    parser.add_argument(
                        '--model',
                        choices=[
                                 'response_times',
                                 'fasttext',
                                 'fasttext_aligned',
                                 'conceptnet',
                                 ] + corpora_choices,
                        required=True,
                        )
    parser.add_argument(
                        '--lang',
                        choices=[
                                 'en',
                                 'de',
                                 'it',
                                 ],
                        required=True
                        )
    parser.add_argument(
                        '--dataset',
                        choices=[
                                ### sim-lex norms
                                'simlex999',
                                'ws353',
                                'en_men',
                                ### fmri
                                'fern2-all',
                                'fern2-areas-all',
                                'fern1-categories',
                                'fern1-all',
                                'fern1-areas-all',
                                'fern2-categories',
                                'de_abstract-fmri',
                                'mitchell2008',
                                ### meeg
                                'dirani-n400',
                                'kaneshiro-n400',
                                ### behav
                                'de_behav',
                                'it_behav',
                                'it_mouse',
                                'it_anew',
                                'it_deafhearing',
                                'it_blindsighted',
                                'picture-naming-seven',
                                ### tms
                                'de_sem-phon',
                                'de_pmtg-prod',
                                'de_sound-act',
                                'it_distr-learn',
                                'it_social-quantity',
                                ],
                        required=True,
                        )
    parser.add_argument(
                        '--stat_approach',
                        choices=['simple', 'bootstrap', 'residualize'],
                        required=True,
                        )
    parser.add_argument(
                        '--approach',
                        choices=['rsa', 'correlation', 'rsa_encoding'],
                        required=True,
                        )
    parser.add_argument(
                        '--evaluation',
                        choices=['spearman', 'r2', 'squared_error'],
                        required=True,
                        )
    #senses = ['auditory', 'gustatory', 'haptic', 'olfactory', 'visual', 'hand_arm']   
    args = parser.parse_args()
    check_args(args)

    return args
