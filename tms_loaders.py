import matplotlib
import numpy
import os

from matplotlib import pyplot

from utf_utils import transform_german_word, transform_italian_word

def plot_tms_raw_rts(sims):
    res = dict()
    for k, v in sims.items():
        general = k.split('#')[0]
        specific = k.split('#')[1].split('_')[0].split('-')[0]
        superspecific = k.split('#')[1].split('_')[0].split('-')[1]
        area = k.split('#')[1].split('_')[1]
        area = '{} {}'.format(area, superspecific)
        #print(specific)
        #print(area)
        try:
            res[specific][area] = [r[2] for r in v]
        except KeyError:
            res[specific] = {area : [r[2] for r in v]}
    for spec, ar_dict in res.items():
        fig, ax = pyplot.subplots(constrained_layout=True)
        std = 1.*max([numpy.std(v) for v in ar_dict.values()])
        ymin = min([numpy.average(v) for v in ar_dict.values()])-std
        ymax = max([numpy.average(v) for v in ar_dict.values()])+std
        count = 0
        #for area, ar_res in ar_dict.items():
        areas = sorted(list(ar_dict.keys()))
        for _ in range(int(len(areas)*2)):
            if count > len(areas)-1:
                continue
            if _ in range(3, int(len(areas)*2), 4):
                continue
            else:
                area = areas[count]
                ar_res = ar_dict[area]
                ax.bar(_, numpy.average(ar_res), label=area)
                ax.errorbar(_, numpy.average(ar_res), yerr=numpy.std(ar_res), color='black')
                count += 1
        ax.legend(
                  ncol=3
                  )
        ymin = 2.75
        ymax = 3.05
        ax.set_ylim(bottom = ymin, top=ymax)
        pyplot.title('{} - {}'.format(general, spec.replace('_', ' ')))
        path = os.path.join('raw_tms_rts', general)
        os.makedirs(path, exist_ok=True)
        gen_path = os.path.join(path, '{}.jpg'.format(spec))
        pyplot.savefig(gen_path)
        print(gen_path)

def read_de_pmtg_production_tms(args):
    lines = list()
    missing = 0
    with open(os.path.join(
                           'data',
                           'tms',
                           'de_pmtg-production.tsv')) as i:
        for l_i, l in enumerate(i):
            line = l.split('\t')
            if l_i == 0:
                header = [w.strip() for w in line]
                continue
            line = [w.strip() for w in line]
            if line[header.index('accuracy')] == '0':
                missing += 1
                continue
            lines.append([w.strip() for w in line])
    print('missing words: {}'.format(missing))
    stims = set([l[header.index('stimulation')] for l in lines])
    if args.stat_approach not in ['residualize', 'bootstrap']:
        conds = {
             'u' : 'unrelated',
             'r' : 'related',
             'ur' : 'all-but-same',
             'urt' : 'all',
             }
    else:
        conds = {
             'urt' : 'all',
             }
    all_sims = dict()
    test_vocab = set()
    for name, cond in conds.items():
        for stim in stims:
            #print(name)
            key = 'de_pmtg-production_{}#{}-{}'.format(args.stat_approach, cond, stim)
            current_cond = [l for l in lines if l[header.index('condition')].strip() in name and \
                                                l[header.index('stimulation')] == stim and \
                                                l[header.index('response')] not in ['0', 'NA'] and \
                                                l[header.index('rt')] not in ['0', 'NA']
                                                ]
            log_rts = [numpy.log10(float(l[header.index('rt')])) for l in current_cond]
            rts = [float(l[header.index('rt')]) for l in current_cond]
            print(rts)
            subjects = [int(l[header.index('sbj')]) for l in current_cond]
            vocab_w_ones = [w for l in current_cond for w in transform_german_word(l[header.index('picture')].split('.')[0])]
            test_vocab = test_vocab.union(set(vocab_w_ones))
            vocab_w_twos = [w for l in current_cond for w in transform_german_word(l[header.index('distractor')])]
            test_vocab = test_vocab.union(set(vocab_w_twos))
            ### picture -> word
            #w_ones = [l[header.index('picture')].split('.')[0] for l in current_cond]
            #w_twos = [l[header.index('distractor')].strip() for l in current_cond]
            ### word -> picture
            w_ones = [l[header.index('distractor')].strip() for l in current_cond]
            w_twos = [l[header.index('picture')].split('.')[0] for l in current_cond]
            all_sims[key]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
    final_sims = reorganize_tms_sims(all_sims)

    return final_sims, test_vocab

def read_de_sem_phon_tms(args):
    sims = dict()
    test_vocab = set()
    lines = list()
    na_lines = list()
    missing = 0
    with open(os.path.join(
                           'data', 
                           'tms', 
                           'de_sem-phon', 
                           'de_tms_sem-phon_ifg.tsv')
                           ) as i:
        for l_i, l in enumerate(i):
            line = l.replace(',', '.').strip().split('\t')
            if 'NA' in line:
                #print(line)
                if 'sem' in line:
                    na_lines.append(line)
                continue
            if l_i == 0:
                header = [w for w in line]
                continue
            #assert len(line)==len(header)
            if '' in line:
                continue
            if len(line) < len(header)-1:
                print('skipping line: {}'.format(line))
                continue
            ### removing trailing spaces
            line = [w.strip() for w in line]
            if line[header.index('ERR')] == '1':
                missing += 1
                continue
            lines.append(line)
    print('missing words: {}'.format(missing))
    print('sem trials containing a NA: {}'.format(len(na_lines)))
    ###
    conditions = set([l[header.index('stim')] for l in lines])
    tasks = set([l[header.index('task')] for l in lines])
    #print(tasks)
    full_sims = dict()
    #for c, name in conditions.items():
    for t in tasks:
        if 'sem' not in t:
            continue
        for c in conditions:
            name = 'de_sem-phon_{}#{}-{}'.format(args.stat_approach, t, c)
            #print(name)
            ###One participant was replaced due to an overall mean error rate of 41.8% - sub 3
            #current_cond = [l for l in lines if l[header.index('stim')] in c and int(l[header.index('subj')])!=3]
            current_cond = [l for l in lines if l[header.index('stim')] in name and l[header.index('task')] in t and int(l[header.index('subj')])!=3]
                    #and l[header.index('utterance')]!='NA']
            tasks = [l[header.index('task')] for l in current_cond]
            assert len(set(tasks)) == 1
            subjects = [int(l[header.index('subj')]) for l in current_cond]
            assert len(set(subjects)) == 24
            #print(subjects)
            rts = [float(l[header.index('RT')]) for l in current_cond]
            log_rts = [numpy.log10(float(l[header.index('RT')])) for l in current_cond]
            vocab_w_ones = [w for l in current_cond for w in transform_german_word(l[header.index('item')].split('.')[0])]
            test_vocab = test_vocab.union(set(vocab_w_ones))
            vocab_w_twos = [w for l in current_cond for w in transform_german_word(l[header.index('utterance')])]
            test_vocab = test_vocab.union(set(vocab_w_twos))
            ### image -> utterance
            w_ones = [l[header.index('item')].split('.')[0] for l in current_cond]
            w_twos = [l[header.index('utterance')] for l in current_cond]
            sims[name]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
    full_sims = reorganize_tms_sims(sims)
    collect_info(full_sims)

    return full_sims, test_vocab

def read_it_social_quantity_tms(args):
    lines = list()
    it_mapper = dict()
    prototypes = {'social' : set(), 'quantity' : set()}
    exclusions = dict()
    with open(os.path.join(
                           'data',
                           'tms',
                           'it_tms_social-quant.tsv')) as i:
        missing = 0
        for l_i, l in enumerate(i):
            line = [w.strip() for w in l.strip().split('\t')]
            if l_i == 0:
                header = [w for w in line]
                continue
            if line[header.index('accuracy')] == '-1':
                #print(line)
                missing += 1
                continue
            sub = line[header.index('subject')]
            if sub not in exclusions.keys():
                exclusions[sub] = dict()
            rt = line[header.index('response_time')]
            cond = line[header.index('condition')]
            cat = line[header.index('target_category')]
            if cat not in exclusions[sub].keys():
                exclusions[sub][cat] = {'cong' : list(), 'incong' : list()}
            w = line[header.index('target')]
            p = line[header.index('prime')]
            if cat[:4] == p[:4]:
                ### congruent
                it_mapper[cat] = p
                if len(cond) == 2:
                    exclusions[sub][cat]['cong'].append(float(rt))
            else:
                if len(cond) == 2:
                    exclusions[sub][cat]['incong'].append(float(rt))
            prototypes[cat].add(w)
            lines.append(line)
    excluded = dict()
    for s, s_data in exclusions.items():
        for c, c_data in s_data.items():
            if numpy.average(c_data['cong']) > numpy.average(c_data['incong']):
                try:
                    excluded[c].append(s)
                except KeyError:
                    excluded[c] = [s]
    print('excluded subjects following original paper: {}'.format(excluded))
    prototypes = {k[:4] : tuple([w for w in v]) for k, v in prototypes.items()}
    print('missing words: {}'.format(missing))
    conds = set([l[header.index('condition')] for l in lines])
    all_sims = dict()
    test_vocab = set()
    for cong in [
                 #'congruent', 'incongruent', 
                 'all',
                 ]:
        if cong == 'congruent':
            cong_lines = [l for l in lines if l[header.index('target_category')][:4]==l[header.index('prime')][:4]]
        elif cong == 'incongruent':
            cong_lines = [l for l in lines if l[header.index('target_category')][:4]!=l[header.index('prime')][:4]]
        elif cong == 'all':
            cong_lines = [l for l in lines]
        for name in conds:
            for marker in [
                           'social', 
                           'quantity', 
                           #'all',
                           ]:
                test_vocab.add(marker)
                if args.stat_approach not in ['residualize', 'bootstrap']:
                    primes = [
                              #'prime-proto', 
                              #'target-proto', 
                              #'target-cat', 
                              #'opposite-target-cat', 
                              'prime-cat',
                              ]
                else:
                    primes = [
                              #'target-cat', 
                              'prime-cat',
                              ]
                for prime in primes:
                    ### in congruent cases primes and targets are the same
                    if 'target' in prime and cong == 'congruent':
                        continue
                    ### removing excluded subjects
                    if marker != 'all':
                        impossible_lines = [l for l in cong_lines if l[header.index('subject')] in excluded[marker]]
                        #print('removed {} lines'.format(len(impossible_lines)))
                        assert len(impossible_lines)>0
                        possible_lines = [l for l in cong_lines if l[header.index('subject')] not in excluded[marker]]
                    else:
                        possible_lines = [l for l in cong_lines]
                    ### not excluding subjects
                    key = 'it_social-quantity_{}#{}-{}-trials-{}_{}'.format(args.stat_approach, marker, cong, prime, name)
                    if marker != 'all':
                        current_cond = [l for l in possible_lines if l[header.index('condition')]==name and l[header.index('target_category')]==marker]
                    else:
                        current_cond = [l for l in possible_lines if l[header.index('condition')]==name]
                    log_rts = [numpy.log10(float(l[header.index('response_time')].replace(',', '.'))) for l in current_cond]
                    rts = [float(l[header.index('response_time')].replace(',', '.')) for l in current_cond]
                    subjects = [int(l[header.index('subject')][1:3]) for l in current_cond]
                    if 'prime' in prime:
                        # ones = primes, twos = targets
                        if prime == 'prime-cat':
                            ### prime -> target
                            w_ones = [l[header.index('prime')].lower() for l in current_cond]
                            vocab_w_ones = [w for ws in w_ones for w in transform_italian_word(ws)] 
                        elif prime == 'prime-proto':
                            w_ones = [prototypes[l[header.index('prime')][:4]] for l in current_cond]
                            vocab_w_ones = [w for ws in w_ones for wz in ws for w in transform_italian_word(wz)] 
                        w_twos = [l[header.index('target')].lower() for l in current_cond]
                    elif 'target' in prime:
                        w_ones = [l[header.index('target')].lower() for l in current_cond]
                        vocab_w_ones = [w for ws in w_ones for w in transform_italian_word(ws)] 
                        ### inverting one and twos: ones = targets, twos = required choice
                        if prime == 'target-cat':
                            ### prime -> target
                            w_twos = [it_mapper[l[header.index('target_category')]] for l in current_cond]
                            vocab_w_twos = [w for ws in w_ones for w in transform_italian_word(ws)] 
                        elif prime == 'target-proto':
                            w_twos = [prototypes[it_mapper[l[header.index('target_category')]][:4]] for l in current_cond]
                            vocab_w_twos = [w for ws in w_ones for wz in ws for w in transform_italian_word(wz)] 
                        elif prime == 'opposite-target-cat':
                            ### prime -> opposite target
                            w_twos = [it_mapper[[k for k in it_mapper.keys() if k!=l[header.index('target_category')]][0]] for l in current_cond]
                            vocab_w_twos = [w for ws in w_ones for w in transform_italian_word(ws)] 
                    test_vocab = test_vocab.union(set(vocab_w_ones))
                    vocab_w_twos = [w for ws in w_twos for w in transform_italian_word(ws)] 
                    test_vocab = test_vocab.union(set(vocab_w_twos))
                    #print(prime)
                    #print(set(vocab_w_twos))
                    print('available subjects per key: {}'.format([key, len(subjects)]))
                    all_sims[key]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
    final_sims = reorganize_tms_sims(all_sims)
    collect_info(final_sims)

    plot_tms_raw_rts(all_sims)
    
    return final_sims, test_vocab


def read_it_distr_learn_tms(args):
    lines = list()
    with open(os.path.join(
                           'data',
                           'tms',
                           'it_distr-learn',
                           'italian_tms_cereb.tsv')) as i:
        missing = 0
        for l_i, l in enumerate(i):
            line = [w.strip() for w in l.strip().split('\t')]
            if l_i == 0:
                header = [w for w in line]
                continue
            if line[header.index('accuracy')] == '0':
                #print(line)
                missing += 1
                continue
            lines.append(line)
    print('missing words: {}'.format(missing))
    conds = set([l[header.index('condition')] for l in lines])
    all_sims = dict()
    all_full_sims = dict()
    related_sims = dict()
    related_full_sims = dict()
    unrelated_sims = dict()
    unrelated_full_sims = dict()
    test_vocab = set()
    for name in conds:
        for m_i, marker in enumerate(['1', '0', 'all']):
            if m_i < 2:
                current_cond = [l for l in lines if l[header.index('condition')]==name and l[header.index('Meaningful')]==marker]
            else:
                current_cond = [l for l in lines if l[header.index('condition')]==name]
            log_rts = [numpy.log10(float(l[header.index('RTs')].replace(',', '.'))) for l in current_cond]
            rts = [float(l[header.index('RTs')].replace(',', '.')) for l in current_cond]
            subjects = [int(l[header.index('Subject')]) for l in current_cond]
            ### noun -> adj
            w_ones = [l[header.index('noun')].lower() for l in current_cond]
            w_twos = [l[header.index('adj')].lower() for l in current_cond]
            vocab_w_ones = [w for ws in w_ones for w in [ws, ws.capitalize()]]
            test_vocab = test_vocab.union(set(vocab_w_ones))
            vocab_w_twos = [w for ws in w_twos for w in [ws, ws.capitalize()]]
            test_vocab = test_vocab.union(set(vocab_w_twos))
            if m_i == 0:
                related_sims[name]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
            elif m_i == 1:
                unrelated_sims[name]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
            elif m_i == 2:
                all_sims[name]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
    related_full_sims = reorganize_tms_sims(related_sims)
    unrelated_full_sims = reorganize_tms_sims(unrelated_sims)
    all_full_sims = reorganize_tms_sims(all_sims)

    final_sims = {'it_distr-learn_{}#all-trials_{}'.format(args.stat_approach, k) : v for k, v in all_full_sims.items()}
    if args.stat_approach not in ['residualize', 'bootstrap']:
        for k, v in related_full_sims.items():
            final_sims['it_distr-learn_{}#related-trials_{}'.format(args.stat_approach, k)] = v
        for k, v in unrelated_full_sims.items():
            final_sims['it_distr-learn_{}#unrelated-trials_{}'.format(args.stat_approach, k)] = v
    collect_info(final_sims)
    
    return final_sims, test_vocab

def reorganize_tms_sims(sims):
    full_sims = dict()
    for n, n_data in sims.items():
        full_sims[n] = dict()
        counter = 0
        for s, ws, rt in n_data:
            if s not in full_sims[n].keys():
                full_sims[n][s] = list()
            full_sims[n][s].append((ws, rt))
    return full_sims

def read_phil_ratings():
    ### first reading ratings
    ratings = dict()
    with open(os.path.join(
                           'data', 
                           'tms', 
                           'de_sound-act', 
                           'phil_annotated_ratings_v9.tsv')) as i:
        for l_i, l in enumerate(i):
            line = l.replace(',', '.').strip().split('\t')
            if l_i == 0:
                header = [w for w in line]
                continue
            ratings[line[0]] = dict()
            sound = line[header.index('Geraeusch')]
            try:
                sound = float(sound)
                ratings[line[0]]['sound'] = sound
            except ValueError:
                pass
            action = line[header.index('Handlung')]
            try:
                action = float(action)
                ratings[line[0]]['action'] = action
            except ValueError:
                pass
            if len(ratings[line[0]].keys()) == 0:
                print('removing line: {}'.format(line[0]))
                del ratings[line[0]]
    return ratings

def read_soundact_prototypes(ratings):
    ### possibilities in task-modelling:
    # centroid overall (all)
    # both positive (both_pos)
    # both negative (both_neg)
    # matched exclusive (action_pos_sound_neg, sound_pos_action_neg)
    # matched non-exclusive (action_pos, sound_pos)
    prototypes = {
                  'action_pos-all' : list(), 
                  'sound_pos-all' : list(),
                  'action_neg-all' : list(),
                  'sound_neg-all' : list(), 
                  'all-all-all' : list(),
                  'all-pos-all' : list(),
                  'all-neg-all' : list(),
                  'action_pos_sound_neg-all' : list(), 
                  'sound_pos_action_neg-all' : list(),
                  }
    with open(os.path.join(
                           'data', 
                           'tms', 
                           'de_sound-act', 
                           'de_tms_pipl.tsv')
                           ) as i:
        for l_i, l in enumerate(i):
            line = l.strip().split('\t')
            if l_i == 0:
                header = [w for w in line]
                continue
            missing = list()
            w = line[header.index('stimulus')]
            ### checking words are all there
            if line[header.index('sound_word')] != 'NA':
                if w not in ratings.keys():
                    missing.append(w)
            if line[header.index('action_word')] != 'NA':
                if w not in ratings.keys():
                    missing.append(w)
            assert len(missing) == 0
            ### distributing prototypes
            ### both
            if line[header.index('action_word')] == '-1' and line[header.index('sound_word')] == '-1':
                prototypes['all-neg-all'].append(line[header.index('stimulus')])
            if line[header.index('action_word')] == '1' and line[header.index('sound_word')] == '1':
                prototypes['all-pos-all'].append(line[header.index('stimulus')])
            ### exclusive
            if line[header.index('action_word')] == '1' and line[header.index('sound_word')] == '-1':
                prototypes['action_pos_sound_neg-all'].append(line[header.index('stimulus')])
            if line[header.index('action_word')] == '-1' and line[header.index('sound_word')] == '1':
                prototypes['sound_pos_action_neg-all'].append(line[header.index('stimulus')])
            ### inclusive
            if line[header.index('action_word')] == '1':
                prototypes['action_pos-all'].append(line[header.index('stimulus')])
            if line[header.index('action_word')] == '-1':
                prototypes['action_neg-all'].append(line[header.index('stimulus')])
            if line[header.index('sound_word')] == '1':
                prototypes['sound_pos-all'].append(line[header.index('stimulus')])
            if line[header.index('sound_word')] == '-1':
                prototypes['sound_neg-all'].append(line[header.index('stimulus')])
            ### if it isn't lexical decision, drop it
            if 'lexical_decision' in line:
                continue
            ### everything
            prototypes['all-all-all'].append(line[header.index('stimulus')])
    prototypes = {k : set(v) for k, v in prototypes.items()}
    #for k, v in prototypes.items():
    #    print('\n')
    #    print('prototypes for {}'.format(k))
    #    print(v)
    ### using only 0.1, 0.5 highest rated words
    top_tenned = [
          'action_pos-all', 
          'sound_pos-all',
          'all-pos-all',
          'all-neg-all',
          'action_pos_sound_neg-all', 
          'sound_pos_action_neg-all',
          ]
    for tenned in top_tenned:
        ### top ten percent
        for percent, mult in [('ten', 0.1), ('fifty', 0.5)]:
            ten_percent = int(len(prototypes[tenned])*mult)
            ### only sound
            if 'sound_pos' in tenned:
                top = [s[0] for s in sorted([(k, ratings[k]['sound']) for k in prototypes[tenned]], key=lambda item : item[1], reverse=True)][:ten_percent]
            ### only action
            if 'action_pos' in tenned:
                top = [s[0] for s in sorted([(k, ratings[k]['action']) for k in prototypes[tenned]], key=lambda item : item[1], reverse=True)][:ten_percent]
            ### both
            else:
                top_s = [s[0] for s in sorted([(k, ratings[k]['sound']) for k in prototypes[tenned]], key=lambda item : item[1], reverse=True)]
                top_a = [s[0] for s in sorted([(k, ratings[k]['action']) for k in prototypes[tenned]], key=lambda item : item[1], reverse=True)]
                top = [s[0] for s in sorted([(k, top_a.index(k)+top_s.index(k)) for k in prototypes[tenned]], key=lambda item : item[1])][:ten_percent]
            #if percent == 'fifty':
            #    print('top-fifty percent prototypes for {}:\n'.format(tenned))
            #    print(sorted(top))
            #    print('\n')
            prototypes['{}-top{}'.format(tenned[:-4], percent)] = top
    prototypes = {k : tuple(v) if type(v)!=tuple else v for k, v in prototypes.items()}
    return prototypes

def return_proto_words(task, proto_mode, prototypes):
    val = proto_mode.split('-')[-1]
    if proto_mode in [
                      'all-all-all', 
                      'all-pos-all', 
                      'all-neg-all',
                      'all-pos-topten', 
                      'all-pos-topfifty', 
                      ]:
        words = prototypes['{}'.format(proto_mode)]
    else:
        if 'incl' in proto_mode:
            if 'matched' in proto_mode:
                ### sound
                if task == 'Geraeusch':
                    words = prototypes['sound_pos-{}'.format(val)]
                elif task == 'Handlung':
                    words = prototypes['action_pos-{}'.format(val)]
                else:
                    raise RuntimeError()
            elif 'opposite' in proto_mode:
                ### sound
                if task == 'Geraeusch':
                    words = prototypes['sound_neg-{}'.format(val)]
                elif task == 'Handlung':
                    words = prototypes['action_neg-{}'.format(val)]
                else:
                    raise RuntimeError()
            else:
                raise RuntimeError()
        elif 'excl' in proto_mode:
            if 'matched' in proto_mode:
                ### sound
                if task == 'Geraeusch':
                    words = prototypes['sound_pos_action_neg-{}'.format(val)]
                elif task == 'Handlung':
                    words = prototypes['action_pos_sound_neg-{}'.format(val)]
                else:
                    raise RuntimeError()
            elif 'opposite' in proto_mode:
                ### sound
                if task == 'Geraeusch':
                    words = prototypes['action_pos_sound_neg-{}'.format(val)]
                elif task == 'Handlung':
                    words = prototypes['sound_pos_action_neg-{}'.format(val)]
                else:
                    raise RuntimeError()
            else:
                raise RuntimeError()
        else:
            raise RuntimeError()
    return words

def read_de_sound_act_tms(args):
    ratings = read_phil_ratings()
    prototypes = read_soundact_prototypes(ratings)
    ### reading dataset
    lines = list()
    errs = 0
    with open(os.path.join(
                           'data', 
                           'tms', 
                           'de_sound-act', 
                           'de_tms_pipl.tsv')
                           ) as i:
        for l_i, l in enumerate(i):
            line = l.strip().split('\t')
            if l_i == 0:
                header = [w for w in line]
                continue
            missing = list()
            w = line[header.index('stimulus')]
            if line[header.index('sound_word')] != 'NA':
                if w not in ratings.keys():
                    missing.append(w)
            assert len(missing) == 0
            if 'lexical_decision' in line:
                continue
            if line[header.index('expected_response')] != line[header.index('response')]:
                errs += 1
                continue
            lines.append(line)
    print('number of error trials: {}'.format(errs))

    if args.stat_approach not in ['residualize', 'bootstrap']:
        proto_modes = [
                 #'all-all-all', 
                 'all-pos-all',
                 #'all-pos-topten',
                 #'all-pos-topfifty',
                 #'all-neg-all',
                 'matched-excl-all',
                 #'matched-excl-topten',
                 #'matched-excl-topfifty',
                 'matched-incl-all',
                 #'matched-incl-topten',
                 #'matched-incl-topfifty',
                 'opposite-excl-all',
                 #'opposite-excl-topten',
                 #'opposite-excl-topfifty',
                 'opposite-incl-all',
                 #'opposite-incl-topten',
                 #'opposite-incl-topfifty',
                 'matched-cat-word-all',
                 'opposite-cat-word-all',
                 ]
    else:
        proto_modes = [
                 'all-pos-all',
                 'matched-excl-all',
                 'matched-incl-all',
                 ]
    sims = dict()
    test_vocab = set()
    conditions = set([l[header.index('condition')] for l in lines])
    tasks = set([l[header.index('task')] for l in lines])
    assert len(tasks) == 2
    ### everything together
    for proto_mode in proto_modes:
        for c in conditions:
            key = 'de_sound-act-aggregated_{}#{}_all_{}'.format(args.stat_approach, proto_mode, c)
            ### both tasks together
            current_cond = [l for l in lines if l[header.index('condition')]==c]
            subjects = [int(l[header.index('subject')]) for l in current_cond]
            rts = [float(l[header.index('rt')]) for l in current_cond]
            log_rts = [numpy.log10(float(l[header.index('log_rt')])) for l in current_cond]
            ### with prototyping, we actually use words as first items
            #w_ones = [l[header.index('task')] for l in current_cond]
            if 'cat-word' not in proto_mode:
                w_ones = [return_proto_words(l[header.index('task')], proto_mode, prototypes) for l in current_cond]
                all_w_ones = [transform_german_word(w) for ws in w_ones for w in ws]
            else:
                if 'matched' in proto_mode:
                    w_ones = [tuple(transform_german_word(l[header.index('task')])) for l in current_cond]
                else:
                    w_ones = [tuple(transform_german_word([tsk for tsk in tasks if tsk!=l[header.index('task')]][0])) for l in current_cond]
                all_w_ones = [w for w in w_ones]
            test_vocab = test_vocab.union(set([w for ws in all_w_ones for w in ws]))
            ### these are the words subjects actually saw
            vocab_w_twos = [w for l in current_cond for w in transform_german_word(l[header.index('stimulus')])]
            test_vocab = test_vocab.union(set(vocab_w_twos))
            ### prototype -> task
            w_twos = [l[header.index('stimulus')] for l in current_cond]
            #print(log_rts)
            sims[key]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
            for t in tasks:
                if 'cat-word' not in proto_mode:
                    #print('\n')
                    print('number of prototypes for {}, {}'.format(proto_mode, t))
                    print(len(return_proto_words(t, proto_mode, prototypes)))
                key = 'de_sound-act-aggregated_{}#{}_{}_{}'.format(args.stat_approach, proto_mode, t, c)
                ### separate tasks
                current_cond = [l for l in lines if l[header.index('condition')]==c and l[header.index('task')]==t]
                subjects = [int(l[header.index('subject')]) for l in current_cond]
                log_rts = [numpy.log10(float(l[header.index('log_rt')])) for l in current_cond]
                exps = [l[header.index('expected_response')] for l in current_cond]
                rts = [float(l[header.index('rt')]) for l in current_cond]
                #vocab_w_ones = [w for l in current_cond for w in transform_german_word(l[header.index('task')])]
                #test_vocab = test_vocab.union(set(vocab_w_ones))
                ### with prototyping, we actually use words as first items
                #w_ones = [l[header.index('task')] for l in current_cond]
                if 'cat-word' not in proto_mode:
                    w_ones = [return_proto_words(l[header.index('task')], proto_mode, prototypes) for l in current_cond]
                    all_w_ones = [transform_german_word(w) for ws in w_ones for w in ws]
                else:
                    if 'matched' in proto_mode:
                        w_ones = [tuple(transform_german_word(l[header.index('task')])) for l in current_cond]
                    else:
                        w_ones = [tuple(transform_german_word([tsk for tsk in tasks if tsk!=l[header.index('task')]][0])) for l in current_cond]
                    all_w_ones = [w for w in w_ones]
                test_vocab = test_vocab.union(set([w for ws in all_w_ones for w in ws]))
                vocab_w_twos = [w for l in current_cond for w in transform_german_word(l[header.index('stimulus')])]
                test_vocab = test_vocab.union(set(vocab_w_twos))
                w_twos = [l[header.index('stimulus')] for l in current_cond]
                #print(w_twos)
                #sims['{}_{}'.format(t, c)]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
                #print(log_rts)
                sims[key]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
    ### if it's not fasttext, we just use aggregate
    if args.model == 'fasttext':
        ### everything split
        for action, action_n in [('lo-', '-1'), ('hi-', '1')]:
            for sound, sound_n in [('lo-', '-1'), ('hi-', '1')]:
                for proto_mode in proto_modes:
                    for c in conditions:
                        key = 'de_sound-act-detailed_{}#{}_all-{}sound-{}action_{}'.format(args.stat_approach, proto_mode, sound, action, c)
                        ### both tasks together
                        current_cond = [l for l in lines if l[header.index('condition')]==c and l[header.index('action_word')]==action_n and l[header.index('sound_word')]==sound_n]
                        subjects = [int(l[header.index('subject')]) for l in current_cond]
                        rts = [float(l[header.index('rt')]) for l in current_cond]
                        log_rts = [numpy.log10(float(l[header.index('log_rt')])) for l in current_cond]
                        ### with prototyping, we actually use words as first items
                        #w_ones = [l[header.index('task')] for l in current_cond]
                        if 'cat-word' not in proto_mode:
                            w_ones = [return_proto_words(l[header.index('task')], proto_mode, prototypes) for l in current_cond]
                            all_w_ones = [transform_german_word(w) for ws in w_ones for w in ws]
                        else:
                            if 'matched' in proto_mode:
                                w_ones = [tuple(transform_german_word(l[header.index('task')])) for l in current_cond]
                            else:
                                w_ones = [tuple(transform_german_word([tsk for tsk in tasks if tsk!=l[header.index('task')]][0])) for l in current_cond]
                            all_w_ones = [w for w in w_ones]
                        test_vocab = test_vocab.union(set([w for ws in all_w_ones for w in ws]))
                        ### these are the words subjects actually saw
                        vocab_w_twos = [w for l in current_cond for w in transform_german_word(l[header.index('stimulus')])]
                        test_vocab = test_vocab.union(set(vocab_w_twos))
                        w_twos = [l[header.index('stimulus')] for l in current_cond]
                        sims[key]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
                        for t in tasks:
                            if 'cat-word' not in proto_mode:
                                #print('\n')
                                print('number of prototypes for {}, {}'.format(proto_mode, t))
                                print(len(return_proto_words(t, proto_mode, prototypes)))
                            key = 'de_sound-act-detailed_{}#{}_{}-{}sound-{}action_{}'.format(args.stat_approach, proto_mode, t, sound, action, c)
                            ### separate tasks
                            current_cond = [l for l in lines if l[header.index('condition')]==c and l[header.index('task')]==t and l[header.index('action_word')]==action_n and l[header.index('sound_word')]==sound_n]
                            subjects = [int(l[header.index('subject')]) for l in current_cond]
                            log_rts = [numpy.log10(float(l[header.index('log_rt')])) for l in current_cond]
                            exps = [l[header.index('expected_response')] for l in current_cond]
                            rts = [float(l[header.index('rt')]) for l in current_cond]
                            #vocab_w_ones = [w for l in current_cond for w in transform_german_word(l[header.index('task')])]
                            #test_vocab = test_vocab.union(set(vocab_w_ones))
                            ### with prototyping, we actually use words as first items
                            #w_ones = [l[header.index('task')] for l in current_cond]
                            if 'cat-word' not in proto_mode:
                                w_ones = [return_proto_words(l[header.index('task')], proto_mode, prototypes) for l in current_cond]
                                all_w_ones = [transform_german_word(w) for ws in w_ones for w in ws]
                            else:
                                if 'matched' in proto_mode:
                                    w_ones = [tuple(transform_german_word(l[header.index('task')])) for l in current_cond]
                                else:
                                    w_ones = [tuple(transform_german_word([tsk for tsk in tasks if tsk!=l[header.index('task')]][0])) for l in current_cond]
                                all_w_ones = [w for w in w_ones]
                            test_vocab = test_vocab.union(set([w for ws in all_w_ones for w in ws]))
                            vocab_w_twos = [w for l in current_cond for w in transform_german_word(l[header.index('stimulus')])]
                            test_vocab = test_vocab.union(set(vocab_w_twos))
                            w_twos = [l[header.index('stimulus')] for l in current_cond]
                            #sims['{}_{}'.format(t, c)]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
                            sims[key]= [(sub, (w_one, w_two), rt) for sub, w_one, w_two, rt in zip(subjects, w_ones, w_twos, log_rts)]
    full_sims = reorganize_tms_sims(sims)
    collect_info(full_sims)

    return full_sims, test_vocab, prototypes

def collect_info(full_sims):
    labels = set(full_sims.keys())
    subjects = set([s for subs in full_sims.values() for s in subs.keys()])
    trials = set([len(set([ws[0] for ws in s])) for subs in full_sims.values() for s in subs.values()])
    print('labels: ')
    print(labels)
    print('subjects: ')
    print(subjects)
    print('trials: ')
    print(trials)
