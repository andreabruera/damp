import matplotlib
import numpy
import os
import random
import re

from matplotlib import colormaps, pyplot
from tqdm import tqdm

results = dict()

with tqdm() as counter:
    for root, direc, fz in os.walk(
                              os.path.join(
                                  'test_results',
                                  #'old_results',
                                  )):
        for f in fz:
            if 'raw' in root: 
                continue
            elif 'sym' in root: 
                continue
            elif 'lm' in root:
                continue
            #elif 'pt' in root:
            #    continue
            #elif 'll' in root:
            #    continue
            elif 'concept' in root:
                continue
            elif 'aligned' in root:
                continue
            parts = root.split('/')
            #['test_results', 'rsa', 'bootstrap', 'spearman', 'it', 'llama-3b_layer-21', 'llama-3b_layer-21']
            approach = parts[1]
            stat_approach = parts[2]
            evaluation = parts[3]
            setup = '-'.join([approach, stat_approach, evaluation])
            if setup not in results.keys():
                results[setup] = dict()
            lang = parts[4]
            if lang not in results[setup].keys():
                results[setup][lang] = dict()
            with open(os.path.join(root, f)) as i:
                for l in i:
                    #print(l)
                    line = l.strip().split('\t')
                    #lang = line[0]
                    model = line[1]
                    if 'ppmi-vecs' in model or 'layer' in model:
                        print(model)
                        if 'lm' in model or 'llama' in model or 'pt' in model:
                            num = float(model.split('-')[-1])
                            num = num*10000
                            short_model = '{}-vecs'.format(model.split('_')[0])
                        else:
                            num = float(model.split('_')[-2])
                            if 'wiki' in model:
                                short_model = '_'.join(model.split('_')[2:-2])
                            else:
                                short_model = '_'.join(model.split('_')[1:-2])
                    #print(model)
                    #if 'fasttext' not in model and 'mitchell' not in model and 'concept' not in model and 'prob' not in model and 'surprisal' not in model and 'length' not in model and 'best' not in model:
                    dataset = line[2]
                    if 'en_men' in dataset:
                        task = 'simrel_norms'
                    elif '999' in dataset:
                        task = 'simrel_norms'
                    elif '353' in dataset:
                        task = 'simrel_norms'
                    elif 'mitchell' in dataset:
                        task = 'fmri'
                    elif 'fern' in dataset:
                        task = 'fmri'
                    elif 'dirani' in dataset:
                        task = 'meeg'
                    elif 'kaneshiro' in dataset:
                        task = 'meeg'
                    elif 'abstract' in dataset:
                        task = 'fmri'
                    elif 'lexical' in dataset:
                        task = 'behavioural'
                    elif 'naming' in dataset:
                        task = 'behavioural'
                    elif 'abs-conc-decision' in dataset:
                        task = 'behavioural'
                    elif 'sem-phon' in dataset:
                        task = 'tms'
                    elif 'sound-act' in dataset:
                        task = 'tms'
                    elif 'pmtg-prod' in dataset:
                        task = 'tms'
                    elif 'distr-learn' in dataset:
                        task = 'tms'
                    elif 'social-quant' in dataset:
                        task = 'tms'
                    else:
                        #continue
                        raise RuntimeError
                    if task not in results[setup][lang].keys():
                        results[setup][lang][task] = {dataset : dict()}
                    if dataset not in results[setup][lang][task].keys():
                        results[setup][lang][task][dataset] = dict()
                    res = numpy.array(line[3:], dtype=numpy.float32)
                    #if 'fasttext' not in model and 'mitchell' not in model and 'concept' not in model and 'prob' not in model and 'surprisal' not in model and 'length' not in model and 'best' not in model and ':
                    if 'ppmi-vecs' in model or 'layer' in model:
                        if short_model not in results[setup][lang][task][dataset].keys():
                            results[setup][lang][task][dataset][short_model] = dict()
                        results[setup][lang][task][dataset][short_model][num] = res
                    else:
                        results[setup][lang][task][dataset][model] = res
                    counter.update(1)

with tqdm() as counter:
    for setup, setup_results in results.items():
        for lang, l_res in setup_results.items():
            print(lang)
            for general_task, task_res in l_res.items():
                print(general_task)
                for task, t_res in task_res.items():
                    print(task)
                    ### creating folder
                    specific_task = task.split('_')[min(1, len(task.split('_'))-1)]
                    assert len(specific_task) > 0
                    folder = os.path.join(
                                          'test_lineplots',
                                          setup,
                                          lang, 
                                          general_task,
                                          specific_task,
                                          )
                    if not os.path.exists(folder):
                        os.makedirs(folder, exist_ok=True)
                    '''
                    if 'en_men' in task:
                        pass
                    elif '999' in task:
                        ymin = -.05
                        ymax = .9
                    elif '353' in task:
                        ymin = -.05
                        ymax = .9
                    elif 'fern' in task:
                        ymin = -.05
                        ymax = .2
                    elif 'dirani' in task:
                        ymin = -.02
                        ymax = .15
                    elif 'kaneshiro' in task:
                        ymin = -.02
                        ymax = .1
                    elif 'abstract' in task:
                        pass
                    elif 'anew-lexical' in task:
                        ymin = -0.05
                        ymax = 0.4
                    elif 'anew-word' in task:
                        ymin = -0.05
                        ymax = 0.2
                    elif 'deaf' in task:
                        ymin = -0.05
                        ymax = 0.1
                    elif 'blind' in task:
                        ymin = -0.05
                        ymax = 0.2
                    elif 'lexical' in task:
                        ymin = -0.05
                        ymax = 0.35
                    elif 'naming' in task:
                        ymin = -0.05
                        ymax = 0.15
                    elif 'abs-conc-decision' in task:
                        ymin = -.1
                        ymax = .1
                    elif 'social-quant' in task:
                        ymin = -.25
                        ymax = .25
                    elif 'mitchell' in task:
                        ymin = -.05
                        ymax = .15
                    elif 'sem-phon' in task:
                        ymin = -0.05
                        ymax = 0.5
                    elif 'sound-act' in task:
                        ymin = -.3
                        ymax = .3
                    elif 'distr-learn' in task:
                        ymin = -.1
                        ymax = .3
                    elif 'pmtg-prod' in task:
                        ymin = -0.02
                        ymax = 0.36
                    '''
                    fig, ax = pyplot.subplots(
                                              constrained_layout=True,
                                              figsize=(20, 10),
                                              )
                    ### dividing into lines and regular values
                    fts = [k for k in t_res.keys() if 'fast' in k or 'concept' in k or 'length' in k]
                    surprs = [k for k in t_res.keys() if 'surpr' in k or 'best' in k]
                    abss = [k for k in t_res.keys() if 'abs-prob' in k]
                    mitchs = [k for k in t_res.keys() if 'mitch' in k and 'rowincol' not in k]
                    xs = [k for k in t_res.keys() if 'vecs' in k]
                    others = {
                              k : sorted(vals.items(), key=lambda item : item[0]) \
                                      for k, vals in t_res.items() \
                                      if 'vecs' in k
                                      }
                    #print([k for k in others.keys() if 'lm' in k])
                    ### plotting horizontal lines
                    all_vals = [val[0] for v in others.values() for val in v] + [0.]
                    ax.hlines(
                              y=0,
                              xmin=-.1,
                              xmax=max(all_vals)+.1,
                              color='black',
                              )
                    alls = list()
                    ### fasttext
                    #print('fasttext')
                    for ds, style in [(fts, 'solid'), (surprs, 'dotted'), (abss, 'dashdot')]: 
                        ft_colors = matplotlib.colormaps['hsv'](numpy.linspace(0, 1, len(ds)+1))
                        for ft_i, ft in enumerate(ds):
                            print(ft)
                            #style = random.choice(['solid', 'dashdot', 'dotted', ])
                            color = ft_colors[ft_i]
                            y = numpy.average(t_res[ft])
                            alls.append(y)
                            ax.hlines(
                                  y=y,
                                  xmin=-.1,
                                  xmax=max(all_vals)+.1,
                                  label=ft,
                                  linestyles=style,
                                  color=color,
                                  )
                    ### mitchell dimensions
                    #print('mitchell')
                    for mitch, col in zip(mitchs, numpy.linspace(0, 1, len(mitchs))):
                        y = numpy.average(t_res[mitch])
                        alls.append(y)
                        ax.hlines(
                                  y=y,
                                  xmin=-.1,
                                  xmax=max(all_vals)+.1,
                                  label=mitch,
                                  color=matplotlib.colormaps['cividis'](col)
                                  )
                    ### count models
                    ### we use rainbow as a set of colours to sample from
                    #print('count')
                    colors = {k : v for k, v in zip(others.keys(), matplotlib.cm.rainbow(numpy.linspace(0, 1, len(others.keys()))))}
                    for case, sort_freqs in others.items():
                        xs_freqs = [v[0] for v in sort_freqs]
                        ys_freqs = [v[1] for v in sort_freqs]
                        ys = [numpy.average(v) for v in ys_freqs]
                        alls.extend(ys)
                        assert len(ys) == len(xs_freqs)
                        ax.plot(
                                xs_freqs,
                                ys,
                                label=case,
                                color=colors[case]
                                )
                        ax.scatter(
                                xs_freqs,
                                ys,
                                s=50,
                                marker='d',
                                edgecolors='white',
                                linewidths=1,
                                color=colors[case],
                                zorder=3
                                )
                    ### checking the plot is actually not empty
                    if len(alls) == 0:
                        continue
                    ### setting plot limits
                    '''
                    corr = numpy.nanstd(alls)
                    ax.set_ylim(
                                #bottom=min(0, min(alls))-corr, 
                                #top=max(0, max(alls))+corr
                                bottom=ymin,
                                top=ymax,
                                )
                    '''
                    ax.set_xlim(
                                #left=0., 
                                right=500000,
                                )
                    #pyplot.ylabel('Pearson correlation')
                    #pyplot.ylabel('Spearman correlation')
                    pyplot.ylabel(evaluation.replace('_', ' '))
                    ### legend
                    pyplot.legend(
                            ncols=5,
                            loc=8,
                            fontsize=10,
                            )
                    ### saving figure
                    #print('saving')
                    pyplot.savefig(
                                    os.path.join(
                                                 folder,
                                         '{}.jpg'.format(task))
                                    )
                    #print('saved')
                    pyplot.clf()
                    pyplot.close()
                    print(os.path.join(folder, task))
                    counter.update(1)
