# FUNCTIONS TO GENERATE NEURAL MODEL INPUTS FOR FINE-TUNING AND TESTING
# ---------------------------------------------------------------------
from encoding import DatasetBuilder
from itertools import product

import sys

# FUNCTIONS
# my range
def r(s,e):
    return [str(i) for i in range(s, e+1)]

# unseen lengths 
def unseen(lengths, exp, data, u=5):
    u_lengths = {'com_train': {k:v[u:] for k,v in lengths.items()},
                 'com_test': {k:v[:u] for k,v in lengths.items()},
                 'rec_train': {k:v[:-u] for k,v in lengths.items()},
                 'rec_test': {k:v[-u:] for k,v in lengths.items()}
                }

    return u_lengths[f'{exp}_{data}']

# arguments to create training/validation datasets
def args_train_val(model, task, exp, fraction):
    L = {'ove':None, 'com':unseen(lengths[task], 'com', 'train'), 'rec':unseen(lengths[task], 'rec', 'train')}

    return {f'{t}_{model}_{task}_{exp}': {'datasets':data,
                                          'fraction':fraction,
                                          'clengths':L[exp],
                                         } for t,data in [('train', train_ds), ('val', val_ds)]}

# arguments to create test datasets
def args_test(model, task, exp, fraction):
    L = {'ove':None, 'com':unseen(lengths[task], 'com', 'test'), 'rec':unseen(lengths[task], 'rec', 'test')}
    if exp == 'rec':
        DS = test2_ds
    else:
        DS =  test_ds

    return {f'{ds}_{s}_{model}_{task}_{exp}': {'datasets':[ds],
                                               'fraction':fraction,
                                               'clengths':L[exp],
                                               'subs':[str(s)]
                                              } for ds in DS for s in r(1,3)}

# DATA SPLIT
train_ds = [f'ds_{i}' for i in r(1,9) + r(11,19) + r(21,29)]
val_ds = ['ds_10', 'ds_20', 'ds_30']
test_ds = [f'tds_S_{i}' for i in r(1,10)] + [f'tds_M_{i}' for i in r(1,10)] + [f'tds_L_{i}' for i in r(1,10)]
test2_ds = [f'tds_L2_{i}' for i in r(1,60)]

# STRUCTURE LENGTHS (BY TASK)
lengths = {'pbc': {'1':r(1,12), '3':r(0,20), '4':r(1,13), '5':r(1,20), '7':r(0,15)},
           'ps': {'1':r(0,12), '2':r(1,10), '3':r(0,20), '4':r(1,13), '5':r(1,20), '6':r(0,17), '7':r(0,15)}}

# MAIN FUNCTION
def generate(save_path, model=['T5', 'GPT'], task=['pbc', 'ps'], exp=['ove', 'com', 'rec'], data_split=['train', 'test']):
    for m,t,e,d in product(model, task, exp, data_split):
        # set default test data split (according to task)
        if d == 'test' and e == 'rec':
            d = 'test2'

        # set default number of permutations (according to the data split)
        permutations = 3 if d == 'train' else 1        

        # set default data fraction (according to data split and model)
        if d == 'train':
            if m == 'T5':
                fraction = 0.8
            else:
                fraction = 0.25
        else:
            fraction = 1

        # get arguments
        if d == 'train':            
            P = args_train_val(m, t, e, fraction)
        else:
            P = args_test(m, t, e, fraction) 

        # compute inputs
        for name,args in P.items():
            data = DatasetBuilder(name=name, 
                                  source=f"structures/{d}", 
                                  save_path=save_path,
                                  substitutions=f"substitutions/substitutions_{d}.json", 
                                  permutations=permutations, 
                                  task=t, 
                                  model=m)
            data.build(**args)

if __name__ == "__main__":
    save_path = sys.argv[1]
    model = sys.argv[2]
    task = sys.argv[3]
    exp = sys.argv[4]

    generate(save_path=save_path, model=[model], task=[task], exp=[exp])
