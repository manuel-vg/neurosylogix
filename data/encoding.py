import random
import utils

###########################
# STRUCTURE ENCODER CLASS #
###########################
class StructureEncoder:
    def __init__(self, source, substitutions, permutations, task, model):
        self.task = task    # ps: premise selection | pbc: proof by contradiction
        self.model = model  # T5 | GPT
        self.types = self._default_types(task)
        self.data = self._data_augmentation(source, substitutions, permutations)
    
    def _default_types(self, task):
        types = {'ps': ['1','2','3','4','5','6','7'],
                 'pbc': ['1','3','4','5','7']}      
        return types[task]

    # SYMBOLIC->TEXT INTERFACE
    # function to split a formula
    def SF(self, f):
        q = f[0]
        i = f.find('x')
        j = f.rfind('x')
        a = f[i:j]
        b = f[j:]        
        return (q, a, b)

    # function to encode a formula
    def _formula_encoder(self, f, V):
        '''
        Input: symbolic formula (e.g., Ax1x2)
        Output: textual formula using "pseudowords" (e.g., "all preac are ramen")
        '''
        d = {'T5': {'l':'', 'r':''}, 'GPT': {'l':'{', 'r':'}'}} # delimiters
        pr = {'A':('all', 'are'), 'E':('no', 'are'), 'I':('some', 'are'), 'O':('some', 'are not')}
        q, s, p = self.SF(f)                
        l = d[self.model]['l']
        r = d[self.model]['r']

        return f'{pr[q][0]} {l}{V[s]}{r} {pr[q][1]} {l}{V[p]}{r}' 
    
    # SORT INFERENCES BY TYPE AND LENGTH
    # function to unfold A-chains
    def _unfold(self, inf_premises, a_chains):
        unfolded = []
        for p in inf_premises:
            if p in a_chains.keys():
                unfolded += a_chains[p]
            else:
                unfolded.append(p)
        return unfolded
    
    # returns the number of A-formulas of an A-chain
    def _length(self, a_chains, f):
        if f in a_chains.keys():
            return len(a_chains[f])
        return 1
    
    def _get_pairs(self, d, V):
        '''
        returns a dict of encoded pairs: (hypothesis, target) 
        sorted by types and lengths
        '''
        sbl = {t:{} for t in [str(t) for t in self.types]} # only valid syllogisms (types 1-7)
        a_chains = d['KB']['a_chains']
        
        for t in sbl.keys():
            for p in d['inferences'][t]:
                # n: number of A-formulas (length)
                n = sum([self._length(a_chains, f) for f in p[0] if f.startswith('A')])
                
                # encoded pair
                hyp = self._formula_encoder(p[1], V)
                if self.task == 'ps':
                    tar = [self._formula_encoder(f, V) for f in self._unfold(p[0], a_chains)]
                else:
                    F = d['pbc'][t].get(p[1]) # refutation formulas F (for pbc task)
                    if F is None: 
                        continue
                    tar = [self._formula_encoder(f, V) for f in F]
                pair = [hyp, tar]
                
                if n in sbl[t].keys():
                    sbl[t][n].append(pair) 
                else:
                    sbl[t][n] = [pair]
        return sbl
    
    # CREATES THE AUGMENTED DATASET
    def _data_augmentation(self, source, substitutions, permutations):
        # load sets of susbtitutions
        subs = utils.open_json(substitutions)
        
        # create augmented dataset dict structure
        DS = {kb:{s:{} for s in S.keys()} for kb,S in subs.items()}
        
        # set seed for permutation
        start = 1
        
        # iterate over all KB structures 
        for kb,S in DS.items():
            # load structure
            ds = utils.open_json(f"{source}/{kb}.json")
            
            # iterate over term substitutions
            for s,P in S.items():
                V = subs[kb][s] # get substitutions
                
                # encoded KB (set of premises)
                e_kb = [self._formula_encoder(f, V) for f in ds['KB']['premises']]
                
                # input/output pairs
                pairs = self._get_pairs(ds, V)
                
                # iterate over premise permutations
                for p in range(1, permutations+1):                    
                    # new permutation of premises
                    shuffled_e_kb = list(e_kb)
                    random.seed(start)
                    start += 1                    
                    random.shuffle(shuffled_e_kb)

                    # update main dictionary
                    DS[kb][s][str(p)] = {'KB': shuffled_e_kb, **pairs}              
                
        return DS    

#########################
# DATASET BUILDER CLASS #
#########################
class DatasetBuilder:    
    def __init__(self, name, source, save_path, substitutions, permutations=1, task='ps', model='T5'):
        self.name = name    # vectorizer name
        self.save_path = save_path  # path to save datasets 
        # Encoder of KB structures + data augmentation
        self.SE = StructureEncoder(
            source,         # path of kb structures (to a folder containing json files)
            substitutions,  # path of substitutions (to a json file)
            permutations,   # number of premises permutations
            task,           # ps: premise selection | pbc: proof by contradiction
            model           # T5 | GPT
        )  

    def _create_inputs(self, ds_id, kb, pairs, fraction, type, length, half='first'):
        vec = []
        random.seed(7)
        L = list(pairs)
        split = round(len(pairs)*fraction)
        random.shuffle(L)
        io_pairs = L[:split] # if fraction < 1, then get the first part        
        
        for p in io_pairs:
            vec.append({'kb': ', '.join(kb), 
                        'h': p[0], 
                        'target': ', '.join(p[1]), 
                        'ds_id': ds_id,
                        'type': type, 
                        'length': length})

        return vec      

    def _compute_dictionary(self, ids, fraction, clengths):
        inputs = []
        for (d,s,p) in ids:
            ds_id = f"{d}_{s}_{p}"      # data structure  id 
            ds = self.SE.data[d][s][p]  # data structure
            kb = ds['KB']               # knowledge base (set of premises)

            for t in self.SE.types:  
                if clengths: # custom lenghts
                    lengths = [int(l) for l in clengths[t] if int(l) in ds[t].keys()]                
                else: # all lengths from type 't'
                    lengths = [l for l in ds[t].keys()]

                for l in lengths:
                    inputs += self._create_inputs(ds_id, kb, ds[t][l], fraction, t, l)
        
        return inputs

    # MAIN FUNCTION TO BUILD AND SAVE INPUTS FOR NEURAL NETWORKS
    def build(self, datasets, subs=None, fraction=1, clengths=None):
        # filter by datasets
        ids = [(d,s,p) for d in self.SE.data.keys() for s in self.SE.data[d] for p in self.SE.data[d][s] if d in datasets]      
        # filter by substitutions (for the evaluation process)
        if subs:
            ids = [(d,s,p) for (d,s,p) in ids if s in subs]                  

        v_data = self._compute_dictionary(ids=ids, fraction=fraction, clengths=clengths)

        # save data as json file
        utils.save_json(f"{self.save_path}/{self.name}.json", v_data)