from prover import Inference, LaTeX_Inference
import json
import os

class Precomputed_Assistant:
    def __init__(self, filename):
        self.predictions = self._load_file(filename)

    def _load_file(self, filename):
        f = open(filename)
        return json.load(f)

    def predict(self, assistant, query):
        pred = self.predictions[query]
        if assistant in pred.keys():
            return pred[assistant]
        return ['unknown']

class Hybrid_Model:
    c = 'x'
    def __init__(self, name, model, datasets):        
        self.name = name
        self.model = model
        self.datasets = self._load_file(datasets)
        self.d = self._set_delimiters()
        self._set_assistant()

    def _load_file(self, filename):
        f = open(filename)
        return json.load(f)

    # Set delimiters (for GPT model)
    def _set_delimiters(self):
        d = {'L':'', 'R':''}
        if self.model[:3] == "GPT":
            d['L'] = '{'
            d['R'] = '}'
        return d    

    # Load precomputed assistants
    def _set_assistant(self):        
        if self.model == "Symbolic":
            self.assistants = None
            print('Prover: symbolic model (no neural assistance)')        
        else:
            self.assistants = Precomputed_Assistant(f"assistants/A_{self.model}.json")         
            print(f'Prover: hybrid model (symbolic + {self.model} assistance)')

    # Save the statistics
    def _save_statistics(self):
        # set the path to save the statistics file
        folder = f"stats/{self.model}"
        if not os.path.exists(folder):
            os.makedirs(folder)

        filename = os.path.join(folder, f"statistics_{self.name}.json")
        
        with open(filename, 'w') as file:
            json.dump(self.statistics, file)
            file.close()
        
        print(f'File successfully saved to "{filename}"')

    # SYMBOLIC-TEXT INTERFACE 
    # Method to split a formula
    def SF(self, f, sym=False):
        Q = f[:1]
        i = f.find(self.c)
        j = f.rfind(self.c)
        x = f[i:j]
        y = f[j:]        
        if sym:
            return f'{Q}{y}{x}'
        return (Q, x, y)
    
    # Method to encode a formula
    def _formula_encoder(self, f, V):
        '''
        Input: a symbolic formula (f) and a substitution mapping (V)
        Output: a formula in natural language
        example: Ax1x2 -> "All preac are verde" 
        '''
        pr = {'A':('all', 'are'), 'O':('some', 'are not'), 
              'I':('some', 'are'), 'E':('no', 'are')}
        q, s, p = self.SF(f)                
        
        return f"{pr[q][0]} {self.d['L']}{V[s]}{self.d['R']} {pr[q][1]} {self.d['L']}{V[p]}{self.d['R']}"

    # Method to decode a formula
    def _formula_decoder(self, f, V):
        '''
        Input: a formula in natural language (f) and a substitution mapping (V)
        Output: a symbolic formula
        example: "All preac are verde" -> Ax1x2 
        '''
        dec_dict = {f"{self.d['L']}{v}{self.d['R']}":k for k,v in V.items()}
        S = f.split()
        try:            
            # subject and predicate
            s = dec_dict[S[1]]
            p = dec_dict[S[-1]]
            # quantifier
            if S[0] == 'all' and S[2] == 'are' and len(S) == 4:
                q = 'A'
            elif S[0] == 'no' and S[2] == 'are' and len(S) == 4:
                q = 'E'
            elif S[0] == 'some' and S[2] == 'are' and len(S) == 4:
                q = 'I'
            elif S[0] == 'some' and S[2] == 'are' and S[3] == 'not' and len(S) == 5:
                q = 'O'
            
            return f'{q}{s}{p}'
        except:
            return 'unknown'    

    # METHOD TO PROVE ALL HYPOTHESES
    def build_all_inferences(self, save_statistics=False, write_proofs=False):        
        DS = self.datasets.keys()

        self.statistics = {ds:{} for ds in DS}
        for ds in DS:
            print(f'Dataset: {ds}')        
            Gamma = self.datasets[ds]['KB']
            Vocabulary = self.datasets[ds]['substitutions']
            Valid_Hypotheses = self.datasets[ds]['inferences']            
            self.statistics[ds] = {t:{} for t in Valid_Hypotheses.keys()}    
            for t, Hyp in Valid_Hypotheses.items():                                
                print(f'Type {t}:')                
                for H in Hyp:
                    print(f' Hypothesis {H}')
                    self.build_inference(Gamma, H, Vocabulary, [ds, t], write_proofs)
        if save_statistics:
            self._save_statistics()             

    # Prove a hypothesis
    def build_inference(self, Gamma, H, Vocabulary, stats_info=[], write_proofs=False):
        valid_pbc_formulas = []
        # predict formulas (connectionist models)
        if self.assistants:                
            KB = [self._formula_encoder(f, Vocabulary) for f in Gamma]            
            nm_input = 'knowledge base: ' + ', '.join(KB) + ' hypothesis: ' + self._formula_encoder(H, Vocabulary)
                        
            # premise selection assistant 
            ps_prediction = self.assistants.predict("ps", nm_input)[0].split(',')            
            needed_premises = [self._formula_decoder(pred, Vocabulary) for pred in ps_prediction]
            
            # proof by contradiction assistant
            pbc_prediction = self.assistants.predict("pbc", nm_input)[0].split(',')
            pbc_formulas = [self._formula_decoder(pred, Vocabulary) for pred in pbc_prediction]                
            valid_pbc_formulas = [f for f in pbc_formulas if f != 'unknown']
        
        # prover (symbolic model)            
        # for statistics
        total_computations = 0 
        valid = False
        ps_assistance = None
        pbc_assistance = None

        if self.assistants:
            # predicted premises that belong to Gamma
            needed_premises_in_Gamma = [p for p in needed_premises if p in Gamma]
            # first attempt
            inf = Inference(needed_premises_in_Gamma, H)
            valid = inf.prove(valid_pbc_formulas)
            if valid:
                ps_assistance = True
            total_computations += inf.attempts                

        if not valid:
            if self.assistants:
                print(f' ERROR (PS): Derivation failed from {needed_premises}')
                ps_assistance = False
            inf = Inference(Gamma, H)       
            valid = inf.prove(valid_pbc_formulas)
            total_computations += inf.attempts

        print(f' {H} is {valid}')        
                
        if valid:
            print(f' Derivation succeeded from {inf.needed_premises}')
            if inf.pbc_formula:            
                pbc_attempts = ''
                if self.assistants:
                    if inf.pbc_formula not in pbc_formulas:
                        print(f' ERROR (PBC): Derivation failed with {pbc_formulas}')
                        pbc_attempts = f' (found after {inf.pbc_formula_search} attempts)'
                        pbc_assistance = False                                        
                    else:                
                        pbc_assistance = True
                else:
                    pbc_attempts = f' (found after {inf.pbc_formula_search} attempts)'
                
                print(f' Proof by contradiction formula: {inf.pbc_formula}' + pbc_attempts)                                        

            # write proof:            
            if write_proofs:
                folder = f'proofs/{self.name}_tex/{stats_info[0]}/{stats_info[1]}'
                latex = LaTeX_Inference(inf.inference, H)
                latex.write(folder=folder)

        print(f' Derivation function was used {total_computations} times\n')

        # statistics = [<formula validity>, <# of calls to the derivation function>, <predicted formulas>, 
        #               <formula to prove by contradiction>, <number of times searching for the formula>]
        if stats_info:
            statistics = [valid, total_computations, ps_assistance, pbc_assistance, inf.pbc_formula_search]            
            ds, t = stats_info
            self.statistics[ds][t][H] = statistics
