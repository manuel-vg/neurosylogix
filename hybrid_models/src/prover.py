from itertools import permutations
from random import shuffle
import os

class Syllogistic_Proof:
    # Class Attributes
    constant = 'x' # term constant symbol
    q = ['A', 'O', 'I', 'E'] # set of quantifier symbols

    def __init__(self, Gamma):
        self.C = self._get_constants(Gamma)    

    # Method to split a formula
    def SF(self, f):
        Q = f[:1]
        i = f.find(self.constant)
        j = f.rfind(self.constant)
        x = f[i:j]
        y = f[j:]        
        return (Q, x, y)

    # Method to get the negation of a formula
    def neg(self, f):
        Q,x,y = self.SF(f)
        i = self.q.index(Q) 
        if i % 2 == 0:        
            nQ = self.q[i+1]
        else:
            nQ = self.q[i-1]
        return f'{nQ}{x}{y}'

    # Method to get all constant symbols from 'Gamma' in a list
    def _get_constants(self, Gamma):
        C = []
        for g in Gamma:
            C += [self.SF(g)[1], self.SF(g)[2]]
        # drop duplicates and shuffle
        C = list(set(C)) 
        shuffle(C)        
        return C

class Inference(Syllogistic_Proof):
    def __init__(self, Gamma, hypothesis):
        super().__init__(Gamma)            
        self.Gamma = Gamma
        self.H = hypothesis        
        self.inference = {}
        self.attempts = 0 # number of times the prover visits the "find_derivation" method
        self.pbc_formula = None # formula to prove by contradiction        
        self.pbc_formula_search = 0 # number of attempts for "proof by contradiction"
        self.needed_premises = []
    
    # Recursive method to get needed premises
    def _get_premises(self, der, H):
        if der[H][0] == 'premise':
            self.needed_premises.append(H)
            return
        if len(der[H]) == 2:            
            self._get_premises(der, der[H][0])
        else:            
            self._get_premises(der, der[H][0])
            self._get_premises(der, der[H][1])

    # Recursive method to display the derivation
    def _show_proof(self, der, H, n=1):
        if der[H][0] == 'premise':
            print(f'{n}\t{H} {der[H][1]}')
            return 
        if len(der[H]) == 2:
            print(f'{n}\t{H} derived from {der[H][0]} by {der[H][1]}')
            n += 1
            self._show_proof(der, der[H][0], n)
        else:
            print(f'{n}\t{H} derived from {der[H][0]} and {der[H][1]} by {der[H][2]}')
            n += 1
            self._show_proof(der, der[H][0], n)
            self._show_proof(der, der[H][1], n)

    # Method to list of all possible contradictions
    def _contradictions(self):
        # generate all possible formulas from the set of term symbols
        F = [f'{Q}{a}{b}' for a,b in permutations(self.C, 2) for Q in self.q]
        shuffle(F)

        return [(self.neg(P), P) for P in F]

    # Methods to prove 'H' from 'Gamma'
    # 1) proof by rules: type (i) and type (ii) 
    def proof_by_rules(self):
        der = Derivation(self.Gamma, self.H)
        if der.derive():    
            self.attempts = der.counter
            self.inference = der.inference
            return True            
        else:
            self.attempts = der.counter
        return False

    # 2) proof by contradiction: type (iii)
    def proof_by_contradiction(self, nm_aid):
        Q,_,_ = self.SF(self.H)
        if Q in ['O', 'I']:            
            assistance = [(self.neg(P), P) for P in nm_aid]
            for n, (P,nP) in enumerate(assistance + [p for p in self._contradictions() if p not in assistance], 1):
                # right derivation                                    
                right = Derivation(self.Gamma, nP)
                # check if the right part of the derivation is TRUE
                if right.derive():
                    # left derivation                
                    left = Derivation(self.Gamma + [self.neg(self.H)], P)                
                    # check if the left part of the derivation is TRUE
                    if left.derive(): 
                        # update inference tree                                            
                        self.inference = {**left.inference, **right.inference}    
                        self.inference[self.H] = [P, nP, '(iii)']      
                        # update statistics                    
                        self.attempts += (1 + left.counter + right.counter)
                        self.pbc_formula = nP
                        self.pbc_formula_search = n              
                        return True              
                    else:
                        self.attempts += (1 + left.counter + right.counter)
                else:                    
                    self.attempts += (1 + right.counter)
        
        return False

    # Method to display the inference rules used to derive H
    def show_derivations(self):
        self._show_proof(self.inference, self.H)    

    # Method to check whether the premises used in the inference are correct
    def get_needed_premises(self):
        return self._get_premises(self.inference, self.H)    

    # MAIN METHOD
    def prove(self, nm_aid=[]):
        if self.proof_by_rules() or self.proof_by_contradiction(nm_aid):
            self.get_needed_premises()
            return True
        else:
            return False #'Invalid hypothesis'    

class Derivation(Syllogistic_Proof):
    def __init__(self, Gamma, hypothesis):
        super().__init__(Gamma)
        self.Gamma = Gamma    # Knowledge Base
        self.H = hypothesis   # Hypothesis
        self.derivations = {} # All possible derivations
        self.inference = {}   # Final derivation
        self.counter = 0      # How many times did a method is called?
        self.loop = []        # List of derived 'E' formulas

    def _get_final_derivation(self, H, der={}):                
        if self.derivations[H][0][-1] in ['(r2)', '(r3)']:
            self.loop += [H]
        for d in self.derivations[H]:
            der[H] = d
            if d[-1] == '(i)': # premise
                return der
            if d[-1] in ['(r3)', '(r4)']: # unary inference
                if d[0] not in self.loop:
                    return self._get_final_derivation(d[0], der) 
                else:
                    der.clear()
            if d[-1] in ['(r1)', '(r2)']: # binary inference
                if d[0] not in self.loop and d[1] not in self.loop:
                    return {**self._get_final_derivation(d[0], der), **self._get_final_derivation(d[1], der)}                    
                else:
                    der.clear()    

    def _apply_rule(self, Q, hQ, H, rule):
        if Q == hQ:            
            if (H,rule) not in self.visited or H in self.derivations.keys():
                if H not in self.derivations.keys():                    
                    self.visited.append((H,rule))
                return True
        return False

    def _add_derivation(self, der):
        if der[0] not in self.derivations.keys():
            self.derivations[der[0]] = [der[1:]]
        elif der[1:] not in self.derivations[der[0]]:
            self.derivations[der[0]].append([der[1:]])

    def _binary_inf(self, H, hQ, lQ, rQ, rule):
        '''
        H = Hypothesis to be derived (potential conclusion)
        hQ = Hypothesis Quantifier: 'A' or 'E'
        lQ = left formula Quantifier: 'A'
        rQ = right formula Quantifier: 'A' or 'E'
        rule = name of the rule: '(r1)' or '(r2)'
        '''
        Q,a,c = self.SF(H)
        if self._apply_rule(Q, hQ, H, rule):
            for b in [b for b in self.C if b not in [a,c]]:
                # build formulas
                f1 = f'{lQ}{a}{b}'
                f2 = f'{rQ}{b}{c}'
                # left derivation                
                if f1 not in self.derivations.keys():
                    left = self._find_derivation(f1)
                else:
                    left = True
                # check if the left part of the derivation is TRUE
                if left: 
                    # right derivation                                    
                    if f2 not in self.derivations.keys():
                        right = self._find_derivation(f2)
                    else:
                        right = True                                        
                    # check if the right part of the derivation is TRUE
                    if right:
                        self._add_derivation([H, f1, f2, rule])                    
                        return True     
        return False

    def _unary_inf(self, H, hQ, fQ, rule):
        '''
        H = Hypothesis to be derived (potential conclusion)
        hQ = Hypothesis Quantifier: 'E' or 'I'
        fQ = formula Quantifier: 'E' or 'A'
        rule = name of the rule: '(r3)' or '(r4)'
        '''
        Q,a,b = self.SF(H)
        if self._apply_rule(Q, hQ, H, rule):
            # build formula
            f = f'{fQ}{b}{a}'                
            # derivation
            if f not in self.derivations.keys():
                der = self._find_derivation(f)
            else:
                der = True
            # check if the derivation is TRUE
            if der:
                self._add_derivation([H, f, rule])
                return True         
        return False

    def _find_derivation(self, H):
        self.counter += 1
        # type i:
        if H in self.Gamma:
            self._add_derivation([H, 'premise', '(i)'])
            return True
        # type ii (rule 1)
        if self._binary_inf(H, 'A', 'A', 'A', '(r1)'):
            return True
        # type ii (rule 2)
        if self._binary_inf(H, 'E', 'A', 'E', '(r2)'):
            return True
        # type ii (rule 3)
        if self._unary_inf(H, 'E', 'E','(r3)'):
            return True
        # type ii (rule 4)
        if self._unary_inf(H, 'I', 'A', '(r4)'):
            return True
        
        return False

    def derive(self):
        self.visited = []       
        if self._find_derivation(self.H):
            self.inference = self._get_final_derivation(self.H)
            return True
        else:
            return False            

class LaTeX_Inference:
    def __init__(self, inference, hypothesis):
        self.inference = inference
        self.H = hypothesis
        self.s = []
        self.l = 0
        
    # Methods to write a '.tex' file
    def _add_lines(self, dtype, rule, formula):
        label = '\\RightLabel{\\scriptsize' + rule +'}'
        if dtype == 'a':
            # lines = [b + 'AxiomC{}', label, b + 'UnaryInfC{' + formula + '}']
            lines = ['\\AxiomC{}', label, '\\UnaryInfC{' + formula + '}']
        elif dtype == 'u':
            lines = [label, '\\UnaryInfC{' + formula + '}']
        else:
            lines = [label, '\\BinaryInfC{' + formula + '}']

        return lines

    def _write_proof(self, der, H):
        if der[H][0] == 'premise':
            self.s += self._add_lines('a', der[H][1], H)
            self.l += 1
            return
        if len(der[H]) == 2:
            self._write_proof(der, der[H][0])
            self.s += self._add_lines('u', der[H][1], H)
        else:
            self._write_proof(der, der[H][0])
            self._write_proof(der, der[H][1])
            self.s += self._add_lines('b', der[H][2], H)

    # Method to write the inference as a LaTeX file
    def write(self, folder=''):        
        filename = f'{self.H}.tex'

        if folder:
            # check if folder exists    
            if not os.path.exists(folder):
                os.makedirs(f'{folder}/')        
            # create a path filename
            filename = f'{folder}/{filename}'

        self._write_proof(self.inference, self.H)
        with open(filename, 'w') as f:
            # write head of the file
            f.write('\\documentclass{article}\n')
            f.write('\\usepackage[paperwidth=' + str(self.l + 2) + 'in]{geometry}\n')
            f.write('\\usepackage{bussproofs}\n')
            f.write('\\usepackage{graphicx}\n')
            f.write('\\begin{document}\n')
            f.write('\\begin{prooftree}\n')
            
            # write actual inference
            for line in self.s:
                f.write(f'{line}\n')

            # write the end
            f.write('\\end{prooftree}\n')
            f.write('\\end{document}')
        f.close()
