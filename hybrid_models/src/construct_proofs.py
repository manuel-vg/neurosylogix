import sys
sys.setrecursionlimit(10**6)
from hybrid_models import Hybrid_Model

def proof_construction_experiment(models, runs):
    datasets = 'dataset/syllogisms.json'
    for model in models:
        for run in range(1, runs+1): 
            print(f"\nRun: {run}/{runs}")
            hm = Hybrid_Model(name=f"{model}_{run}", model=model, datasets=datasets)
            hm.build_all_inferences(save_statistics=True, write_proofs=True)

if __name__ == "__main__":
    models = ['Symbolic', 'GPT_OVE', 'T5_OVE', 'GPT_COM', 'T5_COM', 'GPT_REC', 'T5_REC']
    proof_construction_experiment(models, runs=2)