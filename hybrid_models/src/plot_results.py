import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_number_of_steps(models):
    all_values = {}

    for model in models:
        folder = f"stats/{model}"
        # Load the runs
        runs = []
        for s in os.scandir(folder):
            if not s.name.startswith('.'):
                with open(s.path) as f:
                    runs.append(json.load(f))

        values = []

        # KBs
        for kb in runs[0]:
            # Inference types
            for typ in runs[0][kb]:
                # Hypotheses
                for hypothesis in runs[0][kb][typ]:
                    # Second value (number of steps) across the runs
                    vec = [run[kb][typ][hypothesis][1] for run in runs]
                    # Mean of log10 across the runs
                    values.append(np.mean(np.log10(vec)))

        all_values[model] = np.array(values)

    return all_values


def plot_number_of_steps(models):
    all_values = get_number_of_steps(models)

    # compute mean and standard deviation
    d = {k: [np.mean(v), np.std(v)] for k,v in all_values.items()}

    # create a data frame
    df = pd.DataFrame.from_dict(d, orient='index', columns=['Mean', 'SD'])

    # plot
    color = ['red'] + ['darkblue', 'orange']*3
    ax = df.plot(kind='barh', y='Mean', legend=False, title='Geometric Mean and SD', xerr='SD', color=color, alpha=0.5, figsize=(6,3))
    ax.set_xlabel('Number of steps')        
    ax.set_xticks([x for x in range(1,8)], ['$10^1$', '$10^2$', '$10^3$', '$10^4$', '$10^5$', '$10^6$', '$10^7$'])
    ax.invert_yaxis() # show inverted y axis

    plt.show()

if __name__ == "__main__":
    models = ['Symbolic', 'GPT_OVE', 'T5_OVE', 'GPT_COM', 'T5_COM', 'GPT_REC', 'T5_REC']
    plot_number_of_steps(models)