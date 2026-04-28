import os
import re
import sys
import json
from argparse import ArgumentParser

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils import resolve_datasets, get_evaluator_class


def get_all_results(save_dir, ds_mappings, centre_ds):
    res = {}
    for dataset, param_config in ds_mappings.items():
        params_str = f'temp_{param_config["temperature"]}_n_{param_config["n_samples"]}'
        metric_path = os.path.join(save_dir, params_str, dataset, 'metrics.json')
        if not os.path.isfile(metric_path):
            raise FileNotFoundError(f"{metric_path} does not exist")
        
        with open(metric_path, 'r') as f:
            metrics = json.load(f)
        
        for ds, value in metrics.items():
            if ds in centre_ds:
                res[ds] = value

    return res

def print_metrics(metrics, report_order):
    for dataset, values in metrics.items():
        print(f"\n# {dataset}")
        for key, value in values.items():
            print(f"    {key}: {100*value:.2f}")
    
    pass_1_list = [metrics[x]['pass@1'] for x in report_order]
    pass_1_list.append(sum(pass_1_list) / len(pass_1_list))

    print('\n# Summary')
    report_order.append('Average')
    print(' & '.join(report_order))
    print(' & '.join([f"{100*x:.2f}" for x in pass_1_list]))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--datasets', nargs='+', required=True, help='List of datasets')
    parser.add_argument('--save_dir', type=str, required=True)
    
    args = parser.parse_args()
    save_dir = args.save_dir

    ds_mappings = resolve_datasets(args.datasets)

    # calculate average
    mappings = {
        'aider': ['aider_0', 'aider_1']
    }

    # Just for convenient copy-paste
    report_order = ['editeval', 'humanevalfix-python', 'canitedit', 'aider', 'humanevalfix-js']

    centre_ds = []
    for ds in report_order:
        if ds not in ds_mappings:
            continue
        if ds in mappings:
            centre_ds.extend(mappings[ds])
        else:
            centre_ds.append(ds)

    metrics = get_all_results(save_dir, ds_mappings, set(centre_ds))
    print_metrics(metrics, centre_ds)
