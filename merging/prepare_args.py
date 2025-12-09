import argparse

def create_parser():
    _parser = argparse.ArgumentParser(description='Configuration for MergeBench')

    # DDP
    _parser.add_argument("--local_rank", type=int, default=0)

    # Task arithmetic
    _parser.add_argument('--scaling-coef', default=1, type=float)

    # TIES
    _parser.add_argument('--K', default=0.2, type=float)
    _parser.add_argument('--merge_func', default="sum", type=str)

    # DARE
    _parser.add_argument('--p', default=0.8, type=float)

    # LocalizeAndStitch
    _parser.add_argument('--sigmoid_bias', default=3, type=float)
    _parser.add_argument('--sparsity', default=0.1, type=float)
    _parser.add_argument('--lr', default=1e8, type=float)
    _parser.add_argument('--n_epochs', default=1, type=int)
    _parser.add_argument('--l1_strength', default=0.0, type=float)
    _parser.add_argument('--dataless', action='store_true')

    # Consensus
    _parser.add_argument('--k', default=2, type=int)
    _parser.add_argument('--lamda', default=0.5, type=float)

    # RegMean and RegMeanPlusPlus
    _parser.add_argument('--task_names', type=str)
    _parser.add_argument('--reduction', type=float)

    # Fisher
    _parser.add_argument("--fisher_only", action="store_true", help='fisher-stage1')
    _parser.add_argument("--merge_only", action="store_true", help='fisher-stage2')
    _parser.add_argument("--save_group", type=str, default=None)
    _parser.add_argument("--model_coeff_value", type=float, default=0.3)
    _parser.add_argument("--keep_checkpoints", action="store_true", help='whether delete intermediate files')
    
    
    return _parser

def prepare_args(params):
    kwargs = {}
    if params.algo == 'TaskArithmetic':
        kwargs['scaling_coef'] = params.scaling_coef
    elif params.algo == 'TIES':
        kwargs['scaling_coef'] = params.scaling_coef
        kwargs['merge_func'] = params.merge_func
        kwargs['K'] = params.K
    elif params.algo == 'DARE':
        kwargs['scaling_coef'] = params.scaling_coef
        kwargs['p'] = params.p
    elif params.algo == 'LocalizeAndStitch':
        kwargs['sparsity'] = params.sparsity
        kwargs['dataless'] = params.dataless
        kwargs['sigmoid_bias'] = params.sigmoid_bias
        if not params.dataless:
            kwargs['learning_rate'] = params.lr
            kwargs['num_train_epochs'] = params.n_epochs
            kwargs['l1_strength'] = params.l1_strength
    elif params.algo == 'Consensus':
        kwargs['scaling_coef'] = params.scaling_coef
        kwargs['k'] = params.k
        kwargs['lamda'] = params.lamda
    elif params.algo == 'RegMean' or params.algo == 'RegMeanPlusPlus':
        kwargs['task_names'] = params.task_names
        kwargs['reduction'] = params.reduction
    elif params.algo == 'Fisher':
        kwargs['task_names'] = params.task_names
        kwargs['merge_only'] = params.merge_only
        kwargs['fisher_only'] = params.fisher_only
        kwargs['save_group'] = params.save_group
        kwargs['model_coeff_value'] = params.model_coeff_value
        kwargs['keep_checkpoints'] = params.keep_checkpoints
    else:
        raise ValueError('No support merging method {}'.format(params.algo)) 

    return kwargs