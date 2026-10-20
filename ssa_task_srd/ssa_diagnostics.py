"""Fixed SSA reconstructions and component deletions; no cleaning or perturbation."""
import numpy as np


def regroup(x,full_components,rank):
    x=np.asarray(x,dtype=np.float64);c=np.asarray(full_components,dtype=np.float64)
    if x.ndim!=2 or c.ndim!=3 or c.shape[0]!=x.shape[0] or c.shape[2]!=x.shape[1]:
        raise ValueError('Expected window matrix and corresponding SSA component tensor')
    if not isinstance(rank,int) or not 1<=rank<c.shape[1]:raise ValueError('Invalid diagnostic rank')
    retained=c[:,:rank,:]
    return np.concatenate([retained,(x-retained.sum(axis=1))[:,None,:]],axis=1)


def fixed_variants(x,full_components,ranks,ablation_rank):
    x=np.asarray(x,dtype=np.float64);c=np.asarray(full_components,dtype=np.float64)
    if any(not isinstance(k,int) or not 1<=k<c.shape[1] for k in ranks):raise ValueError('Invalid retained rank')
    grouped=regroup(x,c,ablation_rank)
    variants={'raw':x.copy(),'all_components_identity':c.sum(axis=1)}
    for rank in ranks:variants[f'ssa_rank_{rank}']=c[:,:rank,:].sum(axis=1)
    for i in range(ablation_rank):variants[f'drop_C{i+1}']=x-grouped[:,i,:]
    variants[f'drop_residual_after_{ablation_rank}']=grouped[:,:-1,:].sum(axis=1)
    return variants
