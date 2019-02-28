import warnings

import empiricalutilities as eu
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

from project_stats import TrainProjections, impute

plt.style.use('fivethirtyeight')

# %%

def main():
    n_sims = 50
    essential_stats = False
    impute_methods = [
        # 'BiScaler',
        'IterativeImpute',
        # 'IterativeSVD',
        'KNN',
        # 'MatrixFactorization',
        'Mean',
        'Median',
        # 'NuclearNorm',
        'SoftImpute',
        ]
    
    if essential_stats:
        impute_stats = {
            'QB': ['Pass Yds', 'Pass TD', 'Pass Int', 'Rush Yds', 'Rush TD'],
            'RB': ['Rush Yds', 'Rush TD', 'Receptions', 'Rec Yds', 'Rec TD'],
            'WR': ['Rush Yds', 'Rush TD', 'Receptions', 'Rec Yds', 'Rec TD'],
            'TE': ['Receptions', 'Rec Yds', 'Rec TD'],
            'DST': ['PA', 'YdA', 'TD', 'Sack', 'Int', 'Fum Rec'],
            }
        fid_MAE = '../figures/impute_MAE'
        fid_RMSE = '../figures/impute_RMSE'
    else:
        impute_stats = {
            'QB': ['Receptions', 'Rec Yds', 'Rec TD', '2PT'],
            'RB': ['Pass Yds', 'Pass TD', 'Pass Int', '2PT'],
            'WR': ['Pass Yds', 'Pass TD', 'Pass Int', '2PT'],
            'TE': ['Pass Yds', 'Pass TD', 'Pass Int', 'Rush Yds',
                   'Rush TD', '2PT'],
            'DST': ['Saf', 'Blk'],
            }
        fid_MAE = '../figures/nonessential_impute_MAE'
        fid_RMSE = '../figures/nonessential_impute_RMSE'
        
    
    j = 0
    x_ticks = []
    n_stats = sum(len(stats) for stats in impute_stats.values())
    mae = np.zeros([len(impute_methods), n_stats])
    rmse = np.zeros([len(impute_methods), n_stats])
    with tqdm(total=n_stats) as pbar:
        for pos in impute_stats.keys():
            proj = TrainProjections(pos)
            proj.load_data(weeks=range(1,18), impute_method=False)
            for stat in impute_stats[pos]:
                df = proj._stats_df[stat]
                
                # Subset DataFrame to only include only projection columns.
                ignored_cols = ['Player', 'Team', 'Pos', 'Week', 'STATS']
                impute_cols = [c for c in list(df) if c not in ignored_cols]
                proj_df = df[impute_cols].copy()
                
                # Find percentage of NaNs in for the stat.
                n, m = proj_df.shape
                nan_pct = np.sum(np.sum(proj_df.isnull())) / (n * m)
                
                # Remove rows with NaNs.
                proj_df.dropna(how='any', axis=0, inplace=True)
                
                print(f'\n{pos} - {stat}\n')
                # Get average RMSE and MAE for each imputation method.
                for i, method in enumerate(impute_methods):
                    try:
                        rmse[i, j], mae[i, j] = simulate_imputing(
                            proj_df.copy(), method, n_sims, nan_pct)
                    except ValueError:
                        pass
                x_ticks.append(f'{pos} - {stat}')
                j += 1
                pbar.update(1)
            

    
    # %%
    mae_df = pd.DataFrame(mae, columns=x_ticks, index=impute_methods)
    mae_df = (mae_df - mae_df.mean()) / mae_df.std()
    plot_heatmap(mae_df, cbar_label='Normalized MAE')
    eu.save_fig(fid_MAE)
    plt.show()
    
    # %%
    rmse_df = pd.DataFrame(rmse, columns=x_ticks, index=impute_methods)
    rmse_df = (rmse_df - rmse_df.mean()) / rmse_df.std()
    plot_heatmap(rmse_df, cbar_label='Normalized RMSE')
    eu.save_fig(fid_RMSE)
    plt.show()
    # %%



    
    
def plot_heatmap(df, cbar_label):
    """Plot heatmap for RMSE or MAE."""
    warnings.filterwarnings('ignore', category=DeprecationWarning)
    
    fig, ax = plt.subplots(1, 1, figsize=[12, 6])
    cbar_kwargs = {'shrink': 0.82, 'label': cbar_label}
    sns.heatmap(df, ax=ax, cmap='RdYlGn_r', cbar_kws=cbar_kwargs)
    ax.set_xticklabels(df.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticklabels(df.index, rotation='horizontal', fontsize=8)
    
    
def simulate_imputing(df, method, sims, nan_pct):
    """
    Simulate imputing method on full Data, returning RMSE and MAE.
    
    Parameters
    ----------
    method: str/bool
        Imputation method for missing data.
            - False: Do not impute missing data.
            - 'BiScalar'
            - 'IterativeImpute'
            - 'KNN': Impute with nearest neighbors.
            - 'Mean': Impute missing with average of other sources.
            - 'NuclearNorm'
            - 'SoftImpute'
    sims: int
        Number of simulations to run.
    nan_pct: float
        Percentage of NaNs to add to DataFrame.
    
    Returns
    -------
    rmse: float
        Average RMSE of simulations.
    mae: float
        Average MAE of simulations.
    """
    
    rmse = np.zeros(sims)
    mae = np.zeros(sims)
    for i in range(sims):
        # Create duplicate DataFrame with randomly removed values at the
        # the same NaN percentage as the original data.
        missing_mask = np.random.rand(*df.shape) < nan_pct
        df_incomplete = df.copy()
        df_incomplete[missing_mask] = np.nan
        
        # Impute missing data.
        df_imputed = impute(df_incomplete.copy(), method=method)
        
        # Find RMSE and MAE.
        
        e = (df.values - df_imputed.values).flatten()
        
        rmse[i] = np.mean(e**2)**0.5
        mae[i] = np.mean(np.abs(e))
        
    return np.mean(rmse), np.mean(mae)
    
if __name__ == '__main__':
    main()
