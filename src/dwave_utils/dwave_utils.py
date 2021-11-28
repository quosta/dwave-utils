import dimod
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import minorminer
from dwave.embedding import embed_bqm, EmbeddedStructure
from dwave.system.samplers import DWaveSampler

# =============================================================================
# BQM functions
# =============================================================================

def save_bqm(bqm, path):
    '''
    Save BQM as dimod binary file

    Parameters
    ----------
    bqm : dimod.BQM
        A Binary Quadratic Model
    path : str
        Valid BQMfilename path

    Returns
    -------
    None.

    '''
    serializable = bqm.to_file()
    with open(path,"bw") as f:
        f.write(serializable.read())


def open_bqm(path):
    '''
    Load BQM from previously stored file

    Parameters
    ----------
    path : str
        Valid BQM filename path
        
    Returns
    -------
    bqm : dimod.BQM
        A Binary Quadratic Model

    '''
    with open(path,"br") as f:
        bqm = dimod.BQM.from_file(f)
    return bqm    


def random_dense_bqm(n):
    '''
    Create a random dense BQM. The weights are normalized between -1 and 1, 
    and the distribution of the weights is uniform.

    Parameters
    ----------
    n : int
        size of the BQM

    Returns
    -------
    dimod.BQM
        Output BQM

    '''
    A = np.random.rand(n,n)
    A = np.triu(A, k=0)
    times = int(n*(n+1)//4)
    i = 0
    seen = []
    while i < times:
        x, y = np.random.choice(A.shape[0], 2, replace=True)
        if x <= y and (x,y) not in seen:
            A[x,y] = -A[x,y]
            i += 1
            seen.append((x,y))
    return dimod.BQM(A,"BINARY")    


def random_sparse_bqm(n,connectivity=0.4):
    '''
    Create a sparse BQM, with sparsity given as parameter. The rest of the 
    weights is normalized between -1 and 1, and the distribution of the weights 
    is uniform.     

    Parameters
    ----------
    n : int
        size of the BQM
    connectivity : float, optional
        Sets the sparsity of the matrix. The default is 0.4.

    Returns
    -------
    dimod.BQM
        Output BQM
    '''
    A = random_dense_bqm(n)
    A = A.to_numpy_matrix()
    times = int(n*(n-1)//2*(1-connectivity))
    i = 0
    while i < times:
        x, y = np.random.choice(A.shape[0], 2, replace=False)
        if x < y and A[x,y] != 0:
            A[x,y] = 0
            i += 1
    return dimod.BQM(A,"BINARY")    


def bruteforce_bqm(bqm: dimod.BQM, 
                   df=True, 
                   verbose = True):
    '''
    Solve the BQM by direct multiplication of the matrix with all the possible
    binary strings
    
    Parameters
    ----------
    bqm : dimod.BQM
        A Binary Quadratic Model
    df : TYPE, optional
        Return the DataFrame with bitstr and solutions. The default is True.
    verbose : TYPE, optional
        Print the lowest energy state. The default is True.

    Returns
    -------
    Pandas DataFrame or tuple
        Depending on the df flag, it returns either the complete DataFrame or
        just the minimal energy state as (bitstr, energy)

    '''
    n = len(bqm.linear)
    max_n = 2**n
    sol_list = np.matrix([list(np.binary_repr(i,n)) for i in range(max_n)]).astype(int)

    sol_df = pd.DataFrame(columns=['bitstr','energy'])
    mat = bqm.to_numpy_matrix()
    E_min = 10**3
    for s in sol_list:
        energy = s @ mat @ s.T
        if df is True:
            sol_df = sol_df.append({
                    'bitstr':s.tolist()[0], 
                    'energy':energy.item()
                    },ignore_index=True)
            if verbose:
                print("Lowest energies: ", sol_df[sol_df.energy == sol_df.energy.min()])
            return sol_df
        else:
            if energy.item() < E_min:
                E_min = energy.item() 
                sol_min = s.tolist()[0]
    return sol_min, E_min
        

def bqm_info(bqm: dimod.BQM,
             verbose=False):
    '''
    Returns number of linear and quadratic variables and connectivity
    The connectivity is defined as a float between 0 and 1, and is equal
    to 0 if there are no quadratic weights, equal to 1 if there is every
    possible quadratic weight. The linear weights do not affect the 
    connectivity.

    Parameters
    ----------
    bqm : dimod.BQM
        A Binary Quadratic Model
    verbose : boolean
        Print the result of the analysis. Default is False.

    Returns
    -------
    Python dict:
        num_lin : int
            Number of linear variables
        num_quad : int
            Number of quadratic variables
        connectivity : float
            Connectivity measure
    '''
    num_lin = len(bqm.linear)
    num_quad = len(bqm.quadratic)
    
    connectivity = num_quad / (num_lin*(num_lin-1)//2) 
    if verbose:
        print('num linear weights = ', num_lin)
        print('num quadratic weights = ', num_quad)
        print('connectivity = ', connectivity)
    
    ret = {'num_lin': num_lin,
           'num_quad': num_quad,
           'connectivity': connectivity}
    return ret


def bqm_heatplot(bqm: dimod.BQM, 
                 figsize=None,
                 scale_font=1,
                 n_variables=None,
                 sort_key=None,
                 annot=False,
                 vmin=None,
                 vmax=None, 
                 cmap='coolwarm', 
                 grid=True,
                 grid_lines=True,
                 all_ticks=True,
                 axis_lbl='',
                 cbar_lbl='',
                 title_lbl='',
                 show=True,
                 save_path=None):
    '''    
    Plots the Binary Quadratic Model as a colored heatmap

    Parameters
    ----------
    bqm : dimod.BQM
        A Binary Quadratic Model
    figsize : tuple, optional
        Specify the dimension of the plot as a tuple. If not provided it will
        be inferred.
    scale_font : float, optional
        Scale the dimension of the font. The default is 1.
    n_variables : int, optional
        Reduce the number of variables to be shown. The default is None.
    sort_key : function, optional
        Pass a key function to sort the name of the variables. 
        The default is None.
    annot: bool, optional
        Print value inside the square. The default is False.
    vmin : float, optional
        Anchor the colorbar to a minimum value. If not provided the lowest 
        value in the BQM will be selected. 
    vmax : float, optional
        Anchor the colorbar to a maximum value. If not provided the lowest 
        value in the BQM will be selected.
    cmap : str, optional
        A colormap supported by seaborn. The default is 'coolwarm'.
    grid : bool, optional
        The heatmap is displayed on a background grid. The default is True.
    grid_lines : bool, optional
        Show the grid lines in grey if the grid param is True. 
        The default is True.
    all_ticks : bool, optional
        Print the labels of all axis. If set to False try to densely plot 
        non-overlapping labels. The default is True.
    axis_lbl: str, optional
        Print a label on the x and y axis. The default is blank.
    cbar_lbl: str, optional
        Print a label next to the color bar. The default is blank.
    title_lbl: str, optional
        Print a title over the chart. The default is blank.
    show : bool, optional
        Show the heatmap to screen. The default is True.
    save_path : file, optional
        Save the heatmap to a file. The default is None.

    Returns
    -------
    None.

    '''    
    
    bqm_df = bqm_getdf(bqm, n_variables)
    
    if vmin is None:
        vmin = bqm_df.h.min()

    if vmax is None:
        vmax = bqm_df.h.max()

    bqm_df = bqm_df.pivot('x', 'y', 'h')
    
    if sort_key is not None:
        bqm_df = bqm_df.sort_index(axis=0, key=lambda a: a.map(sort_key))
        bqm_df = bqm_df.sort_index(axis=1, key=lambda a: a.map(sort_key)) 
 
    if figsize is None:
        figsize = 8 + 0.15*len(bqm_df)
        fig, ax = plt.subplots(figsize = (figsize,figsize))
    else:
        fig, ax = plt.subplots(figsize = figsize)
    
    linewidths = 1 if grid else 0
    linecolor = 'grey' if grid_lines else 'white'
        
    scale_font 
    sns.set(font_scale=scale_font)
    sns.heatmap(bqm_df, 
                annot=annot,
                linewidths=linewidths, 
                linecolor=linecolor,
                xticklabels=all_ticks, 
                yticklabels=all_ticks,
                vmin=vmin, 
                vmax=vmax, 
                cmap=cmap,
                ax=ax,
                cbar_kws={'label': cbar_lbl})
    ax.set_xlabel(axis_lbl)
    ax.set_ylabel(axis_lbl)
    ax.set_title(title_lbl)

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()


def bqm_getdf(bqm: dimod.BQM, 
               n_variables=None, 
               linear=True, 
               quadratic=True):
    '''
    Transform a Binary Quadratic Model to a pandas DataFrame

    Parameters
    ----------
    bqm : dimod.BQM
        A Binary Quadratic Model
    n_variables : int, optional
        Reduce the number of variables. The default is None.
    linear : bool, optional
        Include the linear factors (bias). The default is True.
    quadratic : bool, optional
        Include the quadratic factors (couplers). The default is True.

    Returns
    -------
    bqm_df : pandas.DataFrame

    '''
    
    assert linear or quadratic, "At least one of 'linear' or 'quadratic' must be True."

    bqm_linear = bqm.linear
    bqm_quadratic = bqm.quadratic

    if quadratic and len(bqm_quadratic)>0:
        bqm_dict = {
            'x': [k[0] for k in bqm_quadratic],
            'y': [k[1] for k in bqm_quadratic],
            'h': [bqm_quadratic[k] for k in bqm_quadratic]
        }
    else:
        bqm_dict = {'x': [], 'y': [], 'h': []}

    if linear and len(bqm_linear)>0:
        bqm_dict['x'].extend([k for k in bqm_linear])
        bqm_dict['y'].extend([k for k in bqm_linear])
        bqm_dict['h'].extend([bqm_linear[k] for k in bqm_linear])

    bqm_df = pd.DataFrame(bqm_dict)

    if n_variables is not None:
        bqm_df = bqm_df[bqm_df['x'] < n_variables]
        bqm_df = bqm_df[bqm_df['y'] < n_variables]

    return bqm_df


def bqm_distplot(bqm: dimod.BQM, 
                 linear=True, 
                 quadratic=True, 
                 quantile=None, 
                 separate_weights=False,
                 figsize=(8.48,4.8), 
                 scale_font=0.8,
                 xaxis_lbl='Weight values',
                 yaxis_lbl='Occurrences',
                 title_lbl='',
                 show=True, 
                 save_path=None):
    '''
    Plot a distribution of the weights of a Binary Quadratic Model.    

    Parameters
    ----------
    bqm : dimod.BQM
        A Binary Quadratic Model
    linear : bool, optional
        Include the linear factors (bias). The default is True.
    quadratic : bool, optional
        Include the quadratic factors (variance). The default is True.
    separate_weights : bool, optional
        Plot different distributions for linear (biases) and quadratic
        (variance) terms. The default is False.
    figsize : tuple, optional
        Specify the dimension of the plot as a tuple.
    scale_font : float, optional
        Scale the dimension of the font. The default is 0.8.
    xaxis_lbl: str, optional
        Print a label under the x-axis. The default is 'Weight values'.
    yaxis_lbl: str, optional
        Print a label next to the y-axis. The default is 'Occurrences'.
    title_lbl: str, optional
        Print a title over the chart. The default is blank.
    show : bool, optional
        Show the distplot to screen. The default is True.
    save_path : file, optional
        Save the distplot to a file. The default is None.

    Returns
    -------
    None.

    '''
    
    fig, ax = plt.subplots(figsize = figsize)
    
    if separate_weights:
        linear = True
        quadratic = True

    bqm_df = bqm_getdf(bqm, linear=linear, quadratic=quadratic)

    if separate_weights:
        bqm_df['Weight type'] = 'Quadratic'
        bqm_df['Weight type'] = bqm_df['Weight type'].where(bqm_df['x'] != bqm_df['y'], 'Linear')
        hue="Weight type"
    else:
        hue = None
        
    sns.set(font_scale=scale_font)
    s = sns.displot(data=bqm_df,x='h',hue=hue)

    s.axes[0][0].set_xlabel(xaxis_lbl)
    s.axes[0][0].set_ylabel(yaxis_lbl)
    s.axes[0][0].set_title(title_lbl)
        
    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()
        
        
def embedded_bqm(bqm, 
                 target='advantage',
                 embedding=False):
    '''
    Obtain a new BQM that is fit for the provided topology. Optionally return
    the embedding as dict.
    
    Parameters
    ----------
    bqm : dimod.BQM
        A Binary Quadratic Model
    target : str, optional
        The target topology, either 'advantage' or '2000'. The default is 'advantage'.
    embedding : Boolean, optional
        If True also return embedding dict. The default is True.

    Returns
    -------
    dimod.BQM
        Embedded BQM
    '''
    # if target == 'pegasus':
    #     topology = dnx.pegasus.pegasus_graph(16)
    # if target == 'chimera':
    #     topology = dnx.chimera.chimera_graph(8)
    if target == '2000':
        sampler = DWaveSampler(solver='DW_2000Q_6')
        topology = sampler.to_networkx_graph()
    if target == 'advantage':
        sampler = DWaveSampler(solver='Advantage_system4.1')
        topology = sampler.to_networkx_graph()
        
    emb = minorminer.find_embedding(bqm.to_networkx_graph(), topology)
    embedding = EmbeddedStructure(topology.edges(), emb)

    new_bqm = embed_bqm(bqm,embedding)
        
    if embedding:
        return new_bqm, emb
    else:
        return new_bqm
    
    
def analyze_embedding(embedding, verbose=False):
    '''
    Provide info regarding the embedding

    Parameters
    ----------
    embedding : dict
        Embedding dict
    verbose : Boolean, optional
        Print the chain properties. The default is False.

    Returns
    -------
    chain_df : DataFrame
        DataFrame containing the chains lenghts ant their occurrences

    '''
    chain_lengths = [len(value) for value in embedding.values()]
    min_chain = min(chain_lengths)
    max_chain = max(chain_lengths)

    chain_df = pd.DataFrame(columns = ['chain_length','num_occurrences'])    
    
    for i in range(min_chain,max_chain+1):
        occ = chain_lengths.count(i)
        chain_df = chain_df.append({'chain_length':i,
                                    'num_occurrences':occ},ignore_index=True)
    if verbose:
        print('Shortest chain is: ', min_chain)
        print('Longest chain is: ', max_chain)
        print('Total num of chain is: ', chain_df['num_occurrences'].sum())
    return chain_df
    
   
def compare_bqm(old_bqm, new_bqm,embedding=False):
    '''
    Parameters
    ----------
    old_bqm : dimod.BQM
        Old Binary Quadratic Model
    new_bqm : dimod.BQM
        New, embedded, Binary Quadratic Model
    embedding : dict
        Embedding dict will be passed to "analyze_embedding" to give some info
        
    Returns
    -------
    int
        Extra qubits needed
    '''
    old_bqm_info = bqm_info(old_bqm)
    new_bqm_info = bqm_info(new_bqm)
    var_increase = new_bqm_info['num_lin'] - old_bqm_info['num_lin']
    print("Old BQM var: ", old_bqm_info['num_lin'],"\nNew BQM var: ",
          new_bqm_info['num_lin'], '\nExtra qubits needed: ', var_increase)
    print("Old BQM connectivity: ", old_bqm_info['connectivity'], 
          "\nNew BQM connectivity: ", new_bqm_info['connectivity'])
    if embedding:
        analyze_embedding(embedding,verbose=True)    
    return var_increase
    
# =============================================================================
# SampleSet functions
# =============================================================================

def hist_helper(df,field='chain_length'):
    new = pd.DataFrame(columns = [field])
    for i,rows in df[[field,'num_occurrences']].iterrows():
        energy = rows[0]
        occ = int(rows[1])
        for i in range(occ):
            new = new.append({field:energy},ignore_index=True) 
    return new

def chain_histogram(embedding,
                    single_qubit=False,
                    bins=50,
                    binwidth=None,
                    figsize=(8.48,4.8), 
                    scale_font=1,
                    show=True, 
                    save_path=None):
     
     chain_df = analyze_embedding(embedding)
     if not single_qubit:
         chain_df = chain_df[chain_df['chain_length']>1]
     min_chain = chain_df['chain_length'].min()
     max_chain = chain_df['chain_length'].max()

     fig, ax = plt.subplots(figsize = figsize)
     sns.set(style="whitegrid", font_scale = scale_font)    

     chain_df = hist_helper(chain_df, 'chain_length')
     s = sns.displot(chain_df,
                 x = 'chain_length',
                 bins = 50,
                 binwidth=binwidth,
                 )
                 
     s.axes[0][0].set_xticks(np.arange(min_chain, max_chain+1, 1))
     s.axes[0][0].set(xlabel='Chain length', ylabel='Occurrences')
     s.axes[0][0].grid(True,which="both")

     if show:
         plt.show()
     if save_path is not None:
         plt.savefig(save_path)       


def sample_histogram(sampleset: dimod.SampleSet,
                     bins=50,
                     binwidth=0.5,
                     figsize=(8.48,4.8), 
                     scale_font=1,
                     label='QA (Advantage)',
                     show=True, 
                     save_path=None):    
    fig, ax = plt.subplots(figsize = figsize)
    sns.set(style="whitegrid", font_scale = scale_font)    
    
    hist_df = hist_helper(sampleset.to_pandas_dataframe())
    sns.histplot(hist_df['energy'],bins=bins,binwidth=binwidth,color='blue', label=label)

    # ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set(xlabel='Energy', ylabel='Occurrences')
    ax.grid(True,which="both")
    ax.legend()
    
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)        
        
        
def chain_histogram_from_sample(sampleset: dimod.SampleSet,
                                single_qubit=False,
                                bins=50,
                                binwidth=None,
                                figsize=(8.48,4.8), 
                                scale_font=1,
                                show=True, 
                                save_path=None):
     embedding = sampleset.info['embedding_context']['embedding']
     return chain_histogram(embedding,
                            single_qubit=single_qubit,
                            bins=bins,
                            binwidth=binwidth,
                            figsize=figsize, 
                            scale_font=scale_font,
                            show=True, 
                            save_path=None)