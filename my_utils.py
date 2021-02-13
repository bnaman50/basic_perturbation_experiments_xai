# Begin: Python 2/3 compatibility header small
# Get Python 3 functionality:
from __future__ import\
    absolute_import, print_function, division, unicode_literals
from future.utils import raise_with_traceback, raise_from
# catch exception with: except Exception as e
from builtins import range, map, zip, filter
from io import open
## I built on top of the file provided in the innvestigate lib and added my own functions
import six
# End: Python 2/3 compatability header small

import matplotlib.pyplot as plt
import numpy as np
import os, ipdb
import PIL.Image
import shutil
import time


def load_image(path, size):
    ret = PIL.Image.open(path)
    #ret.show()
    ret = ret.resize((size, size))
    ret = np.asarray(ret, dtype=np.uint8).astype(np.float32)
    return ret

def plot_image_grid_final(grid,
                          row_labels_left,
                          row_labels_right,
                          col_labels,
                          file_name=None):
    # import ipdb; ipdb.set_trace()
    nRows = len(grid)
    nCols = len(grid[0])
    plt.rcParams.update({'font.size': 5})
    plt.rc("font", family="sans-serif")
    plt.rc("axes.spines", top=False, right=False, left=False, bottom=False)
    
    print('Plotting the figure')
    
    tRows = nRows + 2 # total rows
    tCols = nCols + 2 # total cols
    
    wFig = tCols # Figure width (two more than nCols because I want to add ylabels on the very left and very right of figure)
    hFig = tRows # Figure height (one more than nRows becasue I want to add xlabels to the top of figure)
    
    fig, axes = plt.subplots( nrows = tRows, ncols = tCols, figsize = ( wFig, hFig ) )
    fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    axes = np.reshape( axes, (tRows, tCols) )
    
    scale = 0.8
    
    for r in range(tRows):
        for c in range(tCols):
            ax = axes[r][c]
            l, b, w, h = ax.get_position().bounds
            ax.set_position([l, b, w*scale, h*scale])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.spines['left'].set_visible(False)  
            ax.set_xticks([])
            ax.set_yticks([])
            
            if r>0 and c>0 and r<tRows-1 and c<tCols-1:
                im = ax.imshow(grid[r-1][c-1], interpolation='none')            

                if not r-1:
                    if col_labels != []:
                        ax.set_title(col_labels[c-1],
                                     rotation=45,
                                     horizontalalignment='left',
                                     verticalalignment='bottom')
                                     
                if not c-1:
                    if row_labels_left != []:
                        txt_left = [l+'\n' for l in row_labels_left[r-1]]
                        ax.set_ylabel(''.join(txt_left),
                                      rotation=0,
                                      verticalalignment='center',
                                      horizontalalignment='right',
                                     )
                else: 
                    w_cbar = 0.005
                    h_cbar = h*scale
                    b_cbar = b
                    l_cbar = l + scale*w + 0.001  
                    cbaxes = fig.add_axes([l_cbar, b_cbar, w_cbar, h_cbar])                    
                    cbar = fig.colorbar(im, cax = cbaxes) 
                    cbar.outline.set_visible(False)
                    #cbar.set_ticks([])
                    cbar.ax.tick_params(labelsize=2)

                    if c == tCols-2:
                        if row_labels_right != []:
                            txt_right = [l+'\n' for l in row_labels_right[r-1]]
                            ax2 = ax.twinx()
                            
                            #ax2.axis('off')
                            ax2.set_xticks([])
                            ax2.set_yticks([])
                            ax2.spines['top'].set_visible(False)
                            ax2.spines['right'].set_visible(False)
                            ax2.spines['bottom'].set_visible(False)
                            ax2.spines['left'].set_visible(False)
                            ax2.set_ylabel(''.join(txt_right),
                                           rotation=0,
                                           verticalalignment='center',
                                           horizontalalignment='left',
                                          )
        
    print('Saving figure to {}'.format(file_name))
    tp = file_name
    tp = tp.split('/')
    itp = '/' + tp[-1]
    tp = tp[:-1]
    tp = '/'.join(tp)
    mkdir_p(tp)
    
    plt.savefig((tp+itp),
                orientation='landscape', 
                dpi=224/scale, 
                transparent=True, 
                frameon=False, 
                ) 
    plt.close(fig)

      
      
def mkdir_p(mypath):
    '''Creates a directory. equivalent to using mkdir -p on the command line'''

    from errno import EEXIST
    from os import makedirs,path

    try:
        makedirs(mypath)
    except OSError as exc: # Python >2.5
        if exc.errno == EEXIST and path.isdir(mypath):
            pass
        else: raise
