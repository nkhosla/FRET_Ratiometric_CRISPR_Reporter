import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from scipy.optimize import curve_fit

class AxisPackage:
    def __init__(self, xy_pairs, labels, errs, axis_names):
        self.xy_pairs = xy_pairs
        self.labels = labels
        self.errs = errs
        self.x_name = axis_names[0]
        self.y_name = axis_names[1]
    
def insert_line_with_errorfill(x,y,err,ax,color, err_scalar=1, label=None):
    ax.plot(x,y,color=color, label=label, linewidth=0.8)
    ax.fill_between(x, y - err*err_scalar, y + err*err_scalar, color=color, alpha=0.2, linewidth=0)

def general_lineplot_errorfill(axis_packages, stdmult=1, neg_std_mult=1, figsize=(3.30708661, 6.61417323),  publish_ready=False, colormap=None, color_output=False, ax=None, fig=None, legend=True, **kwargs):
    #colors = ['#785EF0', '#DC267F', '#FE6100', '#FFB000'] # The origianl nice circus colors
    #colors = ['#b2e2e2', '#66c2a4', '#2ca25f', '#006d2c'] # green gradient is meh
    #colors = ['#045a8d', '#2b8cbe','#74a9cf', '#bdc9e1'] # blue gradient is nice
    #colors = ['#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b']
    # figure out how many lines we need
    if not colormap:
        ocolormap = mpl.colormaps['Blues']
        colors = ocolormap(np.linspace(0.35, 1.0, 1024))
        colormap = mpl.colors.ListedColormap(colors)


    #colors = ['#deebf7', '#c6dbef', '#9ecae1', '#6baed6', '#4292c6', '#2171b5', '#08519c', '#08306b', '#FE6100', '#FFB000', '#2b8cbe']
    
    
    num_graph_cols = np.shape(axis_packages)[1]
    num_graph_rows = np.shape(axis_packages)[0]

    big_title_size = 9
    small_title_size = 9
    axis_size = 8
    legend_size = 8
    axis_tick_size = 6


    if not ax:
        fig, ax = plt.subplots(num_graph_rows, num_graph_cols, figsize=figsize, **kwargs)

    for col in range(num_graph_cols):
        for row in range(num_graph_rows):
            #print(f'col {col+1} of {num_graph_cols} and row {row+1} of {num_graph_rows}')
            axpack = axis_packages[row][col]
            lines = axpack.xy_pairs
            colors = colormap(np.linspace(0, 1, len(lines)))
            if color_output:
                print(np.multiply(colors, 255))
            #print(colors)
            color_iter = iter(colors)
            for i in range(len(lines)):
                line = lines[i]
                c = next(color_iter)

                if len(np.shape(ax)) == 0: # if the shape is empty, then it's a single point
                    ax = [[ax]]
                elif len(np.shape(ax)) == 1: # one dimensional array, so it's a single row
                    ax = [ax]
                elif len(np.shape(ax)) == 2: # two dimensional array, so it's a single column
                    pass
                else:
                    raise ValueError('The axis array is too big. The maximum dimension of the axis_packages array is 2')
                
                axis = ax[row][col]
                #print(f'row: {row}, col: {col}, i: {i}')

                insert_line_with_errorfill(line[0], line[1], axpack.errs[i], axis, c, err_scalar=stdmult, label=axpack.labels[i])

                if not publish_ready:
                    axis.set_xlabel(axpack.x_name, fontsize=axis_size)
                    axis.set_ylabel(axpack.y_name, fontsize=axis_size)
           
                axis.yaxis.set_major_locator(plt.MaxNLocator(6))

                axis.ticklabel_format(axis='y', style='sci', scilimits=(0,2))

                axis.tick_params(axis='both', labelsize=axis_tick_size, direction='out', which='both')

                axis.spines['right'].set_visible(False)
                axis.spines['top'].set_visible(False)
                #axis.tick_params(axis='y', labelsize=axis_tick_size)

                axis.yaxis.get_offset_text().set_fontsize(axis_tick_size)
                axis.yaxis.set_tick_params(labelsize=axis_tick_size)

                axis.tick_params(axis='both', which='major', pad=1)

            #axis.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

            #axis.set_xticklabels(axis.get_xticks(), fontsize=axis_tick_size)
            #axis.set_yticklabels(axis.get_yticks(), fontsize=axis_tick_size)

          
    if not publish_ready and legend:
        fig.legend(loc='upper center', bbox_to_anchor=(0.5, 0.01),
            fancybox=True, shadow=True, ncol=4, fontsize=legend_size, frameon=False)
    
    return fig, ax


def tts_plot(data_dict, n_stdev=3, colormap=None, dpi=500, figsize=(8.5, 8.5), publish_ready=False, include_fit=False, color_output=False, ax=None, **kwargs):
    """
    The dict in should be of the format {label: [[x_arr], [y_arr], [err_arr]], label2:....}
    """

    big_title_size = 9
    small_title_size = 9
    axis_size = 8
    legend_size = 8
    axis_tick_size = 6

    if not colormap:
        ocolormap = mpl.colormaps['Blues']
        colors = ocolormap(np.linspace(0.35, 1.0, 1024))
        colormap = mpl.colors.ListedColormap(colors)

    colors = colormap(np.linspace(0, 1, len(data_dict.keys())))
    if color_output:
        print(np.multiply(colors, 255))

    color_iter = iter(colors)

    if not ax:
        fig, axis = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    else:
        axis = ax

    for k,v in data_dict.items():
        #print('k: ', k)
        #print('v: ', v)

        x = np.array(v[0])
        y = np.array(v[1])
        err = np.array(v[2])
        scaled_err = np.multiply(err, n_stdev)
        c = next(color_iter)

        #print('x: ', x)
        #print('y: ', y)
        #print('err: ', err)
        #print('c: ', c)

        axis.errorbar(x, y, yerr=scaled_err, c=c, label=k, fmt='o', markersize=2.5, elinewidth=1, capsize=1.5, capthick=1)

        

        if include_fit:
            #f = lambda x, a, b, c: a * np.exp(-b * x) + c
            f = lambda x, a, b: a*(2.718**(-x))
            f = lambda x, a, n, c : a/(x**n) + c

            # get rid of nans
            xfit = x[~np.isnan(y)]
            yfit = y[~np.isnan(y)]
            errfit = err[~np.isnan(y)]

            x_vals_for_fit = np.linspace(min(x[~np.isnan(y)]), max(x[~np.isnan(y)]), 1000000)
        
            popt, pcov, info, msg, iflag = curve_fit(f, xfit, yfit, full_output=True)
            #print(msg)
            #print('x: ', xfit)
            #print('y: ', yfit)
            print ('popt: ', popt)

            y_fit = f(x_vals_for_fit, *popt)

            ax.plot(x_vals_for_fit, y_fit, c=c, ls='--', lw=1)

        # Pretty up the plot
        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)
        axis.tick_params(axis='both', labelsize=axis_tick_size, direction='out', which='both')
        axis.yaxis.set_major_locator(plt.MaxNLocator(6))
        axis.ticklabel_format(axis='y', style='sci', scilimits=(0,2))
        axis.yaxis.get_offset_text().set_fontsize(axis_tick_size)
        axis.yaxis.set_tick_params(labelsize=axis_tick_size)

    




    if not publish_ready:
        axis.set_ylabel('Time (s)', fontsize=axis_size)
        axis.set_xlabel('Trigger Concentration (uM)', fontsize=axis_size)
        axis.legend(fontsize=legend_size)
        axis.set_title('Time to Assay Significance')

    
    if not ax:
        return fig, axis
    else:
        return ax