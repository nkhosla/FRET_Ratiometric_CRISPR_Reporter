import pandas as pd
import pickle
import matplotlib.pyplot as plt
from scipy import stats, fft
import scipy.signal as signal
import statsmodels.api as sm
import seaborn as sns
import math
import numpy as np
import matplotlib as mpl
import os
from dotmap import DotMap

import re
from scipy.optimize import curve_fit



tc = lambda x: np.divide(x, 60)

def convert_hms_to_s(time_string):
    """
    Converts a time string in the format of hh:mm:ss to seconds.
    """
    h, m, s = time_string.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)


def read_FAM_TAMRA_data(filepath, fam_range=(62, 154), tamra_range=(157,249), na_values=None):
    TAMRA_skipped_rows = lambda x: not (x>=tamra_range[0]-1 and x<tamra_range[1])
    FAM_skipped_rows =  lambda x: not  (x>=fam_range[0]-1 and x<fam_range[1])

    all_FAM_data = pd.read_csv(filepath, skiprows=FAM_skipped_rows, na_values=na_values)
    all_TAMRA_data = pd.read_csv(filepath, skiprows=TAMRA_skipped_rows, na_values=na_values)

    #all_sample_wells = {'Time'}
    #all_wells = set(all_FAM_data.columns.values) 

    #for k,v in sample_wells_dict.items():
    #    all_sample_wells = all_sample_wells | set(v)

    #empty_wells = all_wells - all_sample_wells

    #convert the time to seconds
    all_FAM_data['Time'] = all_FAM_data['Time'].apply(convert_hms_to_s)
    all_TAMRA_data['Time'] = all_TAMRA_data['Time'].apply(convert_hms_to_s)   

    # Time shift the TAMRA read to match the FAM read
    time_delta = all_TAMRA_data['Time'][0] - all_FAM_data['Time'][0]
    all_TAMRA_data_not_timeshifted = all_TAMRA_data.copy()
    all_TAMRA_data['Time'] = all_TAMRA_data['Time'] - time_delta

    #FAM_data = all_FAM_data.drop(empty_wells, axis=1).set_index('Time', drop=True) # drop the empty wells and set the time as the index
    #TAMRA_data = all_TAMRA_data.drop(empty_wells, axis=1).set_index('Time', drop=True) # drop the empty wells and set the time as the index

    #for k,v in sample_wells_dict.items():
   #     sample_wells_dict[k] = [x for x in v if x not in empty_wells]

    return all_FAM_data, all_TAMRA_data



def prop_stats_div(df1, df2):
    """
    Propagates the error of a division of two dataframes along the rows
    outputs df1 / df2
    """

    means1 = df1.mean(axis=1)
    means2 = df2.mean(axis=1)

    stdevs1 = df1.std(axis=1)
    stdevs2 = df2.std(axis=1)

    df_out = pd.DataFrame()

    for k in df1.index:
        cov = np.cov(df1.loc[k], df2.loc[k])[0][1]
        f = means1[k] / means2[k]
        t1 = (stdevs1[k] / means1[k])**2
        t2 = (stdevs2[k] / means2[k])**2
        t3 = -2 * cov / (means1[k] * means2[k])
        t = t1 + t2 + t3
        #print(k,': ',t1, t2, cov)
        df_out.loc[k, 'stdev'] = f * np.sqrt(t)
        df_out.loc[k, 'mean'] = f
    
    return df_out

def simple_div_err_prop(df1, df2):
    f = df1['mean']/df2['mean']
    s = np.sqrt((df1['stdev']/df1['mean'])**2 + (df2['stdev']/df2['mean'])**2)
    ss = f*s

    dfout = pd.DataFrame({'mean': f, 'stdev': ss})
    return dfout


def read_and_process_quotient_data(filepath, well_dict, fam_range=(62, 154), tamra_range=(157,249)):
    all_FAM_data, all_TAMRA_data = read_FAM_TAMRA_data(filepath, fam_range=fam_range, tamra_range=tamra_range)
    neg_wells = well_dict['neg'] # negtive
    low_wells = well_dict['low'] # '2pM HPV16 Target DNA'
    med_wells = well_dict['med'] # '20pM HPV16 Target DNA'
    high_wells = well_dict['high'] # '200pM HPV16 Target DNA'

    neg_FAM_data = all_FAM_data.set_index('Time')[[*neg_wells]].copy()
    low_FAM_data = all_FAM_data.set_index('Time')[[*low_wells]].copy()
    med_FAM_data = all_FAM_data.set_index('Time')[[*med_wells]].copy()
    high_FAM_data = all_FAM_data.set_index('Time')[[*high_wells]].copy()

    neg_TAMRA_data = all_TAMRA_data.set_index('Time')[[*neg_wells]].copy()
    low_TAMRA_data = all_TAMRA_data.set_index('Time')[[*low_wells]].copy()
    med_TAMRA_data = all_TAMRA_data.set_index('Time')[[*med_wells]].copy()
    high_TAMRA_data = all_TAMRA_data.set_index('Time')[[*high_wells]].copy()

    # calculate the quotient
    neg_quotient = neg_FAM_data / neg_TAMRA_data
    low_quotient = low_FAM_data / low_TAMRA_data
    med_quotient = med_FAM_data / med_TAMRA_data
    high_quotient = high_FAM_data / high_TAMRA_data

    # calculate the mean and std
    # Find proper standard deviations of the quotients using propoigation of error

    neg_quotient_stats = prop_stats_div(neg_FAM_data, neg_TAMRA_data)
    low_quotient_stats = prop_stats_div(low_FAM_data, low_TAMRA_data)
    med_quotient_stats = prop_stats_div(med_FAM_data, med_TAMRA_data)
    high_quotient_stats = prop_stats_div(high_FAM_data, high_TAMRA_data)

    retdict = {
        'neg_quotient': neg_quotient,
        'low_quotient': low_quotient,
        'med_quotient': med_quotient,
        'high_quotient': high_quotient,
        'neg_quotient_stats': neg_quotient_stats,
        'low_quotient_stats': low_quotient_stats,
        'med_quotient_stats': med_quotient_stats,
        'high_quotient_stats': high_quotient_stats,
        'neg_FAM_data': neg_FAM_data,
        'low_FAM_data': low_FAM_data,
        'med_FAM_data': med_FAM_data,
        'high_FAM_data': high_FAM_data,
        'neg_TAMRA_data': neg_TAMRA_data,
        'low_TAMRA_data': low_TAMRA_data,
        'med_TAMRA_data': med_TAMRA_data,
        'high_TAMRA_data': high_TAMRA_data
    }

    return retdict

def time_to_significance(sdf, ndf, nstdev=3, req_len_sep=4):
    """explanation"""
    # First we need to adjust the DFs so we have the neg + nstdev*stdev
    # as well as the sample - nstdev*stdev

    adj_neg_val = ndf.mean(axis=1) + (nstdev * ndf.std(axis=1))
    #print(adj_neg_val)

    asdf = pd.DataFrame()

    sdf_stds = sdf.std(axis=1)
    for cname in sdf.columns:
        asdf[cname] = sdf[cname] - (nstdev * sdf_stds)
    #print(asdf)

    # Now we need to mask it to get where asdf is greater than andf
    tracker_df = pd.DataFrame(columns=asdf.columns)


    for cname in asdf.columns:
        tracker_df[cname] = asdf[cname].where(asdf[cname] > adj_neg_val, 0)
    
    tracker_df.mask(tracker_df > 0, 1, inplace=True)

    rolling_avg = tracker_df.rolling(window=req_len_sep).mean()

    #print(rolling_avg)
    times = {}

    for sample in rolling_avg.columns:
        try:
            times[sample] = rolling_avg[rolling_avg[sample] == 1].index[0]
        except:
            times[sample] = np.nan

    v = np.array(list(times.values()))
    v = v[~np.isnan(v)]
    if len(v) == 0:
        m = float('nan')
        s = 0
    else:
        m = np.nanmean(v)
        s = np.nanstd(v)
    #print(v)
    stats = {
        'mean': m,
        'std': s,
    }
        

    return times, stats

def read_and_process_extended_conc_quotient_data(filepath, well_dict, fam_range=(62, 154), tamra_range=(157,249)):
    all_FAM_data, all_TAMRA_data = read_FAM_TAMRA_data(filepath, fam_range=fam_range, tamra_range=tamra_range)
    neg_wells = well_dict['neg'] # negtive
    fm274_wells = well_dict['274fm'] # '247fm HPV16 Target DNA'
    fm823_wells = well_dict['823fm'] # '823fm HPV16 Target DNA'
    pm2_wells = well_dict['2.47pM'] # '2.47pM HPV16 Target DNA'
    pm7_wells = well_dict['7.41pM'] # '7.41pM HPV16 Target DNA'
    pm22_wells = well_dict['22.2pM'] # '22.2pM HPV16 Target DNA'
    pm67_wells = well_dict['66.7pM'] # '66.7pM HPV16 Target DNA'
    pm200_wells = well_dict['200pM'] # '200pM HPV16 Target DNA'


    neg_FAM_data = all_FAM_data.set_index('Time')[[*neg_wells]].copy()
    fm274_FAM_data = all_FAM_data.set_index('Time')[[*fm274_wells]].copy()
    fm823_FAM_data = all_FAM_data.set_index('Time')[[*fm823_wells]].copy()
    pm2_FAM_data = all_FAM_data.set_index('Time')[[*pm2_wells]].copy()
    pm7_FAM_data = all_FAM_data.set_index('Time')[[*pm7_wells]].copy()
    pm22_FAM_data = all_FAM_data.set_index('Time')[[*pm22_wells]].copy()
    pm67_FAM_data = all_FAM_data.set_index('Time')[[*pm67_wells]].copy()
    pm200_FAM_data = all_FAM_data.set_index('Time')[[*pm200_wells]].copy()

    neg_TAMRA_data = all_TAMRA_data.set_index('Time')[[*neg_wells]].copy()
    fm274_TAMRA_data = all_TAMRA_data.set_index('Time')[[*fm274_wells]].copy()
    fm823_TAMRA_data = all_TAMRA_data.set_index('Time')[[*fm823_wells]].copy()
    pm2_TAMRA_data = all_TAMRA_data.set_index('Time')[[*pm2_wells]].copy()
    pm7_TAMRA_data = all_TAMRA_data.set_index('Time')[[*pm7_wells]].copy()
    pm22_TAMRA_data = all_TAMRA_data.set_index('Time')[[*pm22_wells]].copy()
    pm67_TAMRA_data = all_TAMRA_data.set_index('Time')[[*pm67_wells]].copy()
    pm200_TAMRA_data = all_TAMRA_data.set_index('Time')[[*pm200_wells]].copy()



    """
    neg_FAM_data = all_FAM_data.set_index('Time')[[*neg_wells]].copy()
    low_FAM_data = all_FAM_data.set_index('Time')[[*low_wells]].copy()
    med_FAM_data = all_FAM_data.set_index('Time')[[*med_wells]].copy()
    high_FAM_data = all_FAM_data.set_index('Time')[[*high_wells]].copy()

    neg_TAMRA_data = all_TAMRA_data.set_index('Time')[[*neg_wells]].copy()
    low_TAMRA_data = all_TAMRA_data.set_index('Time')[[*low_wells]].copy()
    med_TAMRA_data = all_TAMRA_data.set_index('Time')[[*med_wells]].copy()
    high_TAMRA_data = all_TAMRA_data.set_index('Time')[[*high_wells]].copy()
    """

    """
        'neg' : {'H1', 'H2', 'H3', 'H4', 'H5'}, # negtive
        '274fm' : {'G1', 'G2', 'G3', 'G4', 'G5'}, # '274fm HPV16 Target DNA'
        '823fm' : {'F1', 'F2', 'F3', 'F4', 'F5'}, # '823fm HPV16 Target DNA'
        '2.47pM' : {'E1', 'E2', 'E3', 'E4', 'E5'}, # '2.47pM HPV16 Target DNA'
        '7.41pM' : {'D1', 'D2', 'D3', 'D4', 'D5'}, # '7.41pM HPV16 Target DNA'
        '22.2pM' : {'C1', 'C2', 'C3', 'C4', 'C5'}, # '22.2pM HPV16 Target DNA'
        '66.7pM' : {'B1', 'B2', 'B3', 'B4', 'B5'}, # '66.7pM HPV16 Target DNA'
        '200pM' : {'A1', 'A2', 'A3', 'A4', 'A5'}, # '200pM HPV16 Target DNA'
    }"""




    # calculate the quotient
    neg_quotient = neg_FAM_data / neg_TAMRA_data
    fm274_quotient = fm274_FAM_data / fm274_TAMRA_data
    fm823_quotient = fm823_FAM_data / fm823_TAMRA_data
    pm2_quotient = pm2_FAM_data / pm2_TAMRA_data
    pm7_quotient = pm7_FAM_data / pm7_TAMRA_data
    pm22_quotient = pm22_FAM_data / pm22_TAMRA_data
    pm67_quotient = pm67_FAM_data / pm67_TAMRA_data
    pm200_quotient = pm200_FAM_data / pm200_TAMRA_data
    
    """
    low_quotient = low_FAM_data / low_TAMRA_data
    med_quotient = med_FAM_data / med_TAMRA_data
    high_quotient = high_FAM_data / high_TAMRA_data
    """

    # calculate the mean and std
    # Find proper standard deviations of the quotients using propoigation of error

    neg_quotient_stats = prop_stats_div(neg_FAM_data, neg_TAMRA_data)
    fm274_quotient_stats = prop_stats_div(fm274_FAM_data, fm274_TAMRA_data)
    fm823_quotient_stats = prop_stats_div(fm823_FAM_data, fm823_TAMRA_data)
    pm2_quotient_stats = prop_stats_div(pm2_FAM_data, pm2_TAMRA_data)
    pm7_quotient_stats = prop_stats_div(pm7_FAM_data, pm7_TAMRA_data)
    pm22_quotient_stats = prop_stats_div(pm22_FAM_data, pm22_TAMRA_data)
    pm67_quotient_stats = prop_stats_div(pm67_FAM_data, pm67_TAMRA_data)
    pm200_quotient_stats = prop_stats_div(pm200_FAM_data, pm200_TAMRA_data)


    retdict = {
        'neg_quotient': neg_quotient,
        'fm274_quotient': fm274_quotient,
        'fm823_quotient': fm823_quotient,
        'pm2_quotient': pm2_quotient,
        'pm7_quotient': pm7_quotient,
        'pm22_quotient': pm22_quotient,
        'pm67_quotient': pm67_quotient,
        'pm200_quotient': pm200_quotient,
        'neg_quotient_stats': neg_quotient_stats,
        'fm274_quotient_stats': fm274_quotient_stats,
        'fm823_quotient_stats': fm823_quotient_stats,
        'pm2_quotient_stats': pm2_quotient_stats,
        'pm7_quotient_stats': pm7_quotient_stats,
        'pm22_quotient_stats': pm22_quotient_stats,
        'pm67_quotient_stats': pm67_quotient_stats,
        'pm200_quotient_stats': pm200_quotient_stats,
        'neg_FAM_data': neg_FAM_data,
        'fm274_FAM_data': fm274_FAM_data,
        'fm823_FAM_data': fm823_FAM_data,
        'pm2_FAM_data': pm2_FAM_data,
        'pm7_FAM_data': pm7_FAM_data,
        'pm22_FAM_data': pm22_FAM_data,
        'pm67_FAM_data': pm67_FAM_data,
        'pm200_FAM_data': pm200_FAM_data,
        'neg_TAMRA_data': neg_TAMRA_data,
        'fm274_TAMRA_data': fm274_TAMRA_data,
        'fm823_TAMRA_data': fm823_TAMRA_data,
        'pm2_TAMRA_data': pm2_TAMRA_data,
        'pm7_TAMRA_data': pm7_TAMRA_data,
        'pm22_TAMRA_data': pm22_TAMRA_data,
        'pm67_TAMRA_data': pm67_TAMRA_data,
        'pm200_TAMRA_data': pm200_TAMRA_data
    }


    return retdict


def get_slope_data(df, verbose=False):



    """
    At each time t, calculates the slope of the chart from the beginning to that point
    by doing a linear regression of the data from the beginning to that point.
    The regression is done on all data points in the replicate set
    """

    out_df = pd.DataFrame()

    times = df.index.values
    columns=df.columns

    for t in df.index.values:

        if t==0:
            continue
        
        x = []
        y = []
        rel_data = df.loc[times <= t]
        for c in columns:
            x.extend(rel_data.index.values)
            y.extend(rel_data[c].values)

        if verbose:
            print(x)
            print(y)

        slope, intercept, r, p, std_err = stats.linregress(x, y)

        out_df.loc[t, 'slope'] = slope
        out_df.loc[t, 'intercept'] = intercept
        out_df.loc[t, 'r'] = r
        out_df.loc[t, 'p'] = p
        out_df.loc[t, 'std_err'] = std_err

    return out_df

def read_and_process_general_conc_quotient_data(filepath, well_dict, fam_range=(62, 154), tamra_range=(157,249), na_values=None):
    all_FAM_data, all_TAMRA_data = read_FAM_TAMRA_data(filepath, fam_range=fam_range, tamra_range=tamra_range, na_values=na_values)
    outdict = DotMap()

    for k,v in well_dict.items():
        fam = all_FAM_data.set_index('Time')[[*v]].copy()
        tamra = all_TAMRA_data.set_index('Time')[[*v]].copy()
        quot = fam / tamra
        quot_stats = prop_stats_div(fam, tamra)

        outdict[k]['fam'] = fam
        outdict[k]['tamra'] = tamra
        outdict[k]['quot'] = quot
        outdict[k]['quot_stats'] = quot_stats

    return outdict

# slope stuff

def get_slope_data(df, verbose=False):
    """
    At each time t, calculates the slope of the chart from the beginning to that point
    by doing a linear regression of the data from the beginning to that point.
    The regression is done on all data points in the replicate set
    """

    out_df = pd.DataFrame()

    times = df.index.values
    columns=df.columns

    for t in df.index.values:

        if t==0:
            continue
        
        x = []
        y = []
        rel_data = df.loc[times <= t]
        for c in columns:
            x.extend(rel_data.index.values)
            y.extend(rel_data[c].values)

        if verbose:
            print(x)
            print(y)

        slope, intercept, r, p, std_err = stats.linregress(x, y)

        out_df.loc[t, 'slope'] = slope
        out_df.loc[t, 'intercept'] = intercept
        out_df.loc[t, 'r'] = r
        out_df.loc[t, 'p'] = p
        out_df.loc[t, 'std_err'] = std_err

    return out_df
            
        
def slope_time_to_significance(sdf, ndf, nstdev=3, req_len_sep=4, nan_replace=15000):
    """explanation"""
    # First we need to adjust the DFs so we have the neg + nstdev*stdev
    # as well as the sample - nstdev*stdev

    adj_neg_vals = ndf['slope'] + (nstdev * ndf['std_err'])
    #print(adj_neg_val)

    #asdf = pd.DataFrame()

    for cname in sdf.columns:
        adj_slopes = sdf['slope'] - (nstdev * sdf['std_err'])
    #print(asdf)

    # Now we need to mask it to get where asdf is greater than andf
    tracker_df = adj_slopes.where(adj_slopes > adj_neg_vals, 0)


    
    tracker_df.mask(tracker_df > 0, 1, inplace=True)

    rolling_avg = tracker_df.rolling(window=req_len_sep).mean()

    print(rolling_avg)
    try:
        m = rolling_avg[rolling_avg == 1].index[0]
    except:
        m = nan_replace
        
    mean_time = m
    """time = {}

    for sample in rolling_avg.columns:
        try:
            times[sample] = rolling_avg[rolling_avg[sample] == 1].index[0]
        except:
            times[sample] = np.nan

    v = np.array(list(times.values()))
    v = v[~np.isnan(v)]
    if len(v) == 0:
        m = 15000
        s = 0
    else:
        m = np.nanmean(v)
        s = np.nanstd(v)
    print(v)
    stats = {
        'mean': m,
        'std': s,
    }
        """
    
    stats = {
        'mean': m,
        'std': 0,
    }

    return stats

def calc_tts_stat(experiment_dotmap, nstdev=3, req_run_length=4, is_slope=False, publish_ready=False):
    key_pattern = re.compile(r"[A-Za-z]+_[0-9]+", re.IGNORECASE)
    out_dict = DotMap()

    if 'neg' not in list(experiment_dotmap.keys()):
        raise ValueError('No negative control found. Calculating time to significance requires a negative control.')
    
    for conc, all_channels in experiment_dotmap.items():
        if conc!='neg' and not key_pattern.fullmatch(conc):
            continue

        for channel, df in all_channels.items():
            if channel == 'fam':
                if is_slope:
                    out_dict[conc][channel] = DotMap(slope_time_to_significance(df, experiment_dotmap.neg.fam, nstdev=nstdev, req_len_sep=req_run_length))
                else:
                    out_dict[conc][channel] = DotMap(time_to_significance(df, experiment_dotmap.neg.fam, nstdev=nstdev, req_len_sep=req_run_length)[1])

            elif channel == 'quot':
                if is_slope:
                    out_dict[conc][channel] = DotMap(slope_time_to_significance(df, experiment_dotmap.neg.quot, nstdev=nstdev, req_len_sep=req_run_length))
                else:
                    out_dict[conc][channel] = DotMap(time_to_significance(df, experiment_dotmap.neg.quot, nstdev=nstdev, req_len_sep=req_run_length)[1])

            else:
                continue

    return out_dict


def read_and_process_clinical_quotient_data(filepath, well_dict, fam_range=(62, 154), tamra_range=(157,249)):
    all_FAM_data, all_TAMRA_data = read_FAM_TAMRA_data(filepath, fam_range=fam_range, tamra_range=tamra_range)

    bhq_wells = well_dict['bhq']
    tamra_wells = well_dict['tamra']

    fam_bhq_data = all_FAM_data.set_index('Time')[[*bhq_wells]].copy()
    fam_tamra_data = all_FAM_data.set_index('Time')[[*tamra_wells]].copy()
    tamra_data = all_TAMRA_data.set_index('Time')[[*tamra_wells]].copy()

    # calculate the quotient
    quotient = fam_tamra_data / tamra_data

    quotient_stats = prop_stats_div(fam_tamra_data, tamra_data)

    fam_bhq_stats = pd.DataFrame()
    fam_bhq_stats['mean'] = fam_bhq_data.mean(axis=1)
    fam_bhq_stats['stdev'] = fam_bhq_data.std(axis=1)

    retdict = {
        'quotient': quotient,
        'quotient_stats': quotient_stats,
        'fam_bhq_data': fam_bhq_data,
        'fam_bhq_stats': fam_bhq_stats,
        }

    return retdict