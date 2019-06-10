
# coding: utf-8

# In[74]:

import numpy as np
import math
import random
import scipy, sys
import scipy.integrate as integrate
import scipy.signal
from scipy.interpolate import interp1d
from scipy import stats
import pandas as pd
from astropy.io import ascii
from astropy.table import Table
from astropy import cosmology as cosmo
from astropy.coordinates import SkyCoord
from astropy import units as u
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[75]:

# The SNR / unit mass of the galaxy from convolution of SFH & DTD; up to some scaling factor for DTD
# Input age (Gyr), M (units irrelevant; see ratio); optionally a DTD &/or SFH and decay parameter for SFH
# Otherwise these default to an inverse time, exponential decay, random SFH decay parameter (0,1)

def sSNR(age,M,dtd = lambda t: t**-1,sfh = lambda t: np.exp(-t),alpha = np.random.rand()):
    dtdAmp = 1 # Free parameter from dtd for scaling  
    # SFH is scaled to produce the galaxy formed mass over age
    # Then ratio of formed mass to galaxy mass after loss to stellar evolution is treated const
    # For this to be reasonbale assumption, requires galaxies to be similar age and mass 
    Mf = 2.3*M
    # SN rate / unit mass of the galaxy
    sSNR = (Mf/M)*dtdAmp*integrate.quad(lambda t: sfh(alpha*t)*dtd(age-t), 0, age-0.04)[0]
    # To avoid zero division, integrate up to just below the age (about shortest possible dtd ~ 40 Myr below) 
    return sSNR


# In[76]:

# Used in selecting mock hosts 
def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability: break
    return item


# In[77]:

# N mock SN host galaxies based on probabilities determined from their SN rates
# Input how many hosts you'd like to choose from galaxy catalog, the galaxy catalog, index to get galaxy age and mass 
def getMockHosts(N,gal_catalog, ageidx, massidx):   
    rates = []
    # Put the SNR for each galaxy in the catalog into a list
    for i in range(len(gal_catalog)):
        rate = sSNR(gal_catalog[i][ageidx],gal_catalog[i][massidx])[0]*gal_catalog[i][massidx]
        rates.append(rate)
    tot = np.nansum(rates) # nansum is important to get value which is not nan; not real sure why
    probs = np.array(rates)/tot
    hosts = []
    for i in range(len(N)):
        host_gal = random_pick(gal_catalog,probs) # picks host galaxy in catalog based on prob
        hosts.append(host_gal)
    return hosts


# In[ ]:



