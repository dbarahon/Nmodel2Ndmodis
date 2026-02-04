# Nmodel2Ndmodis

Set of translation models to sample model output consistent with MODIS retrievals, so they can be consistently compared, as detailed in Bias in spaceborne cloud droplet number concentration retrieval and new alternatives https://essopenarchive.org/doi/full/10.22541/essoar.175691401.16440458

**MD9002MODIS**: Translates grid average cloud droplet number concentration (Nd, cm-3) at 900 hPa (1 degree resolution) to MODIS-type derived Nd to be compared against retrievals. 

**NDCLDBASE2MODIS**: Translates in-cloud, cloud-base droplet number concentration (Nd, cm-3) to MODIS-type derived Nd to be compared against retrievals. 

MODIS-type retrieva are obtained by applying the algorithm of Grosvenor et al. 2018 (https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2017RG000593) to COSP-simulated MODIS output.

