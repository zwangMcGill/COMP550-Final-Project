import nilearn
import numpy as np
from scipy.stats import zscore
import matplotlib.pyplot as plt
from nilearn.input_data import NiftiLabelsMasker, NiftiMapsMasker
from nilearn import plotting
import os

weight_all = []
root = '/Users/dblabs/Documents/GitHub/brain_language_nlp/predictions'
for i in os.listdir(root):
    if 'len_1' in i:
        tmp = np.load(os.path.join(root, i), allow_pickle=True).item()
        weight_all.append(tmp['corrs_t'][0,:])
weight_mean = np.mean(np.array(weight_all), axis=0)
data = np.load('/Users/dblabs/Documents/GitHub/brain_language_nlp/predictions/predict_01_with_bert_layer_3_len_1.npy',allow_pickle=True).item()
weight = weight_mean
schaefer = nilearn.datasets.fetch_atlas_schaefer_2018(n_rois=100,
                                            yeo_networks=7)
schaefer_rois = np.array(schaefer.labels, dtype=str)
schaefer_atlas = schaefer.maps
masker = NiftiLabelsMasker(labels_img=schaefer_atlas, standardize=False,
                                memory='nilearn_cache').fit()
# data = zscore(weight)
data = weight
data = np.reshape(data, (-1, 1))
nifti = masker.inverse_transform(data.T)
# plotting.plot_glass_brain(nifti, threshold=None, colorbar=True,
#                               plot_abs=False, black_bg=False, cmap='bwr',
#                               symmetric_cbar=True, alpha=1)
#     plt.text(x=10,y=-2,s='test')
view = plotting.view_img_on_surf(nifti, threshold='70%', cmap='RdBu_r',surf_mesh='fsaverage')
view.save_as_html("surface_plot.html")
