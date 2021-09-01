"""
Make volume and surface templates

This script computes the average contrast across sessions. A consistency is created in which all
voxels/vertices are set to zero having inconsistent signs across sessions regardless of their actual 
effect size. 

created by Daniel Haenelt
Date created: 27-06-2020
Last modified: 29-06-2020  
"""
import os
import numpy as np
import nibabel as nb
from lib_column.io.LoadData import LoadData

# input
participant = "p3"
name_sess = ["GE_EPI1", "GE_EPI2", "GE_EPI3", "SE_EPI1", "SE_EPI2", "VASO1", "VASO3"]
path_output = "/data/pt_01880/Experiment1_ODC/p3/analysis/template"
unit = "Z" # Z or raw
contrast = "left_right" # left_right or left_rest

""" do not edit below """

def sum_contrast(arr1, arr2, sess):
    if sess[0] == "V":
        arr2 = np.negative(arr2)
    
    return arr1 + arr2

def threshold_sign(arr1, arr2, sess):
    if sess[0] == "V":
        arr2 = np.negative(arr2)
    
    arr_temp = np.sign(arr1) * np.sign(arr2)
    arr1[arr_temp != 1] = 0
    
    return arr1

# make output folder
if not os.path.exists(path_output):
    os.makedirs(path_output)

# number of images
n = len(name_sess)

# initialize volume
vol = LoadData(participant, None, name_sess[0], unit, contrast).get_transformed()["data"]

# initialize res
header = vol.header
affine = vol.affine

# volume template
arr_res = np.zeros_like(vol.get_fdata())
arr_sign = np.sign(vol.get_fdata())
for i in range(n):
    vol = LoadData(participant, None, name_sess[i], unit, contrast).get_transformed()["data"]
    arr_res = sum_contrast(arr_res, vol.get_fdata()/n, name_sess[i])
    arr_sign = threshold_sign(arr_sign, vol.get_fdata(), name_sess[i])

# compute consistency map
arr_consistency = arr_res.copy()
arr_consistency[arr_sign == 0] = 0

# write output
output = nb.Nifti1Image(arr_res, affine, header)
nb.save(output, os.path.join(path_output, "vol_average_"+unit+"_"+contrast+".nii"))

output = nb.Nifti1Image(arr_consistency, affine, header)
nb.save(output, os.path.join(path_output, "vol_consistency_"+unit+"_"+contrast+".nii"))

# surface template
hemi = ["lh","rh"]
for i in range(len(hemi)):
    
    # initialize volume
    surf = LoadData(participant, hemi[i], name_sess[0], unit, contrast).get_sampled()["data"]
    
    # initialize res
    header = surf.header
    affine = surf.affine
    
    arr_res = np.zeros_like(surf.get_fdata())
    arr_sign = np.sign(surf.get_fdata())
    for j in range(n):
        surf = LoadData(participant, hemi[i], name_sess[j], unit, contrast).get_sampled()["data"]
        arr_res = sum_contrast(arr_res, surf.get_fdata()/n, name_sess[j])
        arr_sign = threshold_sign(arr_sign, surf.get_fdata(), name_sess[j])

    # compute consistency map
    arr_consistency = arr_res.copy()
    arr_consistency[arr_sign == 0] = 0

    # write output
    output = nb.Nifti1Image(arr_res, affine, header)
    nb.save(output, os.path.join(path_output, hemi[i]+".surf_average_"+unit+"_"+contrast+".mgh"))

    output = nb.Nifti1Image(arr_consistency, affine, header)
    nb.save(output, os.path.join(path_output, hemi[i]+".surf_consistency_"+unit+"_"+contrast+".mgh"))