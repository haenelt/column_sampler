"""
Load data for the ODC project.

created by Daniel Haenelt
Date created: 29-06-2020             
Last modified: 13-10-2020  
"""
import os
import sys
import nibabel as nb
from nibabel.freesurfer.io import read_geometry
from gbb.utils.vox2ras import vox2ras

class LoadData:
    """
    Loads necessary data for the columnar analysis of the ODC project.
    """
    
    # fix subjects directory
    subjects_dir = "/data/pt_01880/Experiment1_ODC/"
    
    def __init__(self, subject: str, hemisphere: str, session: str, unit: str, contrast: str):
        self.sub = subject
        self.hemi = hemisphere
        self.sess = session
        self.unit = unit
        self.contrast = contrast
        
        if self.sub not in ["p1", "p2", "p3", "p4", "p5"]:
            sys.exit("No valid participant name!")

    
    def get_transformed(self, s=subjects_dir):
        """
        Loads registered contrast data.
        """
        path = os.path.join(s, self.sub, "odc/results")
        prefix = "Z" if self.unit == "Z" else "psc"
        basename = prefix+"_all_"+self.contrast+"_"+self.sess+"_transformed"
        file_in = os.path.join(path, self.unit, "transformed", basename+".nii")
        
        res = {}
        res["data"] = nb.load(file_in)
        res["name"] = file_in
        
        return res
    
    def get_sampled(self, s=subjects_dir):
        """
        Loads sampled contrast data.
        """
        path = os.path.join(s, self.sub, "odc/results")
        prefix = "Z" if self.unit == "Z" else "psc"
        basename = self.hemi+"."+prefix+"_all_"+self.contrast+"_"+self.sess+"_upsampled_avg_layer2_8"
        file_in = os.path.join(path, self.unit, "sampled", basename+".mgh")
        
        res = {}
        res["data"] = nb.load(file_in)
        res["name"] = file_in
        
        return res

    def get_native(self, s=subjects_dir):
        """
        Loads contrast data in native space.
        """
        path = os.path.join(s, self.sub, "odc/results")
        prefix = "Z" if self.unit == "Z" else "psc"
        basename = prefix+"_all_"+self.contrast+"_"+self.sess
        file_in = os.path.join(path, self.unit, "native", basename+".nii")
        
        res = {}
        res["data"] = nb.load(file_in)
        res["name"] = file_in
        
        return res
    
    def get_deformation(self, s=subjects_dir):
        """
        Loads source to target deformation.
        """
        path = os.path.join(s, self.sub, "deformation/odc", self.sess)
        basename = "source2target"
        file_in = os.path.join(path, basename+".nii.gz")
        
        res = {}
        res["data"] = nb.load(file_in)
        res["name"] = file_in
        
        return res
    
    def get_rim(self, s=subjects_dir):
        """
        Loads rim file.
        """
        path = os.path.join(s, self.sub, "anatomy/layer")
        rim_hemi = "left" if self.hemi =="lh" else "right"
        basename = "rim"
        file_in = os.path.join(path, rim_hemi, basename+".nii")
        
        res = {}
        res["data"] = nb.load(file_in)
        res["name"] = file_in
        
        return res

    def get_mesh(self, s=subjects_dir):
        """
        Loads mesh file.
        """
        path = os.path.join(s, self.sub, "anatomy/mesh")
        basename = "mesh"
        file_in = os.path.join(path, self.hemi+"."+basename)
        
        res = {}
        res["vtx"], res["fac"] = read_geometry(file_in)
        res["name"] = file_in
        
        return res
    
    def get_matrix(self, s=subjects_dir):
        """
        Loads header transformation of native target space.
        """
        path = os.path.join(s, self.sub, "resting_state")
        basename = "mean_udata"
        file_in = os.path.join(path, basename+".nii")
        
        res = {}
        res["vox2ras"], res["ras2vox"] = vox2ras(file_in)
        res["name"] = file_in
        
        return res