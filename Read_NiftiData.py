import nibabel as nib # need to install nibabel package
import os

# nifti_file is .nii.gz file
def read_nifti_file(nifti_file):
    nii_image = nib.load(nifti_file)
    nii_data = nii_image.get_fdata()
    return nii_data


AD_img_path = './Testing/patient/'
CN_img_path = './Testing/health/'

for file in os.listdir(CN_img_path):
    print('file name:', file)
    data = read_nifti_file(CN_img_path+file)
    print('data size ', data.shape)