from vedo import load, Volume, show
import os


path_train_health = "./Training/health"
path_train_patient = "./Training/health"

idx = 8
train_health_filenames = os.listdir(path_train_health)
train_patient_filenames = os.listdir(path_train_patient)
sample_path_health = os.path.join(path_train_health, train_health_filenames[idx])
sample_path_patient = os.path.join(path_train_health, train_health_filenames[idx])

mesh = Volume(sample_path_patient)
show(mesh, title="Health")
