import os


rootdir = './atlas_faisceaux_MNI'
extensions = ('.bundles')

for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        ext = os.path.splitext(file)[-1].lower()
        if ext in extensions:
            print file