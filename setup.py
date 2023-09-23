from setuptools import setup, find_packages

setup(name='neural-volume-sas',
      version='1.0',
      install_requires=['commentjson', 'scipy', 'tqdm', 'matplotlib',
                        'pyqtgraph', 'PyQt6', 'PyOpenGL', 'opencv-python-headless', 'PyMCubes',
                        'tensorboard', 'PyQt5', 'h5py', 'bs4', 'gdown', 'trimesh', 'Rtree',
                        'pandas', 'open3d', 'pytorch3d', 'fvcore', 'iopath', 'PyWavefront', 'pyrender'],
      packages=find_packages())

