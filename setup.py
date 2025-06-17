from setuptools import setup, find_packages

setup(
    name='pointnet_pytorch',
    version='0.1',
    description='PyTorch implementation of PointNet and PointNet++',
    author='yanx27',
    url='https://github.com/yanx27/Pointnet_Pointnet2_pytorch',
    packages=find_packages(include=['pointnet_pytorch', 'pointnet_pytorch.*']),
    install_requires=[
        # 'torch',
        # 'numpy',
        # 'scipy',
        # 'h5py',
        # 'matplotlib',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)