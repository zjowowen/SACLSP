# write a setup.py file for the package

from setuptools import setup, find_packages

setup(
    name='SACLSP',
    version='0.0.1',
    description='PyTorch implementations of deep reinforcement learning algorithms',
    author='zjowowen',

    packages=find_packages(
        exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=[
        'gym[mujoco]==0.25.1',
        'numpy',
        'torch',
        'matplotlib',
        'wandb',
        'rich',
        'mujoco_py',
        'easydict',
    ]
)
