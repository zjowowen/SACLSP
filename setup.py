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
        'gym[mujoco]>=0.26.0',
        'numpy',
        'torch>=2.0.0',
        'matplotlib',
        'wandb',
        'rich',
        'mujoco_py',
        'easydict',
    ]
)
