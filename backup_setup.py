from setuptools import setup, find_packages


setup(
    name='CALM',
    version='1.0',
    packages=find_packages(include=['CALM']),
    package_data={
        'CALM': ['analyze/*', 'calibrate/*', 'core/*','map/*','utilize/*'],
    },
    include_package_data=True,
    python_requires='>=3.6',
    install_requires=['numpy','scipy','networkx','matplotlib','MDAnalysis'],
    entry_points={
        'console_scripts': [
            'CALM=CALM.run_modules:main',
        ],
    },
)
