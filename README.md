# CALM (Calibrate and Analyze Lipid Membranes)

CALM is a command-line tool to swiftly analyze the large scale geometric properties of a lipid membrane. CALM is intended to be used on almost flat membranes to detect, for instance, the curvature induced by a protein. Additionally, it allows the calculation of parameters to translate the protein features to be understood by FreeDTS [FreeDTS](https://github.com/weria-pezeshkian/FreeDTS).

## State

**NOTE**: Under developement!

## Installation

### Prerequisites

Put dependecies here.

### Install CALM
#### Directly from GitHub
```console
pip3 install git+https://github.com/weria-pezeshkian/Membrane_Shape_Analysis
```
#### From source
```console
git clone https://github.com/weria-pezeshkian/Membrane_Shape_Analysis
cd Membrane_Shape_Analysis
python3 -m venv venv && source venv/bin/activate # Not required, but often convenient.
pip3 install .
```

## Usage
For help on CALM and it's executables run:

```console
CALM -h
CALM {calibrate,analyze,link, map} -h
```





