# Installation

Installation requires Rust, some system libraries, and some Python libraries

## Rust

Install Rust with
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

The only modification you need to make to the install process is to select the nightly toolchain rather than stable.

## System libraries:

```
coinor-cbc coinor-libcbc-dev libclang-dev libopenblas-dev
```

These can be installed on Ubuntu with `apt-get`


## Python libaries

```
pip install sphinx sphinxcontrib-apidoc setuptools_rust myst-parser sphinx-rtd-theme
```

docs can be opened at `docs/_build/html/index.html`

## Installation

Run `python setup.py install` which will compile nnv-py with link time optimization and install the library in your virtualenv. It will take a minute or two to run.

docs can be built with 
```
cd docs
make html
```
and opened in a browser at `docs/_build/html/index.html`