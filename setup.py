from setuptools import setup
from setuptools_rust import Binding, RustExtension

with open('README.md') as fobj:
    long_description = fobj.read()

extras_require = {
    'docs': [
        'Sphinx>=1.7.2,<2.2.1', 'sphinx-rtd-theme>=0.3.0',
        'sphinxcontrib-apidoc>=0.3.0'
    ],
}
extras_require['complete'] = sorted(set(sum(extras_require.values(), [])))

setup(
    name="nnv-py",
    use_scm_version=True,
    description="Analyze deep neural networks with verification techniques",
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    author="Eleanor Quint",
    author_email='eleanorquint1@gmail.com',
    install_requires=['numpy', 'dm-tree', 'scipy'],
    setup_requires=[
        "setuptools", "wheel", "setuptools-rust", "setuptools_scm"
    ],
    extras_require=extras_require,
    packages=["nnv_py"],
    url='https://github.com/DrKwint/nnv-py',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License'
    ],
    rust_extensions=[
        RustExtension("nnv_py.nnv_py", binding=Binding.PyO3, native=True)
    ],
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False)
