from setuptools import setup
from setuptools_rust import Binding, RustExtension

setup(
    name="nnv-py",
    version="0.1.0",
    rust_extensions=[
        RustExtension("nnv_py.nnv_py", binding=Binding.PyO3, native=True)
    ],
    packages=["nnv_py"],
    # rust extensions are not zip safe, just like C-extensions.
    zip_safe=False,
)
