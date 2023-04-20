from setuptools import setup

exec (open('einstein_exchange_components/version.py').read())

setup(
    name='einstein_exchange_components',
    version=__version__,
    author='',
    packages=['einstein_exchange_components'],
    include_package_data=True,
    license='MIT',
    description='Custom plotly dash components for Einstein Exchange',
    install_requires=[]
)
