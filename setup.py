from setuptools import setup, find_packages

setup(
    name='gym_azul',
    version='0.0.1',
    install_requires=['gym', 'numpy'],
    url="https://github.com/axelri/gym-azul",
    packages=find_packages(),
    python_requires='>=3.8',
)
