import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="data_kaggle_airport",
    version="0.0.1",
    author="Sheldon Lee-Loy",
    author_email="",
    description="package containing helper script to analyze flight delays kaggle data",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sleeloy99/data_kaggle_airport.git",
    packages=setuptools.find_packages(),
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
)
