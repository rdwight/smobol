import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="smobol",
    version="1.0",
    author="Richard Dwight",
    author_email="richard.dwight@gmail.com",
    description="Sparse-grids and Sobol indices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rdwight/smobol",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
