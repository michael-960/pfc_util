import setuptools
with open('README.txt', 'r') as f:
    long_description = f.read()


setuptools.setup(
    name="pfc_util",
    version="0.0.1",
    author="Michael Wang",
    author_email="mike.wang96029@gapp.nthu.edu.tw",
    description="utitily modules for PFC simulations"
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/michael-960/pfc_util",
    project_urls={
        "Bug Tracker": "https://github.com/michael-960/pfc_util/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.7",
)

