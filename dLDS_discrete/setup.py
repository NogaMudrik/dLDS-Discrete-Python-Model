import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    packages=setuptools.find_packages(),
    author="noga mudrik",

    name="dLDS_discrete",
    version="0.0.8",
    
    author_email="<nmudrik1@jhmi.edu>",
    description="dLDS discrete model package",
    long_description=long_description,
    long_description_content_type="text/markdown",

    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],

    python_requires=">=3.8",
    install_requires = ['numpy', 'matplotlib','scipy','scipy','pandas','webcolors',
                        'seaborn','colormap','sklearn', 'pylops','os','dill','mat73','pickle']
)

#    package_dir={"": "src"},
#    packages=setuptools.find_packages(where="src"),
#author="nmudrik1",
#    url="https://github.com/NogaMudrik/Discrete-Python-Model---dLDS-paper",
#    project_urls={
#        "mmain code": "https://github.com/NogaMudrik/Discrete-Python-Model---dLDS-paper",
#    }