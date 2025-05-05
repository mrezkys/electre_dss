from setuptools import setup, find_packages

setup(
    name="electre_dss",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'numpy>=1.20.0',
    ],
    description="A Python implementation of the ELECTRE I method for Decision Support Systems.",
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author="Your Name / AI Assistant", # You can change this
    author_email="your_email@example.com", # You can change this
    url="", # Optional: Link to your repository
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # Choose your license
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Decision Science"
    ],
    python_requires='>=3.8',
) 