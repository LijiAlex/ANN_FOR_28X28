import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

PROJECT_NAME ="ANN_FOR_28X28"
USERNAME = "LijiAlex"

setuptools.setup(
    name="ann_for_28X28",
    version="0.0.1",
    author=USERNAME,
    author_email="liji.alex@gmail.com",
    description="A generic ANN implementation for 28X28 images.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/LijiAlex/ANN_FOR_28X28",
    project_urls={
        "Bug Tracker": f"https://github.com/{USERNAME}/{PROJECT_NAME}/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages = setuptools.find_packages(where="src"),
    python_requires=">=3.7",
    install_requires=[
        "tensorflow",
        "matplotlib",
        "seaborn",
        "numpy",
        "pandas",
        "pyYAML"
    ]
)