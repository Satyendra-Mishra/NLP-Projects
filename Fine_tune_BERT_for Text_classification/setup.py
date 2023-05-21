import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_desc = f.read()

__version__ = "0.0.0"

REPO_NAME = "Fine_tune_BERT_for_Text_classification"
AUTHOR_NAME = "Satyendra Mishra"

setuptools.setup(
    name=REPO_NAME,
    version=__version__,
    author=AUTHOR_NAME,
    description="A python project on Fine tuning Bert",
    long_description=long_desc,
    long_description_content="text/markdown",
    url=f"https://github.com/{AUTHOR_NAME}/{REPO_NAME}",
    project_urls={
        "Bug tracker": f"https://github.com/{AUTHOR_NAME}/{REPO_NAME}/issues"
    }, 
    package_dir={"": "."},
    packages=setuptools.find_packages(where=".")
)


