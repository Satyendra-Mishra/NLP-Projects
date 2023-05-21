import setuptools

with open("README.md", "r", encoding="utf-8") as f:
    long_desc = f.read()

__version__ = "0.0.0"

REPO_NAME = "Fine_tune_BERT_for_Text_classification"
AUTHOR_NAME = "Satyendra Mishra"
SRC_REPO = "Fine_tune_BERT_for_Text_classification"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=AUTHOR_NAME,
    description="A python project on Fine tuning Bert",
    long_description=long_desc,
    long_description_content="text/markdown",
    url=f"https://hithub.com/{AUTHOR_NAME}/{REPO_NAME}",
    project_urls={
        "Bug tracker": f"https://hithub.com/{AUTHOR_NAME}/{REPO_NAME}/issues"
    },
    package_dir={"": "src"}
    packages=setuptools.find_packages(where="Src")
)


