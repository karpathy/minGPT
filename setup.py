import setuptools

with open("README.md", "r") as fh:
    LONG_DESC = fh.read()
    setuptools.setup(
        name="minGPT",
        version="1.0",
        author="Andrej Karpathy,Omry Yadan",
        description="A PyTorch re-implementation of GPT training",
        long_description=LONG_DESC,
        long_description_content_type="text/markdown",
        tests_require=["pytest"],
        packages=["mingpt"],
        python_requires=">=3.6",
        install_requires=["hydra-core>=1.0.0rc3","tqdm"],
    )
