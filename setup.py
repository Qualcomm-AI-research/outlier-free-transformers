from distutils.core import setup
from pathlib import Path

scripts = ["run_mlm", "run_clm", "validate_mlm", "validate_clm"]
bash_scripts = Path("scripts").glob("*.sh")

setup(
    name="outlier_free_transformers",
    version="1.0.0",
    packages=[
        "quantization",
        "quantization.quantizers",
        "transformers_language",
        "transformers_language.models",
    ],
    py_modules=scripts,
    scripts=[str(path) for path in bash_scripts],
    entry_points={"console_scripts": [f"{script} = {script}:main" for script in scripts]},
    url="https://github.com/Qualcomm-AI-research/outlier-free-transformers",
    license="BSD 3-Clause Clear License",
    author="Yelysei Bondarenko and Markus Nagel and Tijmen Blankevoort",
    author_email="{ybond, markusn, tijmen}@qti.qualcomm.com",
    description='Code for "Quantizable Transformers: Removing Outliers by Helping Attention Heads Do Nothing"',
)
