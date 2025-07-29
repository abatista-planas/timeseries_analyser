from setuptools import setup  # type: ignore

requirements = [
    'importlib-metadata; python_version == "3.10"',
    "param",
    "scikit-learn>=1.1.1",
    "matplotlib",
    "numpy",
    "pandas",
    "seaborn",
    "scipy",
    "statsmodels",
    "pymannkendall",
    "openpyxl",
    "xgboost",
    "joblib",
    "lightgbm",
    "catboost",
    "torch",
    "torchvision",
    "pytorch-lightning",
    "pandas-profiling",
    "umap-learn",
]

requirements_dev = [
    "ruff",
    "isort",
    "jupyter",
    "pre-commit",
    "pytest",
    "pytest-cov",
    # Type stubs
    "pandas-stubs",
]

setup(
    name="timeseries_analyser",
    version="0.1",
    description="A package for time series data preprocessing and analysis",
    url="https://github.com/abatista-planas/timeseries_analyser.git",
    author="Adrian Batista",
    packages=[
        "preprocessing",
        "test",
        "models",
        "visualization",
    ],
    install_requires=requirements,
    extras_require={
        "dev": requirements_dev,
    },
)
