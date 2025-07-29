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
]

requirements_dev = [
    "ruff",
    "isort",
    "jupyter",
    "pre-commit",
    "pytest",
    "pytest-cov",
]

setup(
    name="IOTC_Panel_Dashboard",
    version="0.1",
    description="Panel-based Dashboard",
    package_dir={"": "panel_dashboard"},
    url="https://github.com/IRATT/IoTC.git",
    author="Adrian Batista",
    packages=[
        "preprocessing",
        "test",
    ],
    install_requires=requirements,
    extras_require={
        "dev": requirements_dev,
    },
)