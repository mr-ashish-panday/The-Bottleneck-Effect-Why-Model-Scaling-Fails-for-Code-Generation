from setuptools import setup, find_packages

setup(
    name="code_execution_failures",
    version="0.1.0",
    description="Research code for Paper 11: Why Code Generation Actually Fails",
    author="Your Name",
    author_email="your.email@university.edu",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "datasets>=2.14.0",
        "human-eval>=1.0.3",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "pyyaml>=6.0",
        "tqdm>=4.65.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "black>=23.3.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0",
        ],
        "viz": [
            "matplotlib>=3.7.0",
            "seaborn>=0.12.0",
            "plotly>=5.14.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "generate-samples=scripts.generate_samples:main",
            "run-evaluation=scripts.run_evaluation:main",
            "analyze-failures=scripts.analyze_failures:main",
        ],
    },
)