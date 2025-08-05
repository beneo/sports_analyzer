import pathlib
import setuptools

# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text(encoding="utf-8")

setuptools.setup(
    name="sports_analyzer",
    version='0.1.0',
    python_requires=">=3.8",
    description="Sports analytics toolkit for computer vision-based analysis",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/your-username/sports_analyzer",
    author="Sports Analytics Team",
    author_email="team@sports-analyzer.com",
    license='MIT',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=[
        # Core data science stack - Latest compatible versions
        "numpy>=2.2.0",
        "pandas>=2.2.0", 
        "scipy>=1.16.0",
        
        # Computer vision and ML - Latest versions
        "opencv-python>=4.12.0",
        "supervision>=0.26.0",
        "scikit-learn>=1.7.0",
        "umap-learn>=0.5.9.post2",
        
        # Deep learning - Latest compatible versions
        "torch>=2.7.0",
        "transformers>=4.54.0",
        "sentencepiece>=0.2.0",
        "protobuf>=6.31.0",
        
        # Utilities - Latest versions
        "tqdm>=4.67.0",
        "pillow>=11.3.0",
        "matplotlib>=3.10.0",
        "seaborn>=0.13.0",
        
        # Configuration and logging - Latest versions
        "pydantic>=2.11.0",
        "loguru>=0.7.3",
        "omegaconf>=2.3.0",
        
        # CLI tools - Latest versions
        "click>=8.1.0",
        "rich>=14.0.0",
    ],
    extras_require={
        'dev': [
            # Testing
            'pytest>=7.0.0',
            'pytest-cov>=3.0.0',
            'pytest-mock>=3.7.0',
            
            # Code quality
            'black>=22.0.0',
            'isort>=5.10.0',
            'flake8>=4.0.0',
            'mypy>=0.950',
            
            # Documentation
            'sphinx>=4.5.0',
            'sphinx-rtd-theme>=1.0.0',
            
            # Development tools
            'pre-commit>=2.17.0',
            'jupyter>=1.0.0',
            'ipywidgets>=7.7.0',
        ],
        'gpu': [
            'torch>=2.7.0',
        ],
        'all': [
            'ultralytics>=8.0.0',
            'gdown>=4.6.0',
        ]
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Typing :: Typed',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Operating System :: Unix',
        'Operating System :: MacOS'
    ]
)