from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name='transformers-openai-api',
    packages=find_packages(),
    version='1.1.1',
    description='An OpenAI Completions API compatible server for NLP transformers models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    author='Jeffrey Quesnelle',
    author_email='jq@jeffq.com',
    url='https://github.com/jquesnelle/transformers-openai-api/',
    license='MIT',
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        'console_scripts': [
            'transformers-openai-api = transformers_openai_api.__main__:main'
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="transformers openai api nlp machine-learning",
)
