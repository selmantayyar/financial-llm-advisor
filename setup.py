from setuptools import setup, find_packages

setup(
    name="financial-llm-advisor",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.9,<3.12",
    description="Production-grade fine-tuned LLM for institutional investment decision support",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/selmantayyar/financial-llm-advisor",
    author="Your Name",
    author_email="your.email@example.com",
    license="MIT",
)