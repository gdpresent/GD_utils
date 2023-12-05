import setuptools
setuptools.setup(
                name="GD_utils",
                version="0.2.0.42",
                license='MIT',
                author="GD park",
                author_email="gdpresent@naver.com",
                description="backtest tools",
                long_description=open('README.md').read(),
                url="https://github.com/gdpresent/GD_utils.git",
                packages=setuptools.find_packages(),
                classifiers=[
                    "Programming Language :: Python :: 3",
                    "License :: OSI Approved :: MIT License",
                    "Operating System :: OS Independent"
                            ],
                )