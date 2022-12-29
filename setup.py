import setuptools  # 导入setuptools打包工具

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="UCMEC",
    version="1.0.0",
    author="Langtian Qin",
    author_email="qlt315@mail.ustc.edu.com",
    description="Reinforcement Learning Environment for User-centric Mobile Edge Computing ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/qt315/Multi-agent-UCMEC",  # 自己项目地址，比如github的项目地址
    packages=setuptools.find_packages(),
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    install_requires=["numpy", "gym>=0.12"],
    include_package_data=True,
)