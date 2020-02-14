import setuptools

setuptools.setup(
    name="bert_lid",
    version="0.0.1",
    author="Lucy Linder",
    author_email="lucy.derlin@gmail.com",
    description="LID for Swiss-German",
    license='Apache License 2.0',
    long_description="TODO",
    url="https://github.com/derlin/swisstext-bert-lid",

    packages=setuptools.find_packages(),
    package_data={'bert_lid': ['models/*', 'models/*/*', 'models/*/*/*']},  # include everything under models

    entry_points={
      'console_scripts': [
        'bert_lid_install_model=bert_lid.install_model:main'
      ]
    },

    classifiers=(
        "Programming Language :: Python :: 3.6",
        "License :: Creative Commons (CC BY-NC 4.0)",
        "Operating System :: OS Independent",
    ),
    install_requires=[
        'pytorch_pretrained_bert_lid @ git+https://github.com/derlin/transformers.git#egg=pytorch_pretrained_bert_lid-0.6.2', 
        'pandas', 
        'sklearn'
    ]
)
