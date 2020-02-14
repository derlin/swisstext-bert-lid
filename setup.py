import setuptools

requirements, dependency_links = [], []
with open("requirements.txt", "r") as f:
    lines = [l.strip() for l in f if len(l.strip()) and not l.strip().startswith('#')]
    for r in lines:
        if 'git+' in r:
            assert '#egg=' in r, f'in requirements.txt, line "{r}" is missing an #egg=package_name'
            _, repo_path = r.split('git+')
            link, package = repo_path.split('#egg=')
            #requirements.append(f'{package} @ git+{repo_path}')
            requirements.append(package)
            dependency_links.append(f'git+{repo_path}')
        else:
            requirements.append(r)

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
    classifiers=(
        "Programming Language :: Python :: 3.6",
        "License :: Creative Commons (CC BY-NC 4.0)",
        "Operating System :: OS Independent",
    ),
    install_requires=requirements,
    dependency_links=dependency_links
)
