
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='PWAS',
    version='0.0.1',
    author= 'Mengjia Zhu, Alberto Bemporad',
    author_email='mengjia.zhu@imtlucca.it, alberto.bemporad@imtlucca.it',
    description='PWAS/PWASp - Global and Preference-based Optimization with Mixed Variables using (P)iece(w)ise (A)ffine (S)urrogates',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/mjzhu-p/PWAS.git',
    project_urls = {
        "PWAS/PWASp": "https://github.com/mjzhu-p/PWAS.git"
    },
    license='Apache-2.0',
    packages=['PWAS'],
    install_requires=['numpy','scipy','math','pulp','sklearn','pyDOE','cdd'],
)