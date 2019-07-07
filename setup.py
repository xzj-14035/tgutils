from glob import glob
from setuptools import find_packages
from setuptools import setup

import distutils.cmd
import distutils.log
import os
import re
import subprocess
import tgutils.setup_mypy

VERSION = '0.1'


def readme():
    sphinx = re.compile(':py:[a-z]+:(`[^`]+`)')
    with open('README.rst') as readme_file:
        return sphinx.sub('`\\1`', readme_file.read())


def version_from_hg():
    subprocess.check_call(['tools/install_hg_hooks'])

    # PEP440 forbids placing the commit hash in the version number.
    # Counting the commits since the tag must therefore suffice to identify the commit.
    command = ['hg', 'log', '-r', 'tip', '--template', '{latesttag} {latesttagdistance}']
    results = subprocess.check_output(command).decode('utf8').split(' ')
    latest_tag = results[0]
    commits_count_since_tag = results[1]
    global VERSION
    if latest_tag != VERSION:
        print('WARNING: version updated from: %s to %s; you MUST `hg tag %s` after commit!'
              % (latest_tag, VERSION, VERSION))
    version = '%s.%s' % (VERSION, commits_count_since_tag)

    # PEP440 also forbids having a simple '.dev' suffix.
    # Instead we must give an explicit number (0) which is just noise.
    command = ['hg', 'status']
    local_modifications = subprocess.check_output(command)
    if local_modifications:
        version += '.dev0'

    with open('tgutils/version.py', 'w') as file:
        file.write("'''Version is generated by setup.py.'''\n\n")
        file.write("__version__ = '%s'\n" % version)

    return version


def version_from_file():
    if not os.path.exists('tgutils/version.py'):
        raise RuntimeError('Failed to generate tgutils/version.py')

    with open('tgutils/version.py', 'r') as file:
        regex = re.compile("__version__ = '(.*)'")
        for line in file.readlines():
            match = regex.search(line)
            if match:
                return match.group(1)

    raise RuntimeError('Failed to parse tgutils/version.py')


def generate_version():
    if os.path.exists('.hg'):
        return version_from_hg()
    return version_from_file()


class SimpleCommand(distutils.cmd.Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        subprocess.check_call(self.command)


class AllCommand(SimpleCommand):
    description = 'run all needed steps before commit'

    def run(self):
        self.run_command('no_unknown_files')
        self.run_command('no_todo' + 'x')
        self.run_command('is_formatted')
        self.run_command('pylint')
        self.run_command('mypy')
        self.run_command('check')
        self.run_command('build')
        self.run_command('nose')
        self.run_command('tox')
        self.run_command('html')


class CleanCommand(SimpleCommand):
    description = 'remove all generated files and directories'
    command = ['tools/clean']


class HtmlCommand(SimpleCommand):
    description = 'run sphinx to generate HTML documentation'
    command = ['tools/generate_documentation']


class IsformattedCommand(SimpleCommand):
    description = 'use autopep8 and isort to check the formatting of all Python source files'
    command = ['tools/is_formatted']


class MypyCommand(SimpleCommand):
    description = 'run MyPy on all Python source files'
    command = ['mypy',
               '--warn-redundant-casts',
               '--disallow-untyped-defs',
               '--warn-unused-ignores',
               *glob('tgutils/**/*.py', recursive=True),
               *glob('tests/**/*.py', recursive=True),
               *glob('bin/**/*.py', recursive=True)]


class NoseCommand(SimpleCommand):
    description = 'run nosetests and generate coverage reports'
    command = ['nosetests',
               '--with-coverage',
               '--cover-package=tgutils',
               '--cover-branches',
               '--cover-html']

    def run(self):
        if os.path.exists('.coverage'):
            os.remove('.coverage')
        super().run()


class NoTodo_xCommand(SimpleCommand):
    description = 'ensure there are no leftover TODO' + 'X in the source files'
    command = ['tools/no_todo_x']


class PylintCommand(SimpleCommand):
    description = 'run Pylint on all Python source files'
    command = ['pylint',
               '--init-import=yes',
               '--ignore-imports=yes',
               '--disable=' + ','.join([
                   'bad-continuation',
                   'bad-whitespace',
                   'fixme',
                   'global-statement',
                   'no-member',
                   'ungrouped-imports',
                   'wrong-import-order',
               ]),
               *[path for path in glob('tgutils/**/*.py', recursive=True) if 'stubs' not in path],
               *glob('tests/**/*.py', recursive=True),
               *glob('bin/**/*.py', recursive=True)]


class ReformatCommand(SimpleCommand):
    description = 'use autopep8 and isort to fix the formatting of all Python source files'
    command = ['tools/reformat']


class NoUnknownFilesCommand(SimpleCommand):
    description = 'ensure there are no source files hg is not aware of'
    command = ['tools/no_unknown_files']


class ToxCommand(SimpleCommand):
    description = 'run tests in a virtualenv using Tox'
    command = ['tox']


# TODO: Replicated in tox.ini
INSTALL_REQUIRES = ['dynamake', 'pandas', 'numpy', 'pyyaml']
TESTS_REQUIRE = ['nose', 'parameterized', 'testfixtures', 'coverage']
DEVELOP_REQUIRES = ['autopep8', 'isort', 'mypy', 'pylint', 'sphinx', 'sphinx_rtd_theme', 'tox']

setup(name='tgutils',
      version=generate_version(),
      description='Common utilities used by the Tanay Group Python lab code.',
      long_description=readme(),
      long_description_content_type='text/x-rst',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Programming Language :: Python :: 3.7',
          'Topic :: Scientific/Engineering :: Bio-Informatics',
          'Intended Audience :: Developers',
      ],
      keywords='pandas numpy bioinformatics',
      url='https://bitbucket.org/orenbenkiki/tgutils',
      author='Oren Ben-Kiki',
      author_email='oren@ben-kiki.org',
      license='MIT',
      packages=find_packages(exclude=['tests']) + [
          'tgutils.numpy_stubs.numpy',
          'tgutils.numpy_stubs.pandas',
          'tgutils.numpy_stubs.pandas.core',
      ],
      package_data={'tgutils': ['py.typed']},
      entry_points={'console_scripts': [
          'tg_qsub=tgutils.tg_qsub:main',
      ]},
      # TODO: Replicated in tox.ini
      install_requires=INSTALL_REQUIRES,
      tests_require=TESTS_REQUIRE,
      test_suite='nose.collector',
      extras_require={  # TODO: Is this the proper way of expressing these dependencies?
          'develop': INSTALL_REQUIRES + TESTS_REQUIRE + DEVELOP_REQUIRES
      },
      cmdclass={
          # TODO: Add coverage command (if it is possible to get it to work).
          'all': AllCommand,
          'clean': CleanCommand,
          'html': HtmlCommand,
          'is_formatted': IsformattedCommand,
          'mypy': MypyCommand,
          'nose': NoseCommand,
          'pylint': PylintCommand,
          'reformat': ReformatCommand,
          'no_todo' + 'x': NoTodo_xCommand,
          'no_unknown_files': NoUnknownFilesCommand,
          'tox': ToxCommand,
      })
