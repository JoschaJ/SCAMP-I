from setuptools import setup

setup(name='scampi',
      version='1.0.3',
      url='https://github.com/pulsarise/SCAMP-I',
      author='Lucy Oswald, Marisa Geyer',
      description='Fit scattering tails with MCMC.',
      packages=['scampi'],
      install_requires=['numpy', 'scipy', 'pandas', 'emcee', 'argparse', 'matplotlib', 'corner'],
      scripts=['bin/alpha_mcmc.py', 'bin/basic_plotting.py', 'bin/best_fit_alpha_tau.py', 'bin/check_alpha_chains.py',
               'bin/check_tau_chains.py', 'bin/create_config_ascii_from_archive.py', 'bin/define_alpha_burnin.py',
               'bin/define_tau_passfail_burnin.py', 'bin/parameter_extraction.py', 'bin/run_scatter_mcmc.py']
)