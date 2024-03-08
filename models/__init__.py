from os.path import dirname, basename, isfile, join
import glob
import importlib

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
__models__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py') or f.endswith('dataset.py')]
__name__ = "models"


for module_name in __all__:
    module = importlib.import_module(f"{__name__}.{module_name}")
    globals()[module_name] = module