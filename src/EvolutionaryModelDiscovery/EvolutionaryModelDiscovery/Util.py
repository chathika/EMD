'''EvolutionaryModelDiscovery: Automated agent rule generation and 
importance evaluation for agent-based models with Genetic Programming.
Copyright (C) 2018  Chathika Gunaratne
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.
You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.'''

from typing import List
import re
import pkg_resources
import functools
import inspect
import warnings
from pathlib import Path
import shutil

def netlogo_EMD_line_to_array(netlogo_EMD_line):
    assert '@emd' in netlogo_EMD_line.lower(), f'Not an EMD annotated line: {netlogo_EMD_line}'
    return re.sub('[\s;]','',netlogo_EMD_line).split('@')

def get_model_factors_module_name() -> str:
    return f'ModelFactors'

def get_model_factors_path() -> str:
    fpath = f'{get_model_factors_module_name()}.py'
    model_factors_file_path = pkg_resources.resource_filename('EvolutionaryModelDiscovery', fpath)    
    return str(model_factors_file_path)

def get_lock_fpath() -> str:
    lock_fpath = pkg_resources.resource_filename('EvolutionaryModelDiscovery', '.lock')    
    return lock_fpath

def remove_model_factors_file():
    Path(get_model_factors_path()).unlink()

def create_model_factors_file():
    if Path.exists(Path(get_model_factors_path())):
        open(get_model_factors_path(),'w+')

def slugify(value):
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    value = str(re.sub('[^\w\s-]', '', value).strip().lower())
    value = str(re.sub('[-\s]+', '_', value).lower())
    return value

def purge(dir : str, pattern : str) -> None:
    for f in Path(dir).iterdir():
        if re.search(pattern, str(f)):
            try:
                Path(dir, f).unlink()
            except FileNotFoundError:
                pass

def clear_cache() -> None:
    try:
        path = Path('.cache')
        for child in path.iterdir():
            if child.is_file():
                child.unlink()
            else:
                shutil.rm_tree(child)
        path.rmdir()
    except Exception:
        pass

def remove_model(model_path : str):
    Path(model_path).unlink()

def is_locked(filepath : str) -> bool:
    """
    Checks if a file is locked by opening it in append mode.
    If no exception thrown, then the file is not locked.
    """
    locked = None
    file_object = None
    try:
        buffer_size = 8
        # Opening file in append mode and read the first 8 characters.
        file_object = open(filepath, 'a', buffer_size)
        if file_object:
            locked = False
    except IOError as message:
        raise Exception('(unable to open in append mode).{0}.'.format(message))
        locked = True
    finally:
        if file_object:
            file_object.close()
    return locked

def wait_for_files(filepaths : List[str]) -> None:
    """
    Checks if the files are ready.

    For a file to be ready it must exist and can be opened in append
    mode.
    """
    wait_time = 0.05
    for filepath in filepaths:
        # If the file doesn't exist, wait wait_time seconds and try again
        # until it's found.
        # If the file exists but locked, wait wait_time seconds and check
        # again until it's no longer locked by another process.
        while is_locked(filepath):
            time.sleep(wait_time)


string_types = (type(b''), type(u''))


def deprecated(reason):
    """
    This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used.
    """

    if isinstance(reason, string_types):

        # The @deprecated is used with a 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated("please, use another function")
        #    def old_function(x, y):
        #      pass

        def decorator(func1):

            if inspect.isclass(func1):
                fmt1 = 'Call to deprecated class {name} ({reason}).'
            else:
                fmt1 = 'Call to deprecated function {name} ({reason}).'

            @functools.wraps(func1)
            def new_func1(*args, **kwargs):
                warnings.simplefilter('always', DeprecationWarning)
                warnings.warn(
                    fmt1.format(name=func1.__name__, reason=reason),
                    category=DeprecationWarning,
                    stacklevel=2
                )
                warnings.simplefilter('default', DeprecationWarning)
                return func1(*args, **kwargs)

            return new_func1

        return decorator

    elif inspect.isclass(reason) or inspect.isfunction(reason):

        # The @deprecated is used without any 'reason'.
        #
        # .. code-block:: python
        #
        #    @deprecated
        #    def old_function(x, y):
        #      pass

        func2 = reason

        if inspect.isclass(func2):
            fmt2 = 'Call to deprecated class {name}.'
        else:
            fmt2 = 'Call to deprecated function {name}.'

        @functools.wraps(func2)
        def new_func2(*args, **kwargs):
            warnings.simplefilter('always', DeprecationWarning)
            warnings.warn(
                fmt2.format(name=func2.__name__),
                category=DeprecationWarning,
                stacklevel=2
            )
            warnings.simplefilter('default', DeprecationWarning)
            return func2(*args, **kwargs)

        return new_func2

    else:
        raise TypeError(repr(type(reason)))
