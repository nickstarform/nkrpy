"""Misc Common Functions."""

# internal modules
import inspect

# external modules
import numpy as np

# relative modules
from .misc import Format

# global attributes
__all__ = ['help', 'help_api']
__doc__ = """Generalized help function."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
__LINEWIDTH__ = 79
primative = (int, dict, type, set, float, list, tuple, str)
sigignore = "(*args, **kwargs)"


def resolve_name(cls):
    if '__name__' in dir(cls):
        return cls.__name__
    if '__class__' in dir(cls):
        return resolve_name(cls.__class__)

def resolve_doc(cls):
    """Resolve class documentation."""
    doc = ''
    if f'__doc__' in dir(cls):
        if cls.__doc__ is None:
            d = ''
        else:
            d = cls.__doc__
        doc += d
    for d in ['init', 'call', 'new']:
        if f'__{d}__' in dir(cls):
            attr = getattr(cls, f'__{d}__')
            sig = str(inspect.signature(attr)).replace(' ', '').replace('(self,', '(').split(',')
            sig = ', '.join(sig)
            sig = sig.replace(':', ': ').replace('=', ' = ')
            if sig == sigignore:
                continue
            doc += f'''
    {d}{sig}
    {"-" * (len(d) + len(sig))}
        {attr.__doc__}
            '''
    return doc

def is_scalar(cls):
    d = dir(cls)
    if not is_func(cls) and not is_class(cls) and not is_module(cls):
        return True
    return False


def is_func(cls):
    d = dir(cls)
    if callable(cls) or '__func__' in d:
        return True
    return False


def is_class(cls):
    d = dir(cls)
    if not is_func(cls) and ('__class__' in d or '__module__' in d):
        return True
    return False


def is_module(cls):
    d = dir(cls)
    if not is_func(cls) and '__package__' in d and '__loader__' in d:
        return True
    return False


def resolve_modules(cls, ret: dict, parentname: str, loaded: list, count: int):
    for d in dir(cls):
        count += 1
        if count > 20:
            break
        # skip hidden
        if d.startswith('_'):
            continue
        # gather attributes
        if d in loaded:
            continue
        print('Loading', parent, d)
        attr = getattr(cls, d)
        if isinstance(attr, primative):
            # this is the end
            if ('__bases__' not in dir(attr)) or ('__bases__' in dir(attr) and len(attr.__bases__) == 0):
                print(f'primative: {d}')
                ret[d] = d
                continue
        if is_module(attr):
            # walk through all sublevels
            loaded.append(d)
            parentname += f'.{d}'
            print(f'module: {parentname}')
            ret[parentname] = {}
            resolve_modules(attr, ret=ret, parentname=parentname, loaded=loaded, count=count)
            continue
        if is_class(attr):
            # class get init new call and all methods
            loaded.append(d)
            parentname += f'.{d}'
            print(f'class: {parentname}')
            if '__bases__' in dir(attr):
                base_classes = attr.__bases__
            else:
                base_classes = '()'
            doc = resolve_doc(attr)
            doc = f'''
    {base_classes}
    {"-" * len(base_classes)}
    {doc}
    '''
            ret[parentname] = doc
            continue
        if is_func(attr):
            print(f'func: {d}')
            try:
                ret[d] = str(inspect.signature(attr))
            except:
                ret[d] = resolve_doc(attr)
            continue
        if is_scalar(attr):
            print(f'scalar: {d}')
            ret[d] = attr
            continue
    return ret, loaded

# need to be completely redone
def help(func_or_class, colour: bool = True):
    ret = help_api(func_or_class, colour)
    print(ret)


def help_api(func_or_class, colour: bool = False):
    name = resolve_name(func_or_class).replace('.TYPE', '')
    mapping = []
    top_level_docs = func_or_class.__doc__ if func_or_class.__doc__ else ''
    top_level_mod = '' if '__module__' not in dir(func_or_class) else func_or_class.__module__



    name = (f'{top_level_mod}.{name}').upper()
    ret = ''
    if '__class__' not in dir(func_or_class):
        args = str(inspect.signature(func_or_class))
    else:
        dirsnames = [d for d in dir(func_or_class) if not d.startswith('_')]
        dirs = [getattr(func_or_class, d) for d in dirsnames]
        args = ''
        for i, d in enumerate(dirs):
            #print(dir(d))
            if '__name__' in dir(d):
                inner_name = d.__name__
            else:
                inner_name = dirsnames[i]
            inner_docs = d.__doc__ if d.__doc__ else ''
            if '__call__' in dir(d):
                inner_args = str(inspect.signature(d))
            else:
                inner_args = f'{d}'
            toplevel = f'{inner_name}: {inner_args}'
            args += toplevel + '\n'
            if len(inner_docs) > 20:
                args += ('-' * len(toplevel)) + '\n'
                args += inner_docs + '\n'
            else:
                args += f': {inner_docs}' + '\n'
    if colour:
        ret += Format('HEADER') + Format('BOLD')
    spacer = ' ' * int(__LINEWIDTH__ / 2 - len(name) / 2 - 1)
    ret += ('=' * __LINEWIDTH__) + '\n'
    ret += spacer + name + spacer + '\n'
    ret += ('=' * __LINEWIDTH__) + '\n'
    if colour:
        ret += Format('RESET') + Format('OKBLUE')
    ret += top_level_docs + '\n'
    ret += ('-' * __LINEWIDTH__) + '\n'
    if colour:
        ret += Format('WARNING')
    ret += args
    if colour:
        ret += Format('RESET')
    return ret


# end of code
