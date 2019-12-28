"""."""
# flake8: noqa

# internal modules

# external modules

# relative modules

# global attributes
__all__ = ('scientific_format',)
__doc__ = """."""
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)

def scientific_format(num, precision: int=3):
    form = '{:0.' + f'{precision}' + 'e}'
    num = form.format(num)
    ret = 'x10$^{'.join(num.split('e'))
    ret = f'{ret}' + '}$'
    return ret

# end of code

# end of file
