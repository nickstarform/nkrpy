"""Controls handling of the models."""

# internal modules

# external libs
from numpy import array

# import relative module
from ...misc.functions import find_nearest as nearest

# global attributes
__all__ = ('kappa', 'kappa_models', 'kappascale')
__filename__ = __file__.split('/')[-1].strip('.py')
__path__ = __file__.strip('.py').strip(__filename__)
kappa_models = ('oh1994',)


def _find_model(model_name):
    """Given the model name, finds the model."""
    assert model_name in kappa_models,\
        f"Model is not found in:{kappa_models}"
    model_path = f'{__path__}/models/{model_name}.tsb'  # noqa
    return model_path


def _load_model(model_name):
    """Given model, load the model."""
    model_path = _find_model(model_name)
    model = []
    count = 0
    todo = False
    with open(model_path, 'r') as model_f:
        for i, line in enumerate(model_f):
            line = line.strip('\n').strip(' ')
            # print(f'{i}...<{line}>')
            if r'---------;' in line:
                count = i
                todo = True
            elif (i > count) and todo:
                # print(line)
                if len(line) > 0:
                    model.append(array([float(j) for j in line.split(';')
                                        if j != '']))
            elif not todo:
                header_col = line
    return header_col, array(model)


def kappascale(wav0, wav1, kappa0, beta):
    """Scale closest opacities."""
    return (wav0 / wav1) ** beta * kappa0


def kappa(wav, model_name='oh1994', density=0, beta=1.7, quiet=True):
    """Dust opacity."""
    """ from Ossenkopf & Henning 1994
    kappa_lambda @ 1.3mm
     = 2.3 cm^2/g for protoplanetary disks
     = 1.1 cm^2/g for very dense
     = 0.899cm^2/g dense cores
     = 0.8cm^2/g low dense cores
    beta=1.7 good for s
    """

    # load the model
    if model_name:
        h, model_l = _load_model(model_name)

    model_l = model_l[model_l[:, 0] == density, :]
    found = model_l[nearest(model_l[:, 1], wav)[0], :]
    print(found)
    if not quiet:
        print(f'Closest Kappa Model:\nColumn: {h}\nValues: {found}')
    _ret = tuple([kappascale(found[1], wav, x, beta)
                  for x in found[2:len(found)]])
    return _ret

# end of file
