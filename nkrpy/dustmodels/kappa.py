"""Controls handling of the models."""

# external libs
from numpy import array

# import relative module
from ..miscmath import find_nearest as nearest

__all__ = ('kappa','__models__')

__models__ = ('oh1994',)

__path__ = __file__.strip('.py').strip(__name__)

def _find_model(model_name):
    """Given the model name, finds the model."""
    assert model_name in __models__, f"Model is not found in:{__models__}"
    model_path = f'{__path__}/{model_name}.tsb'  # noqa
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
                    model.append(array([float(j) for j in line.split(';') if j != '']))
    return array(model)


def kappa(wav, model_name='oh1994', beta=1.7):
    """Dust opacity."""
    """ from Ossenkopf & Henning 1994
    kappa_lambda @ 1.3mm
     = 2.3 cm^2/g for protoplanetary disks
     = 1.1 cm^2/g for very dense
     = 0.899cm^2/g dense cores
     = 0.8cm^2/g low dense cores
    beta=1.7 good for s
    """

    def scale(wav0, wav1, kappa0, beta):
        """Scale closest opacities."""
        return (wav0 / wav1) ** beta * kappa0

    # load the model
    if model_name:
        model_l = _load_model(model_name)

    found = model_l[nearest(model_l[:,1], wav)[0], :]
    _ret = tuple([scale(found[1], wav, x, beta) for x in found[2:len(found)]])

    return _ret

# end of file
