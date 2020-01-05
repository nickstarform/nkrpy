"""House various file readers for mercury."""

# internal modules

# external modules
import numpy as np

# relative modules
from ...misc.functions import typecheck


def parse_aei(fname):
    """Parse .aei files intelligently.

    returns objectname and data in a tuple.
    """
    def not_found(line):
        """Parse out blank lines."""
        """iterating through lines and pulls first non blank line
        as the object name
        """
        line = line.replace(' ', '')
        if line != '':
            return False, line
        else:
            return True, None
    count = 0
    header = ''
    data = []
    name = ''
    with open(fname, 'r', errors='ignore') as f:
        for row in f:
            row = row.replace('\n', '')
            if not not_found(row)[0] and not name:
                name = row.replace(' (', '')\
                          .replace(')', '')\
                          .strip(' ').strip('\n').split(' ')[0]
            # get header
            elif not not_found(row)[0] and not header:
                header = [x.strip(' ') for x in row.replace(' (', '')
                                                   .replace(')', '')
                                                   .strip(' ')
                                                   .strip('\n')
                                                   .split(' ')
                          if x.strip(' ') != '']
            # get blank line
            elif not_found(row)[0]:
                count += 1
            elif count == 2:
                row = row.lower().replace('infinity', '1E81')
                _t = np.array([np.float(x) for x in row.split(' ')
                               if x.strip(' ') != ''])
                data.append(_t)
    toret = np.array(data)
    return name, header, toret


def parse_in(fname, ftype='param', params=None):
    """Parse Input Files."""
    f = open(fname, 'r')
    ret = []
    for line in f:
        if ')' == line[0]:
            continue
        if ftype == 'param':
            temp = [x.strip(' ') for x in line.split('=')]
        elif ftype == 'body':
            if '=' not in line:
                continue
            t = line.split(' ')
            d = []
            d.append(t[0])
            for p in t[1:]:
                k, v = p.split('=')
                if typecheck(params):
                    if k.lower() not in params:
                        continue
                d.append(v)
            ret.append(d)
            continue
        temp[0] = '_'.join(temp[0].split(' '))
        if typecheck(params):
            if temp[0].lower() not in params:
                continue
        ret.append(temp)
    return ret

# end of file
