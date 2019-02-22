"""House various file readers for mercury."""

# internal modules

# external modules
import numpy as np

# relative modules


def parse_aei(fname):
    """Parse .aei files intelligently."""
    """returns objectname and data in a tuple."""
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

# end of file
