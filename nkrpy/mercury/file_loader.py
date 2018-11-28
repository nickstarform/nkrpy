"""House various file readers for mercury."""

# internal modules

# external modules
import numpy as np

# relative modules


def parse_aei(fname):
    """Parse .aei files intelligently."""
    """returns objectname and data in a tuple."""
    def object_name(fname):
        """Discern name of object from header."""
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
        toret = ''
        with open(fname, 'r', error='ignore') as f:
            for row in f:
                row = row.replace('\n', '').replace(' ', '')
                if not not_found(row)[0] and not toret:
                    toret = row.replace(' ', '')
                elif not not_found(row)[0]:
                    break
                count += 1
        return count, toret

    def get_header(fname, num=3):
        with open(fname, 'r', error='ignore') as f:
            for i, row in enumerate(f):
                if i == num:
                    return [x for x in row.replace(' (', '')
                                          .replace(')', '')
                                          .strip(' ').strip('\n')
                                          .split(' ') if x != '']
    headerstart, oname = object_name(fname)
    file = np.loadtxt(fname, skiprows=headerstart + 2, dtype=float)
    return oname, get_header(fname, headerstart + 1), file

# end of file
