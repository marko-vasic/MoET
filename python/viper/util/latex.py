import re


# TODO: Rename
def create_latex_macro(key, value):
    return '\\DefMacro{{{0}}}{{{1}}}'.format(key, value)


def create_usage(key):
    return '\\UseMacro{{{0}}}'.format(key)


def unpack_latex_macro(macro):
    m = re.match('\\\\DefMacro{(?P<key>[^}]+)}{(?P<value>[^}]+)}',
                 macro)
    if m:
        return m.group('key'), m.group('value')
    else:
        return None, None
