def _dist_patch():
    import pkg_resources
    ep = pkg_resources.EntryPoint.parse('dummy = xlrd:__name__')
    ep.dist = d = pkg_resources.Distribution('')
    d._ep_map = {'pandalone.xleash.plugins': {'dummy': ep}}
    pkg_resources.working_set.add(d, 'dummy')


if __name__ == '__main__':
    _dist_patch()
    from co2mpas.cli import cli

    cli()
