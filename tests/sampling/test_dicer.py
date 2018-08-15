#! python
# -*- coding: UTF-8 -*-
#
# Copyright 2015-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
"""
If not running a WebStamper on your localhost, set these 2 env-vars::

    WEBSTAMPER_STAMP_URL
    WEBSTAMPER_CHECK_URL

- All :func:`..._B_...()`` TCs required :func:`..._A_...()` to have run.
"""
from co2mpas.__main__ import init_logging
from sampling._vendor.traitlets import config as trtc
from sampling import CmdException, crypto
from sampling.base import PFiles
from sampling.dicer import DicerSpec
from sampling.project import ProjectsDB
from tests.sampling import (
    test_pgp_fingerprint, test_pgp_keys, test_pgp_trust,
)
import logging

import pytest

from py.path import local   # @UnresolvedImport
import os.path as osp


mydir = osp.dirname(__file__)
init_logging(level=logging.INFO)
log = logging.getLogger(__name__)


def reset_git(pdb: ProjectsDB, ref):
    pdb.repo.git.reset(ref, hard=True)
    pdb._current_project = None
    p = pdb.current_project()

    return p


@pytest.fixture(scope='module')
def repodir(tmpdir_factory):
    repo = tmpdir_factory.mktemp('repo')
    log.debug('Temp Git-repo: %s', repo)
    return repo


@pytest.fixture(scope='module')
def gpgdir(tmpdir_factory):
    gpg = tmpdir_factory.mktemp('gpghome')
    log.debug('Temp GPG-home: %s', gpg)
    return gpg


@pytest.fixture()
def traitcfg(repodir, gpgdir):
    cfg = trtc.Config()

    cfg.GpgSpec.gnupghome = str(gpgdir)
    cfg.GpgSpec.keys_to_import = test_pgp_keys
    cfg.GpgSpec.trust_to_import = test_pgp_trust
    cfg.GpgSpec.master_key = test_pgp_fingerprint
    cfg.GpgSpec.allow_test_key = True
    cfg.DiceSpec.user_name = "Test Vase"
    cfg.DiceSpec.user_email = "test@vase.com"

    cfg.ProjectsDB.repo_path = str(repodir)
    cfg.Spec.verbose = cfg.ProjectsDB.verbose = 0
    cfg.WstampSpec.recipients = ["test@gel.bourdien.com"]

    cfg.Project.max_dices_per_project = 10  # due to resets

    return cfg


@pytest.fixture()
def pdb(traitcfg):
    return ProjectsDB.instance(config=traitcfg)  # @UndefinedVariable


def head_sha1(pdb):
    return pdb.repo.head.commit.hexsha[:10]


@pytest.fixture()
def cryptos(traitcfg):
    crypto.GpgSpec(config=traitcfg)

    ## Clean memories from past tests
    #
    crypto.StamperAuthSpec.clear_instance()  # @UndefinedVariable
    crypto.GitAuthSpec.clear_instance()      # @UndefinedVariable
    crypto.VaultSpec.clear_instance()        # @UndefinedVariable


@pytest.fixture()
@pytest.mark.usefixtures("cryptos")
def dicer(traitcfg):
    return DicerSpec(config=traitcfg)


@pytest.fixture()
def iofiles():
    from tests.sampling import test_inp_fpath, test_out_fpath

    ifile = local(test_inp_fpath)
    ofile = local(test_out_fpath)

    return ifile, ofile


@pytest.fixture()
def iofiles_mov(iofiles, tmpdir):
    "Same files served from different folders."
    ifile, ofile = iofiles
    ifile2 = tmpdir.mkdir('inp') / ifile.basename
    ifile.copy(ifile2)
    ofile2 = tmpdir.mkdir('out') / ofile.basename
    ofile.copy(ofile2)

    return ifile2, ofile2


@pytest.fixture()
def iofiles_diff(iofiles):
    "filenames, different content (& folders, how else?)."
    ifile, ofile = iofiles
    ifile2 = osp.join(mydir, 'diff_pfiles', ifile.basename)
    ofile2 = osp.join(mydir, 'diff_pfiles', ofile.basename)

    return ifile2, ofile2


other = osp.join(osp.dirname(__file__), '__init__.py')

#: Sentinel rest TCs will crash if not assigned by A below.
_decided_sha1 = None


def test_dicer_A_new(dicer, pdb, iofiles):
    global _decided_sha1

    ifile, ofile = iofiles
    pfiles = PFiles([ifile], [ofile], [other])
    dicer.do_dice_in_one_step(pfiles)
    assert pdb.current_project().state in ('sample', 'nosample')
    _decided_sha1 = head_sha1(pdb)


def test_dicer_B_fail_DECIDED(dicer, iofiles_mov, pdb):
    ifile, ofile = iofiles_mov
    pfiles = PFiles([ifile], [ofile], [other])

    ## State: 'decided' from A above

    with pytest.raises(
            CmdException,
            match="to forbidden state-transition from '(no)?sample'!"):
        dicer.do_dice_in_one_step(pfiles)
    assert pdb.current_project().state in ('sample', 'nosample')
    assert head_sha1(pdb) == _decided_sha1


def test_dicer_B_ok_TAGGED(dicer, iofiles_mov, pdb):
    reset_git(pdb, '%s~' % _decided_sha1)
    ifile, ofile = iofiles_mov
    pfiles = PFiles([ifile], [ofile], [other])
    dicer.do_dice_in_one_step(pfiles)
    assert pdb.current_project().state in ('sample', 'nosample')
    assert head_sha1(pdb) != _decided_sha1


def test_dicer_B_ok_WLTPIOF_relpaths(dicer, iofiles_mov, pdb):
    reset_git(pdb, '%s~~' % _decided_sha1)
    ifile, ofile = iofiles_mov

    ## Relative paths:
    #
    ifile.dirpath().chdir()
    pfiles = PFiles([ifile.basename], [local('../out') / ofile.basename], [other])
    dicer.do_dice_in_one_step(pfiles)
    assert pdb.current_project().state in ('sample', 'nosample')
    assert head_sha1(pdb) != _decided_sha1


def test_dicer_B_fail_DIFF_files(dicer, iofiles, iofiles_mov, iofiles_diff,
                                 pdb, tmpdir):
    reset_git(pdb, '%s~~' % _decided_sha1)

    ifile, ofile = iofiles
    okfiles = PFiles([ifile], [ofile], [other])

    ofileRen = tmpdir.mkdir('ren') / 'foo.xlsx'
    ofile.copy(ofileRen)

    ifile, ofile = iofiles_mov

    ## Different content `inp`
    pfiles = PFiles([iofiles_diff[0]], [ofile])
    with pytest.raises(CmdException,
                       match="^Project files missmatched"):
        dicer.do_dice_in_one_step(pfiles)

    ## Check if diff-check had leftovers.
    dicer.do_dice_in_one_step(okfiles)
    reset_git(pdb, '%s~~' % _decided_sha1)

    ## Different name 'out'
    pfiles = PFiles([ifile], [ofileRen])
    with pytest.raises(CmdException,
                       match="^Project files missmatched"):
        dicer.do_dice_in_one_step(pfiles)

    ## Check if diff-check had leftovers.
    dicer.do_dice_in_one_step(okfiles)
    reset_git(pdb, '%s~' % _decided_sha1)

    ## Different content `other`
    pfiles = PFiles([ifile], [ofile],
                    [osp.join(osp.dirname(__file__), '..', '__init__.py')])
    with pytest.raises(CmdException,
                       match="^Project files missmatched"):
        dicer.do_dice_in_one_step(pfiles)

    ## Less 'other' files
    pfiles = PFiles([ifile], [ofile])
    with pytest.raises(CmdException,
                       match="^Project files missmatched"):
        dicer.do_dice_in_one_step(pfiles)

    ## Check if diff-check had leftovers.
    dicer.do_dice_in_one_step(okfiles)
    reset_git(pdb, '%s~' % _decided_sha1)
