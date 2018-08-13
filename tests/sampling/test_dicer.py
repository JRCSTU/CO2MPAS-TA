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

"""
from co2mpas.__main__ import init_logging
from co2mpas._vendor.traitlets import config as trtc
from co2mpas.sampling import CmdException, crypto
from co2mpas.sampling.base import PFiles
from co2mpas.sampling.project import DicerSpec, ProjectsDB
from tests.sampling import (
    test_inp_fpath, test_out_fpath,
    test_pgp_fingerprint, test_pgp_keys, test_pgp_trust,
)
import logging
import pytest

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

    return cfg


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
def dicerspec(traitcfg):
    return DicerSpec(config=traitcfg)


def test_dicer_new(dicerspec):
    pfiles = PFiles([test_inp_fpath], [test_out_fpath],
                    [osp.join(osp.dirname(__file__), '__init__.py')])
    dicerspec.do_dice_in_one_step(pfiles)


def test_dicer_rerun(dicerspec):
    pdb = ProjectsDB.instance(config=traitcfg)  # @UndefinedVariable
    cid = pdb.repo.head.commit.binsha

    pfiles = PFiles([test_inp_fpath], [test_out_fpath],
                    [osp.join(osp.dirname(__file__), '__init__.py')])

    ## State: decided
    #
    with pytest.raises(CmdException,
                       message="to forbidden state-transition from 'nosample'!"):
        dicerspec.do_dice_in_one_step(pfiles)
    assert pdb.current_project().state in ('sample', 'nosample')
    assert pdb.repo.head.commit.binsha == cid

    reset_git(pdb, 'HEAD~')
    #
    ## Start-state: tagged
    #
    pfiles = PFiles([test_inp_fpath], [test_out_fpath],
                    [osp.join(osp.dirname(__file__), '__init__.py')])
    dicerspec.do_dice_in_one_step(pfiles)
    assert pdb.current_project().state in ('sample', 'nosample')
    assert pdb.repo.head.commit.binsha != cid

    reset_git(pdb, 'HEAD~~')
    #
    ## Start-state: wltp_iof
    #
    pfiles = PFiles([test_inp_fpath], [test_out_fpath],
                    [osp.join(osp.dirname(__file__), '__init__.py')])
    dicerspec.do_dice_in_one_step(pfiles)
    assert pdb.current_project().state in ('sample', 'nosample')
    assert pdb.repo.head.commit.binsha != cid
