# -*- coding: utf-8 -*-
# !/usr/bin/env python
#
# Copyright 2014-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
#
## From Flask-Bootstrap sample application. This will give you a
#  guided tour around creating an application using Flask-Bootstrap.
#
#  1. To run this application yourself, please install its requirements first:
#
#      $ pip install -r sample_app/requirements.txt
#
#  2. Copy and adapt `default_config.py --> local_config.py` in this package
#     (or in some other path).
#  3. Run the application
#     (and optionally use an absolute path for `WEBSTAMP_CONFIG` envvar):
#
#      $ WEBSTAMP_CONFIG=local_config.py flask --app=sample_app dev
#
#  Afterwards, point your browser to http://localhost:5000, then check out the
#  source.

from co2mpas.__main__ import init_logging
import logging
import os
import sys

from flask import Flask, request
from flask_appconfig import AppConfig
from flask_bootstrap import Bootstrap

import os.path as osp
from raven.contrib.flask import Sentry
import subprocess as sbp


def _get_git_version(log):
    try:
        mydir = osp.dirname(__file__)
        cmd = ['git', '-C', mydir, 'describe', '--long']
        v = sbp.check_output(cmd, universal_newlines=True)
        v = v and v.strip()
        log.info("\n%s\n@@ Starting Flask-app version: %r", '@' * 80, v)

        if v:
            return v

    except Exception as ex:
        log.warning("Failed to get app-version due to: %s", ex)


## NOTE: `configfile` DEPRECATED by `flask-appconfig` in latest dev.
#  Use `<APPNAME>_CONFIG` envvar instead.
#  See https://github.com/mbr/flask-appconfig/blob/master/flask_appconfig/__init__.py#L35
#
def create_app(configfile=None, logconf_file=None):
    from .frontend import frontend

    # We are using the "Application Factory"-pattern here, which is described
    # in detail inside the Flask docs:
    # http://flask.pocoo.org/docs/patterns/appfactories/

    ## log-configuration must come before Flask-config.
    #
    os.environ.get('%s_LOGCONF_FILE' % __name__)
    init_logging(logconf_file=logconf_file,
                 not_using_numpy=True)

    app = Flask(__name__)#, instance_relative_config=True)

    AppConfig(app, configfile)
    app.config['GIT_VERSION'] = _get_git_version(logging.getLogger(__name__))

    ## Automatically discover DSN key:
    #  https://docs.sentry.io/clients/python/integrations/flask/#setup
    #
    if 'SENTRY_DSN' in app.config:
        sentry = Sentry()
        sentry_log_level = app.config.get('SENTRY_LOG_LEVEL')
        sentry.init_app(app,
                        logging=bool(sentry_log_level),
                        level=sentry_log_level)

    # Install our Bootstrap extension
    Bootstrap(app)

    # Our application uses blueprints as well; these go well with the
    # application factory. We already imported the blueprint, now we just need
    # to register it:
    app.register_blueprint(frontend)

    # Because we're security-conscious developers, we also hard-code disabling
    # the CDN support (this might become a default in later versions):
    #app.config['BOOTSTRAP_SERVE_LOCAL'] = True

    return app


## From http://flask.pocoo.org/docs/dev/logging/#injecting-request-information
#
#  .. Warning::
#      Assign it to a logger used only from within a Request.
#
class RequestFormatter(logging.Formatter):
    def format(self, record):
        ## Form more request-data, see:
        #  http://flask.pocoo.org/docs/0.12/api/#incoming-request-data
        record.url = request.url
        record.remote_addr = request.remote_addr
        record.headers = request.headers
        record.cookies = request.cookies
        return super().format(record)


## From latest flask-0.12.2+:
#  https://github.com/pallets/flask/blob/master/flask/logging.py#L12
#
def wsgi_errors_stream():
    return request.environ['wsgi.errors'] if request else sys.stderr


if __name__ == '__main__':
    if __package__ is None:
        __package__ = 'webstamp'  # @ReservedAssignment
    from flask_appconfig import cli

    ## Set WEBSTAMP_CONFIG=local_config.py
    cmd = '--app=webstamp dev'.split()
    #cli.cli(ctx, app_name, configfile, env, amend_path)
    cli.cli(cmd)