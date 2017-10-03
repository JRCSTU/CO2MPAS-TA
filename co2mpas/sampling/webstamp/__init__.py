# Welcome to the Flask-Bootstrap sample application. This will give you a
# guided tour around creating an application using Flask-Bootstrap.
#
# 1. To run this application yourself, please install its requirements first:
#
#     $ pip install -r sample_app/requirements.txt
#
# 2. Copy and adapt `default_config.py --> local_config.py` in this package
#    (or in some other path).
# 3. Run the application
#    (and optionally use an absolute path for `WEBSTAMP_CONFIG` envvar):
#
#     $ WEBSTAMP_CONFIG=local_config.py flask --app=sample_app dev
#
# Afterwards, point your browser to http://localhost:5000, then check out the
# source.

import logging
import os

from co2mpas.__main__ import init_logging
from flask import Flask, request
from flask_appconfig import AppConfig
from flask_bootstrap import Bootstrap

from .frontend import frontend


## NOTE: `configfile` DEPRECATED by `flask-appconfig` in latest dev.
#  Use `<APPNAME>_CONFIG` envvar instead.
#  See https://github.com/mbr/flask-appconfig/blob/master/flask_appconfig/__init__.py#L35
#
def create_app(configfile=None, logconf_file=None):
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
