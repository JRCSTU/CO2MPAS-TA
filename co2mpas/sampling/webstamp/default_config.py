# The default_config module automatically gets imported by Appconfig, if it
# exists. See https://pypi.python.org/pypi/flask-appconfig for details.
#
# Copy them below in your i.e. `local_config.py`.
import logging

## Note: Don't *ever* do this in a real app. A secret key should not have a
#       default, rather the app should fail if it is missing. For the sample
#       application, one is provided for convenience.
#SECRET_KEY = 'devkey'
#WTF_CSRF_ENABLED = True
#BOOTSTRAP_SERVE_LOCAL = True

#DEFAULT_STAMP_RECIPIENTS = ''

MIN_DREPORT_SIZE = 1000
MAILIST_WIDGET_NROWS = 2
DREPORT_WIDGET_NROWS = 17
## Can be a number of head/tail lines to log, or a boolean.
CLIENT_VALIDATION_LOG_FULL_DREPORT = 600
CLIENT_VALIDATION_LOG_LEVEL = logging.DEBUG

# TRAITLETS_CONFIG = {
#     'StamperAuthSpec': {
#         'master_key': ''
#     }
# }
TRAITLETS_CONFIG = {}
