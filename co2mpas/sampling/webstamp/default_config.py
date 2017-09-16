# The default_config module automatically gets imported by Appconfig, if it
# exists. See https://pypi.python.org/pypi/flask-appconfig for details.
#
## Copy them below in your i.e. `local_config.py`.
import logging

## Copy them below in your i.e. `local_config.py`.

## Note: Don't *ever* do this in a real app. A secret key should not have a
#       default, rather the app should fail if it is missing. For the sample
#       application, one is provided for convenience.
#SECRET_KEY = 'devkey'
#WTF_CSRF_ENABLED = True

#DREPORT_CC_DEFAULT = 'foo@bar; \n'
#STAMP_RECIPIENTS_DEFAULT = ''

MIN_DREPORT_SIZE = 1000
MAILIST_WIDGET_NROWS = 2
DREPORT_WIDGET_NROWS = 17
## Can be a number of head/tail lines to log, or a boolean.
CLIENT_VALIDATION_LOG_FULL_DREPORT = 600
CLIENT_VALIDATION_LOG_LEVEL = logging.DEBUG