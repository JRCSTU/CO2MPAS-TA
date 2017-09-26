# The default_config module automatically gets imported by Appconfig, if it
# exists. See https://pypi.python.org/pypi/flask-appconfig for details.
#
# Copy them below in your i.e. `local_config.py`.
import logging

## For DOS, limit requests to:
#  ~= x2 of (~4k each dreport + x2 for session + 10k cookies)
MAX_CONTENT_LENGTH = 50 * 1024

## Note: Don't *ever* do this in a real app. A secret key should not have a
#       default, rather the app should fail if it is missing. For the sample
#       application, one is provided for convenience.
#SECRET_KEY = 'devkey'

# WTF_CSRF_ENABLED = True
# BOOTSTRAP_SERVE_LOCAL = True

#DEFAULT_STAMP_RECIPIENTS = ''

## Preserve CPU by avoiding sign-verification on
#  obviously small dice-reports.
MIN_DREPORT_SIZE = 1200

MAILIST_WIDGET_NROWS = 2
DREPORT_WIDGET_NROWS = 12
## Can be a number of head/tail lines to log, or a boolean.
CLIENT_VALIDATION_LOG_FULL_DREPORT = 600
CLIENT_VALIDATION_LOG_LEVEL = logging.INFO

## Sample cmdline for sending out emails
#  (dice-report will be given in STDIN):
#
#MAIL_CLI_ARGS = ['mail',
#                 '-n',  # ignore `/etc/mail.rc`
#                 '-v'   # verbose and/or request mail-delivery response
#                 '-s', '{subject}',
#                 '--',
#                 '{recipients}']

## Sample traitlet-configs:
#
#TRAITLETS_CONFIG = {
#    'TsignerService': {
#        'stamper_name': <name>,
#        #'stamp_chain_folder': '', # Auto-default based on stamper-name.
#    },
#    'StamperAuthSpec': {
#        'master_key': '',
#    },
#}
