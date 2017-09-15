# The default_config module automatically gets imported by Appconfig, if it
# exists. See https://pypi.python.org/pypi/flask-appconfig for details.
#
## Add the following in i.e. your `local_config.py`.

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
