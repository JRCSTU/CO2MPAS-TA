# -*- coding: utf-8 -*-
# !/usr/bin/env python
#
# Copyright 2014-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
#
## The default_config module automatically gets imported by Appconfig, if it
#  exists. See https://pypi.python.org/pypi/flask-appconfig for details.
#
#  Copy them below in your i.e. `local_config.py`.
import logging

## For DOS, limit requests to:
#  ~= x2 of (~4k each dreport + x2 for session + 10k cookies)
MAX_CONTENT_LENGTH = 50 * 1024

## Note: Don't *ever* do this in a real app. A secret key should not have a
#       default, rather the app should fail if it is missing. For the sample
#       application, one is provided for convenience.
#SECRET_KEY = 'devkey'

## If DSN is missing, dont install Sentry error & log handler.
# SENTRY_DSN = 'see https://sentry.io/'
## If set, log-records of this level create Sentry-events.
#  if not set, logging does not trigger events.
# SENTRY_LOG_LEVEL = logging.FATAL

# WTF_CSRF_ENABLED = True
# BOOTSTRAP_SERVE_LOCAL = True

#DEFAULT_STAMP_RECIPIENTS = []

## Preserve CPU by avoiding sign-verification on
#  obviously small dice-reports.
MIN_DREPORT_SIZE = 1200

MAILIST_WIDGET_NROWS = 2
DREPORT_WIDGET_NROWS = 13
## Can be a number of head/tail lines to log, or a boolean.
CLIENT_VALIDATION_LOG_FULL_DREPORT = 600

## How to report "soft" user validation-errors.
#  NOTE: maybe FATAL configured to send emails on Production??
CLIENT_VALIDATION_LOG_LEVEL = logging.DEBUG

## An optional command that returns 0 if signing-key
#  is in good shape.
#CHECK_SIGNING_KEY_SCRIPT = 'check_key'

## A cmdline as list of args for sending out emails.
#  - recipients will be added at the end of the list
#  - dice-report text will be given in STDIN.
#
#MAIL_CLI_ARGS = [
#    'mail',
#    '-n',                   # ignore `/etc/mail.rc`
#    '-v',                   # verbose and/or request mail-delivery response
#    '-r', 'stamper@bar',    # The `From:` address
#    '-s', '{subject}',
#    '--'
#]

## Sample traitlet-configs:
#
#TRAITLETS_CONFIG = {
#    'TsignerService': {
#        'stamper_name': <name>,
#    },
#    'StamperAuthSpec': {
#        'master_key': '',
#    },
#    'SigChain': {
#        'read_only_files': True,
#        ## if not set, Auto-default based on stamper-name.
#        #'stamp_chain_dir': '',
#    },
#}
