# This contains our frontend; since it is a bit messy to use the @app.route
# decorator style when using application factories, all of our routes are
# inside blueprints. This is the front-facing blueprint.
#
# You can find out more about blueprints at
# http://flask.pocoo.org/docs/blueprints/

from co2dice._vendor.traitlets import config as traitc
from co2dice import crypto, tsigner
import os
import re

from flask import request
import flask
from validate_email import validate_email
import werkzeug.exceptions

import subprocess as sbp
import textwrap as tw

from . import forms, get_bool_arg


class StampingKeyMissing(werkzeug.exceptions.HTTPException):
    code = 503
    description = ("Stamping-key temporarily unavailable! "
                   "JRC has been notified, please try again later.")


def unique_ci(words):
    """
    Eliminate all but 1st duplicate words, case-insensitively.

    >>> words = 'a big A pig In JaPaN in japan JAPAN'.split()
    >>> unique_ci(words)
    ['a', 'big', 'pig', 'In', 'JaPaN']
    """
    lower_words = set()
    uniques = []
    for w in words:
        lw = w.lower()
        if lw not in lower_words:
            lower_words.add(lw)
            uniques.append(w)
    return uniques


## TODO: ESCAPE USER-INPUT!!!!
def recipients_str(recipients):
    return '; '.join(recipients)


frontend = flask.Blueprint('frontend', __name__)


## NOTE: DISABLE LANDING PAGE to avoid spam.
# # Our index-page just shows a quick explanation. Check out the template
# # "templates/index.html" documentation for more details.
# @frontend.route('/')
# def index():
#     return flask.render_template('index.html')

## As a method, for blueprint to access `app` & `config`.
#  See https://stackoverflow.com/a/23037071/548792
#
@frontend.record
def attach_routes(setup_state):
    app = setup_state.app
    log = app.logger
    log.propagate = True  # By default, `False`!!!
    config = app.config

    def _remote_sender(request, sender, max_width):
        """
        Find request's remote address, even if proxied (return the full list).

        :return:
            a string with `sender` and a single or CS client IPs, optionally
            limited by `max_width` with ellipsis.

        Taken From:
          https://stackoverflow.com/questions/12770950/flask-request-remote-addr-is-wrong-on-webfaction-and-not-showing-real-user-ip
        Read also the linked article:
          http://esd.io/blog/flask-apps-heroku-real-ip-spoofing.html
        """
        headers_list = request.headers.getlist("X-Forwarded-For")
        client_ip = str(headers_list) if headers_list else request.remote_addr
        sender = '(%s) %s' % (sender, client_ip)
        sender_len = len(sender)

        if max_width and sender_len > max_width:
            log.info("Dice has a shortened remote-IP: %s", sender)

            if max_width > 3:
                    sender = tw.shorten(sender, max_width)
            else:
                sender = ''

        return sender

    def check_key_exists():
        """Key not hosted locally, maybe missing due to connectivity."""
        check_key_script = config.get('CHECK_SIGNING_KEY_SCRIPT')
        if check_key_script:
            p = sbp.run(check_key_script, stdout=sbp.PIPE, stderr=sbp.PIPE)
            if p.returncode != 0:
                log.fatal("Stamper-key missing!  retcode(%s)"
                          "\n  stdout: %s\n  stderr: %s",
                          p.returncode, p.stdout, p.stderr)
                raise StampingKeyMissing()

    class Stamper:

        @property
        def traits_config(self):
            """Convert Flask-config --> trait-config, respecting URL-args. """
            tconf = getattr(request, 'traitlets_config', None)
            if tconf is None:
                tconf = traitc.Config(config['TRAITLETS_CONFIG'])

                ## Allow override only if not in trait-configs.
                #
                if 'allow_test_key' not in tconf.GpgSpec:
                    flag = get_bool_arg('allow_test_key')
                    if flag is not None:
                        tconf.GpgSpec.allow_test_key = flag

                flag = get_bool_arg('validate_decision')
                if flag is not None:
                    tconf.TsignerService.validate_decision = flag

                flag = get_bool_arg('trim_dreport')
                if flag is not None:
                    tconf.TsignerService.trim_dreport = flag

                request.traitlets_config = tconf

                ## Update singletons to be used that may have been
                #  already created from previous requests!!
                #
                #  NOTE: singletons were designed for CLI one-off commands.
                #
                crypto.GitAuthSpec.clear_instance()      # @UndefinedVariable
                crypto.StamperAuthSpec.clear_instance()  # @UndefinedVariable

            return tconf

        @property
        def allow_test_key(self):
            return self.traits_config.GpgSpec.allow_test_key

        @property
        def signer(self) -> tsigner.TsignerService:
            signer = getattr(request, 'signer', None)
            if signer is None:
                request.signer = signer = tsigner.TsignerService(
                    config=self.traits_config)

            return signer

        def check_dice(self, dice: str, sender: str, recipients: str):
            verdict = self.signer.parse_signed_tag(dice)
            uid = crypto.uid_from_verdict(verdict)

            # TODO: move sig-validation check in `crypto` module.
            if not verdict['valid']:
                raise ValueError(
                    "Cannot validate dice signed with %r: %s" %
                    (uid, verdict['status']))

            ## Check if test-key still used.
            #
            git_auth = crypto.get_git_auth(config=self.traits_config)
            git_auth.check_test_key_missused(verdict['key_id'])

            # TODO: API-validate `sender` & `recipients`.

            return uid

        def sign_dreport(self, dreport: str, sender: str, recipients: str):
            """
            :return:
                tuple(dice_stamp, dice_decision)
            """
            signer = self.signer
            signer.recipients = recipients
            ## Calculate remaining width for IP in parenthesis of::
            #
            #     #    (123.456.123.456) user@home.gr
            max_line_width = 60  # See :meth:`TsignerService.sign_text_as_tstamper()`
            avail_width = max_line_width - len('#    ()') - len(sender)
            sender = _remote_sender(request, sender, avail_width)
            return signer.sign_dreport_as_tstamper(sender,
                                                   dreport)

        def parse_recipients_text(self, text):
            skip_check_mx = get_bool_arg('skip_mail_mx_check', os.name == 'nt')

            recipients = re.split(r'[\s,;]+', text)
            recipients = [s and s.strip() for s in recipients]
            recipients = list(filter(None, recipients))

            ## Prepend "hidden" default-recipients. (JRC & CLIMA?).
            #
            default_recipients = config.get('DEFAULT_STAMP_RECIPIENTS', [])
            recipients = unique_ci(default_recipients + recipients)

            if len(recipients) < len(default_recipients) + 1:
                raise ValueError(
                    'Specify at least 1 extra recipient! Got: %s' %
                    recipients_str(recipients))

            for i, email in enumerate(recipients, 1):
                if not validate_email(email, check_mx=not skip_check_mx):
                    raise ValueError(
                        'Invalid email-address no-%i: `%s`' %
                        (i, recipients_str(recipients)))

            return recipients

    StampForm = forms.create_stamp_form_class(app, Stamper)

    ## FIXME: Make URLs it configurable to avoid DoS!
    @frontend.route('/stamp/', methods=('GET', 'POST'))
    def stamp_with_form():
        log.info("WebStamp URL: %s\n  values: %s",
                 request.url, [str(v)[:1400] for v in request.values.items()])
        check_key_exists()
        try:
            return StampForm().render()
        except Exception as ex:
            log.fatal('WebStamp crashed due to: %s\n  %s',
                      ex, [str(v)[:1400] for v in request.values.items()], exc_info=1)
            raise

    ## FIXME: Make URLs it configurable to avoid DoS!
    @frontend.route('/api-stamp/', methods=('POST', ))
    def stamp():
        check_key_exists()
        stamper = Stamper()
        params = request.values
        dice_stamp, _dice_decision = stamper.sign_dreport(
            params['dice_report'],
            params['sender'],
            stamper.parse_recipients_text(params['recipients']),
        )

        return dice_stamp

    ## FIXME: Make URLs it configurable to avoid DoS!
    @frontend.route('/api-check/', methods=('POST', ))
    def check():
        check_key_exists()
        stamper = Stamper()
        params = request.values

        if 'dice_report' in params:
            ## Without dice, just connectivity and signing-key checked.

            stamper.check_dice(
                params['dice_report'],
                params['sender'],
                stamper.parse_recipients_text(params['recipients']),
            )

        return 'ok'
