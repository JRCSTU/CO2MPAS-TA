# -*- coding: utf-8 -*-
# !/usr/bin/env python
#
# Copyright 2014-2017 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
from co2mpas._vendor.traitlets import config as traitc
from co2mpas.sampling import CmdException, crypto, tsigner, tstamp
import json
import logging
import os
import re
from typing import Text

from boltons.setutils import IndexedSet as iset
from flask import flash, request, session
import flask
from flask.ctx import after_this_request
from flask_wtf import FlaskForm
from markupsafe import escape, Markup
from validate_email import validate_email
import wtforms
import yaml

import pprint as pp
import subprocess as sbp
import textwrap as tw
import traceback as tb
import wtforms.fields as wtff
import wtforms.validators as wtfl
from tests.sampling.test_tsigner import default_recipients


STAMPED_PROJECTS_KEY = 'stamped_projects'
LAST_SENDER_KEY = 'last_sender'
LAST_RECIPIENTS_KEY = 'last_recipients'


logger = logging.getLogger(__name__)


def get_bool_arg(argname):
    """
    True is an arg alone, or any stripped string not one of: ``0|false|no|off``
    """
    args = request.args
    if argname in args:
        param = args[argname].strip()
        return param.lower() not in '0 false no off'.split()


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


def ascii_validator(_form, field):
    value = field.data
    if not all(ord(c) < 128 for c in value):
        raise wtfl.ValidationError("Field does not accept non-ASCII chars.")


def create_stamp_form_class(app):
    ## Prepare various config-dependent constants

    config = app.config
    client_validation_log_full_dreport = config['CLIENT_VALIDATION_LOG_FULL_DREPORT']
    client_validation_log_level = config['CLIENT_VALIDATION_LOG_LEVEL']
    try:
        client_validation_log_level = int(client_validation_log_level)
    except Exception as ex:
        logger.warning("Invalid config param "
                       "`CLIENT_VALIDATION_LOG_LEVEL: %s` due to: %s",
                       client_validation_log_level, ex, exc_info=1)
        client_validation_log_level = logging._nameToLevel.get(
            client_validation_log_level, logging.DEBUG)
    logger.info("Client-faults logged as: %s", client_validation_log_level)

    def get_json_item(adict, key, *, as_list):
        json_type = list if as_list else dict
        cookie_value = adict.get(key)
        if cookie_value:
            try:
                cookie_value = json.loads(cookie_value)
                if isinstance(cookie_value, json_type):
                    return cookie_value

                app.logger.info("Expected JSON(%s) to be %s, it was: %s"
                                "\n  COOKIE=%s",
                                key, json_type.__name__,
                                type(cookie_value), cookie_value)
            except json.JSONDecodeError as ex:
                app.logger.info("Corrupted JSON(%s) due to: %s"
                                "\n  \n  COOKIE==%s",
                                key, ex, cookie_value,
                                exc_info=1)

        return json_type()

    def _get_stamped_projects(cookies):
        return set(get_json_item(cookies, STAMPED_PROJECTS_KEY, as_list=True))

    def is_project_in_stamps_cookie(cookies, project):
        return project in _get_stamped_projects(cookies)

    def add_project_in_stamps_cookie(cookies, project):
        stamped_projects = _get_stamped_projects(cookies)
        stamped_projects.add(project)
        stamped_projects = list(sorted(set(stamped_projects)))

        @after_this_request
        def store_stamped_projects(response):
            response.set_cookie(STAMPED_PROJECTS_KEY, json.dumps(stamped_projects))
            return response

    class StampForm(FlaskForm):
        """
        Form submission boolean args:

        - ``allow_test_key``: allow CBBB52FF
        - ``validate_decision``: append decision
        - ``trim_dreport``: remove garbage suffix from dreport
        """

        _skeys = ['sender', 'stamp_recipients', 'dice_stamp',
                  'dice_decision', 'mail_err']

        sender = wtff.TextField(
            label='Sender:',
            description="""
                Your <em>designated-email</em> and/or your name or id
                (free form, but only ASCII).
            """,
            validators=[wtfl.InputRequired(),
                        wtfl.Length(min=3, max=60),
                        ascii_validator],
            default=lambda: request.cookies.get(LAST_SENDER_KEY),
            render_kw={'autocomplete': 'on'})

        stamp_recipients = wtff.TextAreaField(
            label='Stamp Recipients:',
            description="""
                <strong>
                  NOTE: the standard <em>CLIMA/JRC</em> recipients
                  are appended automatically.
                </strong><br>
                (separate email-addresses by <kbd>,</kbd>, <kbd>;</kbd>,
                <kbd>[Space]</kbd>, <kbd>[Enter]</kbd>, <kbd>[Tab]</kbd> characters)
            """,
            validators=[wtfl.InputRequired()],
            default=lambda: request.cookies.get(LAST_RECIPIENTS_KEY),
            render_kw={'rows': config['MAILIST_WIDGET_NROWS'],
                       'autocomplete': 'on'})

        dice_report = wtff.TextAreaField(
            # label='Dice Report:',  Set in `_manage_session()`.
            render_kw={'rows': config['DREPORT_WIDGET_NROWS']})

        repeat_dice = wtff.BooleanField(
            label="Repeat dice?",
            render_kw={'disabled': True})

        check = wtff.SubmitField(
            '1: Check...',
            render_kw={})

        submit = wtff.SubmitField(
            '2: Stamp!',
            render_kw={})

        def validate_stamp_recipients(self, field):
            """Must contain at least 1 non-standard email-address."""
            text = field.data
            check_mx = get_bool_arg('skip_mail_mx_check')
            if check_mx is None:
                check_mx = os.name != 'nt'

            recipients = re.split('[\s,;]+', text)
            recipients = [s and s.strip() for s in recipients]
            recipients = list(filter(None, recipients))

            ## Prepend "hidden" default-recipients. (JRC & CLIMA?).
            #
            default_recipients = config.get('DEFAULT_STAMP_RECIPIENTS', [])
            recipients = unique_ci(default_recipients + recipients)

            if len(recipients) < len(default_recipients) + 1:
                raise wtforms.ValidationError(
                    'Specify at least 1 extra recipient! Got: %s' %
                    recipients_str(recipients))

            for i, email in enumerate(recipients, 1):
                if not validate_email(email, check_mx=check_mx):
                    raise wtforms.ValidationError(
                        'Invalid email-address no-%i: `%s`' %
                        (i, recipients_str(recipients)))

            return recipients

        def validate_dice_report(self, field):
            min_dreport_size = config['MIN_DREPORT_SIZE']
            data = field.data and field.data.strip()
            if not data:
                raise wtforms.ValidationError("Dice-report is required.")
            if len(data) < min_dreport_size:
                raise wtforms.ValidationError(
                    "Dice-report is too short (less than %s char)." %
                    min_dreport_size)

            return data

        def _log_client_error(self, action, error, *,
                              fatal=None, **log_kw):
            level = (logging.FATAL
                     if fatal
                     else client_validation_log_level)
            if app.logger.isEnabledFor(level):
                dreport = '<hidden>'
                if client_validation_log_full_dreport:
                    dreport = self.dice_report.data
                    if dreport and client_validation_log_full_dreport is not True:
                        dreport = '%s\n...\n%s' % (
                            dreport[:client_validation_log_full_dreport],
                            dreport[-client_validation_log_full_dreport:])
                indent = ' ' * 4
                app.logger.log(
                    level,
                    tw.dedent("""\
                        Client error while %s:
                          stamp_recipients:
                        %s
                          dice_report:
                        %s
                          error:
                        %s
                    """), action,
                    tw.indent(self.stamp_recipients.data, indent),
                    tw.indent(dreport, indent),
                    tw.indent(pp.pformat(error), indent),
                    **log_kw)

        def _manage_session(self, is_stamped):
            """If `is_stamped`, disable & populate fields from session, else clear it."""
            if is_stamped:
                (sender, stamp_recipients, dice_stamp,
                 dice_decision, mail_err) = [session[k]
                                             for k in
                                             self._skeys]
                self.sender.data = sender
                self.stamp_recipients.data = recipients_str(stamp_recipients)
                self.dice_report.data = dice_stamp
                self.dice_report.description = """
                    <strong><kbd>Copy</kbd> + <kbd>Paste</kbd>
                    this <em>stamp</em> above ^^ into your AIO.</strong>
                """
                dreport_label = "Dice Report <em>Stamped</em>:"
                sent_action = mail_err or 'sent'
                flash(Markup("Stamp %s to %i recipient(s): <pre>%s</pre>" %
                             (sent_action, len(stamp_recipients),
                              escape(recipients_str(stamp_recipients)))),
                      'error' if mail_err else 'info')
                flash(Markup(tw.dedent("""
                    Decision:<pre>\n%s</pre>
                    <div class="alert alert-success">
                      NOTE: you still must import the stamp above back
                            into your project!
                    </div>""") %
                             (escape(yaml.dump({'dice': dice_decision},
                                               default_flow_style=False)))),
                      'error'
                      if dice_decision['decision'] == 'SAMPLE'
                      else 'info')
            else:
                ## Clear session and reset form.
                #
                self.repeat_dice.render_kw['disabled'] = True
                dreport_label = "Dice Report:"
                for k in self._skeys:
                    session.pop(k, None)

            form_disabled = is_stamped
            btns = [self.check, self.submit]
            fields = [self.sender, self.stamp_recipients, self.dice_report]
            for i in fields:
                i.render_kw['readonly'] = form_disabled
            for i in btns:
                i.render_kw['disabled'] = form_disabled
            self.dice_report.label.text = dreport_label

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

        def _check_key_exists(self):
            """Key not hosted locally, maybe missing due to connectivity."""
            check_key_script = config.get('CHECK_SIGNING_KEY_SCRIPT')
            if check_key_script:
                p = sbp.run(check_key_script, stdout=sbp.PIPE, stderr=sbp.PIPE)
                if p.returncode != 0:
                    logger.fatal("Stamper-key missing!  retcode(%s)"
                                 "\n  stdout: %s\n  stderr: %s",
                                 p.returncode, p.stdout, p.stderr)
                    raise CmdException(
                        "Signing temporarily unavailable! "
                        "JRC has been notified, please try again later.")

        def _do_check(self):
            """
            :return:
                tuple(dice_stamp, dice_decision)
            """
            dreport = self.dice_report.data

            try:
                self._check_key_exists()
                recipients = self.validate_stamp_recipients(self.stamp_recipients)

                verdict = self.sign_validator.parse_signed_tag(dreport)
                uid = crypto.uid_from_verdict(verdict)

                # TODO: move sig-validation check in `crypto` module.
                if not verdict['valid']:
                    raise CmdException(
                        "Cannot validate dice-report signed with %r: %s" %
                        (uid, verdict['status']))

                ## Check if test-key still used.
                #
                git_auth = crypto.get_git_auth(config=self.traits_config)
                git_auth.check_test_key_missused(verdict['key_id'])

            except CmdException as ex:
                self._log_client_error("Checking", ex)
                flash(str(ex), 'error')
            except Exception as ex:
                self._log_client_error("Checking", ex, exc_info=1)
                if app.debug:
                    raise
                flash(Markup("Stamp-signing failed due to: %s(%s)"
                      "<br>  Contact JRC for help." % (type(ex).__name__, ex)),
                      'error')
            else:
                flash(Markup("""
                    OK. Now click the "Stamp!" button,
                    and the the dice-report which is signed by the key:
                    <pre>%s</pre>
                    will be stamped, and mailed to %s recipient(s):
                    <pre>%s</pre>
                """ % (uid, len(recipients), recipients_str(recipients))))

        @property
        def sign_validator(self) -> tstamp.TstampReceiver:
            sign_validator = getattr(request, 'sign_validator', None)
            if sign_validator is None:
                request.sign_validator = sign_validator = tstamp.TstampReceiver(
                    config=self.traits_config)

            return sign_validator

        def _sign_dreport(self, dreport, recipients):
            """
            :return:
                tuple(dice_stamp, dice_decision)
            """
            signer = self.signer
            signer.recipients = recipients
            sender = '(%s) %s' % (request.remote_addr, self.sender.data)
            return signer.sign_dreport_as_tstamper(sender,
                                                   dreport)

        def _sendmail(self, mail_cli, txt: Text):
            mail_err = None
            try:
                logger.info("Executing email-CLI: %s", mail_cli)
                p = sbp.Popen(mail_cli, stdin=sbp.PIPE,
                              stderr=sbp.PIPE)
                _stdout, mail_err = p.communicate(txt.encode('utf-8'))
                retcode = p.returncode
                if mail_err:
                    mail_err = mail_err.decode('utf-8')
            except Exception as _:
                mail_err = tb.format_exc()

            if mail_err or retcode:
                self._log_client_error('mail', mail_err,
                                       fatal=True, exc_info=1)

                ## TODO: Fail (no stamp replied!) on PRODUCTION.
                mail_err = 'NOT SENT due to: <pre>%s</pre>' % (
                    mail_err or 'retcode(%s)' % retcode)

            return mail_err

        def _send_stamp_mails(self, sender, recipients,
                              dice_stamp, dice_decision):
            mail_err = None
            mail_cli = config.get('MAIL_CLI_ARGS')
            if mail_cli:
                is_test = crypto.is_test_key(dice_decision['issuer'])
                tag = dice_decision['tag']  # stop after 7 Hash-chars
                if tag:
                    tag = tag[6:-33]  # Skip 'dices/' and stop after 7 Hash-chars
                subject = '[dice%s] %s FROM %s' % (
                    '.test' if is_test else '', tag, sender)

                mail_cli = [m.format(subject=subject)
                            for m in mail_cli] + recipients
                mail_err = self._sendmail(mail_cli, dice_stamp)
            else:
                mail_err = 'NOT SENT'

            return mail_err

        def _do_stamp(self):
            sender = self.sender.data
            recipients = self.validate_stamp_recipients(self.stamp_recipients)
            dreport = self.dice_report.data

            try:
                self._check_key_exists()
                dice_stamp, dice_decision = self._sign_dreport(dreport,
                                                               recipients)
            except CmdException as ex:
                self._log_client_error("Signing", ex)
                flash(str(ex), 'error')
            except Exception as ex:
                self._log_client_error("Signing", ex,
                                       fatal=True, exc_info=1)
                if app.debug:
                    raise
                flash(Markup("Stamp-signing failed due to: %s(%s)"
                      "<br>  Contact JRC for help." % (type(ex).__name__, ex)),
                      'error')
            else:
                mail_err = self._send_stamp_mails(sender, recipients,
                                                  dice_stamp, dice_decision)

                self.repeat_dice.render_kw['disabled'] = True
                session.update(zip(self._skeys,
                                   [sender, recipients, dice_stamp,
                                    dice_decision, mail_err]))
                project = dice_decision['tag']
                add_project_in_stamps_cookie(request.cookies, project)

                @after_this_request
                def store_form_field_as_cookies(response):
                    response.set_cookie(LAST_SENDER_KEY, self.sender.data)
                    response.set_cookie(LAST_RECIPIENTS_KEY, self.stamp_recipients.data)
                    return response

                self._manage_session(True)

        def render(self):
            if not self.is_submitted():
                ## Clear session.
                self._manage_session(False)
            else:
                if 'dice_stamp' in session:
                    flash("Already diced! Click 'New stamp...' above.", 'error')
                    self._manage_session(True)
                else:
                    if not self.validate():
                        self._log_client_error('Stamping', self.errors)
                    else:
                        ## Check if user has diced project another time
                        #  and present a confirmation check-box.
                        #
                        project = self.signer.extract_dice_tag_name(
                            None, self.dice_report.data)
                        if (is_project_in_stamps_cookie(request.cookies, project) and
                                not self.repeat_dice.data):
                            self.repeat_dice.render_kw['disabled'] = False
                            flash(Markup(
                                "You have <em>diced</em> project %r before.<br>"
                                "Please confirm that you really want to dice it again." %
                                project), 'error')
                        else:
                            if self.check.data:
                                self._do_check()
                            elif self.submit.data:
                                self._do_stamp()
                            else:
                                assert False, "Both submission buttons false!"

            # ## NOTE: Enable this code to update `/logconf.yaml`.
            # print('\n'.join(sorted(logging.Logger.manager.loggerDict)))

            return flask.render_template('stamp.html', form=self)

    return StampForm  # class
