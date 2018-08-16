# -*- coding: utf-8 -*-
# !/usr/bin/env python
#
# Copyright 2014-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl
from sampling import CmdException, crypto, tstamp
from collections import namedtuple
from flask_wtf import FlaskForm
import json
import logging
from typing import List, Text

from flask import flash, request
import flask
from flask.ctx import after_this_request
from markupsafe import escape, Markup
import wtforms
import yaml

import pprint as pp
import subprocess as sbp
import textwrap as tw
import textwrap as w
import traceback as tb
import wtforms.fields as wtff
import wtforms.validators as wtfl

from .import frontend


STAMPED_PROJECTS_KEY = 'stamped_projects'
LAST_SENDER_KEY = 'last_sender'
LAST_RECIPIENTS_KEY = 'last_recipients'


logger = logging.getLogger(__name__)


def ascii_validator(_form, field):
    value = field.data
    if not all(ord(c) < 128 for c in value):
        raise wtfl.ValidationError("Field does not accept non-ASCII chars.")


def create_stamp_form_class(app, Stamper):
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

    def mark_project_as_stamped(cookies, project):
        stamped_projects = _get_stamped_projects(cookies)
        stamped_projects.add(project)
        stamped_projects = list(sorted(set(stamped_projects)))

        return stamped_projects

    #: The form-keys of stamp-data submitted and processed.
    #: The `recipients` is there bc it may modify on each request roundtrip.
    FData = namedtuple("FData",
                       'stamp_recipients dice_stamp dice_decision mail_err')

    class StampForm(FlaskForm, Stamper):
        """
        Form submission boolean args:

        - ``allow_test_key``: allow CBBB52FF
        - ``validate_decision``: append decision
        - ``trim_dreport``: remove garbage suffix from dreport
        """

        sender = wtff.TextField(
            label='Sender:',
            description="""
                Your <em>designated-email</em> and/or your name or id
                (free form, but only ASCII).
            """,
            validators=[wtfl.InputRequired(),
                        wtfl.Length(min=3, max=40),
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
            # label='Dice Report:',  Set in `_manage_gui_state()`.
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

        def validate_stamp_recipients(self, field) -> List[str]:
            """Must contain at least 1 non-standard email-address."""
            recipient_txt = field.data
            try:
                recipients = self.parse_recipients_text(recipient_txt)
            except ValueError as ex:
                raise wtforms.ValidationError(str(ex))
            except Exception as ex:
                raise wtforms.ValidationError("%s: %s" % (type(ex).__name__, ex))

            self.stamp_recipients.data = frontend.recipients_str(recipients)

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
                    w.dedent("""\
                        Client error while %s:
                          stamp_recipients:
                        %s
                          dice_report:
                        %s
                          error:
                        %s
                    """), action,
                    w.indent(self.stamp_recipients.data, indent),
                    tw.indent(dreport, indent),
                    tw.indent(pp.pformat(error), indent),
                    **log_kw)

        def _manage_gui_state(self, fdata: FData = None):
            """If `is_stamped`, disable & populate fields from session, else clear it."""
            form_disabled = bool(fdata and fdata.dice_stamp)
            if form_disabled:
                self.dice_report.data = fdata.dice_stamp
                self.repeat_dice.render_kw['disabled'] = True
                dreport_label = "Dice Report <em>Stamped</em>:"
                sent_action = fdata.mail_err or 'sent'
                flash(Markup("Stamp %s to %i recipient(s): <pre>%s</pre>" %
                             (sent_action, len(fdata.stamp_recipients),
                              escape(self.stamp_recipients.data))),
                      'error' if fdata.mail_err else 'info')
                flash(Markup(tw.dedent("""
                    Decision:<pre>\n%s</pre>
                    <div class="alert alert-success">
                      NOTE: you still must import the stamp above back
                            into your project!
                    </div>""") %
                             (escape(yaml.dump({'dice': fdata.dice_decision},
                                               default_flow_style=False)))),
                      'error'
                      if fdata.dice_decision['decision'] == 'SAMPLE'
                      else 'info')
            else:
                dreport_label = tw.dedent("""
                    <em>Paste</em> Dice Report below, or <em>Upload</em> it from a local file:
                    <input type="file" class="btn" id="upload-filename" style="display: inline-block;" />
                    <span id="upload-spinner", class="spinner hidden" />
                """)

            btns = [self.check, self.submit]
            fields = [self.sender, self.stamp_recipients, self.dice_report]
            for i in fields:
                i.render_kw['readonly'] = form_disabled
            for i in btns:
                i.render_kw['disabled'] = form_disabled
            self.dice_report.label.text = dreport_label
            ## Notice that `repeat_dice` is not touched here (sticky).

        @property
        def sign_validator(self) -> tstamp.TstampReceiver:
            sign_validator = getattr(request, 'sign_validator', None)
            if sign_validator is None:
                request.sign_validator = sign_validator = tstamp.TstampReceiver(
                    config=self.traits_config)

            return sign_validator

        def _do_check(self):
            """
            :return:
                tuple(dice_stamp, dice_decision)
            """
            recipients = self.validate_stamp_recipients(self.stamp_recipients)
            sender = self.sender.data
            dreport = self.dice_report.data

            try:
                uid = self.check_dice(dreport, sender, recipients)
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
                    and the dice-report which is signed by the key:
                    <pre>%s</pre>
                    will be stamped, and mailed to %s recipient(s):
                    <pre>%s</pre>
                """ % (uid, len(recipients), frontend.recipients_str(recipients))))

            return FData(recipients, None, None, None)

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
            mail_err = None

            try:
                dice_stamp, dice_decision = self.sign_dreport(dreport,
                                                               sender, recipients)
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

                project = dice_decision['tag']
                stamped_projects = mark_project_as_stamped(request.cookies, project)

                @after_this_request
                def store_form_field_as_cookies(response):
                    response.set_cookie(LAST_SENDER_KEY, self.sender.data)
                    response.set_cookie(LAST_RECIPIENTS_KEY, self.stamp_recipients.data)
                    response.set_cookie(STAMPED_PROJECTS_KEY, json.dumps(stamped_projects))

                    return response

                return FData(recipients, dice_stamp, dice_decision, mail_err)

        def render(self):
            if not self.is_submitted():
                ## New session.
                #
                # Sticky until old-stampe found OR stamped.
                self.repeat_dice.render_kw['disabled'] = True
                self._manage_gui_state()
            else:
                fdata = None

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
                            "Please confirm to <strong>repeat dice</strong>, above." %
                            project), 'error')

                    else:
                        if self.check.data:
                            fdata = self._do_check()
                        elif self.submit.data:
                            fdata = self._do_stamp()
                        else:
                            raise AssertionError("Both submission buttons false!")

                self._manage_gui_state(fdata)

            # ## NOTE: Enable this code to update `/logconf.yaml`.
            # print('\n'.join(sorted(logging.Logger.manager.loggerDict)))

            return flask.render_template('stamp.html', form=self)

    return StampForm  # class
