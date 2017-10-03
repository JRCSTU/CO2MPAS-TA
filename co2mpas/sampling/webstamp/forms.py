import json
import logging
import os
import re

from co2mpas._vendor.traitlets import config as traitc
from co2mpas.sampling import CmdException, crypto, tsigner, tstamp
from flask import flash, request, session
from typing import Text
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
import wtforms.fields as wtff
import wtforms.validators as wtfl


STAMPED_PROJECTS_KEY = 'stamped_projects'


def get_bool_arg(argname):
    """
    True is an arg alone, or a (possibly empty) string but (`no | false | 0`).
    """
    args = request.args
    if argname in args:
        param = args[argname].strip()
        return param.lower() not in ['0', 'false', 'no', 'off']


def create_stamp_form_class(app):
    ## Prepare various config-dependent constants

    config = app.config
    client_validation_log_full_dreport = config['CLIENT_VALIDATION_LOG_FULL_DREPORT']
    client_validation_log_level = config['CLIENT_VALIDATION_LOG_LEVEL']
    try:
        client_validation_log_level = int(client_validation_log_level)
    except:  # @IgnorePep8
        client_validation_log_level = logging._nameToLevel.get(
            client_validation_log_level, logging.DEBUG)

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

        _skeys = 'dice_stamp stamp_recipients dice_decision mail_err'.split()

        stamp_recipients = wtff.TextAreaField(
            label='Stamp Recipients:',
            description="(separate email-addresses by <kbd>,</kbd>, <kbd>;</kbd>, "
            "<kbd>[Space]</kbd>, <kbd>[Enter]</kbd>, <kbd>[Tab]</kbd> characters)",
            validators=[wtfl.InputRequired()],
            default=config.get('DEFAULT_STAMP_RECIPIENTS'),
            render_kw={'rows': config['MAILIST_WIDGET_NROWS']})

        dice_report = wtff.TextAreaField(
            # label='Dice Report:',  Set in `_manage_session()`.
            render_kw={'rows': config['DREPORT_WIDGET_NROWS']})

        repeat_dice = wtff.BooleanField(
            label="Repeat dice?",
            render_kw={'disabled': True})

        check = wtff.SubmitField(
            'Check...',
            render_kw={})

        submit = wtff.SubmitField(
            'Stamp!',
            render_kw={})

        def validate_stamp_recipients(self, field):
            text = field.data
            check_mx = os.name != 'nt'

            mails = re.split('[\s,;]+', text)
            mails = [s and s.strip() for s in mails]
            mails = list(filter(None, mails))
            for i, email in enumerate(mails, 1):
                if not validate_email(email, check_mx=check_mx):
                    raise wtforms.ValidationError(
                        'Invalid email-address no-%i: `%s`' % (i, email))

            return mails

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

        def _log_client_error(self, action, error, **log_kw):
            dreport = '<hidden>'
            if client_validation_log_full_dreport:
                dreport = self.dice_report.data
                if dreport and client_validation_log_full_dreport is not True:
                    dreport = '%s\n...\n%s' % (
                        dreport[:client_validation_log_full_dreport],
                        dreport[-client_validation_log_full_dreport:])
            if app.logger.isEnabledFor(client_validation_log_level):
                indent = ' ' * 4
                app.logger.log(
                    client_validation_log_level,
                    tw.dedent("""
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
                dice_stamp, stamp_recipients, dice_decision, mail_err = [session[k]
                                                                         for k in
                                                                         self._skeys]
                self.dice_report.data = dice_stamp
                self.stamp_recipients.data = '; '.join(stamp_recipients)
                dreport_label = "Dice Report <em>Stamped</em>:"
                sent_action = mail_err or 'sent'
                flash(Markup("<em>Dice-stamp</em> %s to %i recipient(s): %s"
                             "<br>Decision:<pre>\n%s</pre>" %
                             (sent_action, len(stamp_recipients),
                              escape('; '.join(stamp_recipients)),
                              escape(yaml.dump({'dice': dice_decision},
                                               default_flow_style=False)))),
                      'error' if mail_err else 'info')
            else:
                ## Clear session and reset form.
                #
                self.repeat_dice.render_kw['disabled'] = True
                dreport_label = "Dice Report:"
                for k in self._skeys:
                    session.pop(k, None)

            form_disabled = is_stamped
            btns = [self.repeat_dice, self.check, self.submit]
            fields = [self.stamp_recipients, self.dice_report]
            for i in fields:
                i.render_kw['readonly'] = form_disabled
            for i in btns:
                i.render_kw['disabled'] = form_disabled
            self.dice_report.label.text = dreport_label

        @property
        def traits_config(self):
            ## Convert Flask-config --> traitlets-config
            #  and respect `allow_test_key` form-param.
            #
            traits_config = traitc.Config(config['TRAITLETS_CONFIG'])

            flag = get_bool_arg('allow_test_key')
            if flag is not None:
                traits_config.GpgSpec.allow_test_key = flag

            flag = get_bool_arg('validate_decision')
            if flag is not None:
                traits_config.TsignerService.validate_decision = flag

            flag = get_bool_arg('trim_dreport')
            if flag is not None:
                traits_config.TsignerService.trim_dreport = flag

            return traits_config

        _signer = None

        @property
        def signer(self) -> tsigner.TsignerService:
            if not self._signer:
                self._signer = tsigner.TsignerService(
                    config=self.traits_config)

            return self._signer

        def _do_check(self):
            """
            :return:
                tuple(dice_stamp, dice_decision)
            """
            dreport = self.dice_report.data
            check_key_script = config.get('CHECK_SIGNING_KEY_SCRIPT')

            try:
                if check_key_script:
                    if sbp.run(check_key_script).returncode != 0:
                        raise CmdException("Signing temporarily unavailable! "
                                           "Please try later.")

                verdict = self.sign_validator.parse_signed_tag(dreport)
                uid = crypto.uid_from_verdict(verdict)

                # TODO: move sig-validation check in `crypto` module.
                if not verdict['valid']:
                    raise CmdException(
                        "Cannot validate dice-report signed with %r: %s" %
                        (uid, verdict['status']))
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
                flash("Dice-report signed with %r key is OK."
                      " You may proceed with stamping." % uid)

        _sign_validator = None

        @property
        def sign_validator(self) -> tstamp.TstampReceiver:
            if not self._sign_validator:
                self._sign_validator = tstamp.TstampReceiver(
                    config=self.traits_config)

            return self._sign_validator

        def _sign_dreport(self, dreport, recipients):
            """
            :return:
                tuple(dice_stamp, dice_decision)
            """
            signer = self.signer
            signer.recipients = recipients
            return signer.sign_dreport_as_tstamper(dreport)

        def _sendmail(self, mail_cli, txt: Text):
            mail_err = None
            try:
                p = sbp.Popen(mail_cli, stdin=sbp.PIPE,
                              stderr=sbp.PIPE)
                _stdout, mail_err = p.communicate(txt.encode('utf-8'))
                retcode = p.returncode
            except Exception as ex:
                self._log_client_error('mail', ex, exc_info=1)
                mail_err = ex

            if mail_err or retcode:
                mail_err = 'NOT SENT (due to: %s)' % (
                    mail_err or 'retcode(%s)' % retcode)

            return mail_err

        def _send_stamp_mails(self, recipients, dice_stamp, dice_decision):
            mail_err = None
            mail_cli = config.get('MAIL_CLI_ARGS')
            if mail_cli:
                is_test = crypto.is_test_key(dice_decision['issuer'])
                subject = '[dice%s] %s' % ('.test' if is_test else '',
                                           dice_decision['tag'])

                mail_cli = [m.format(recipients=' '.join(recipients),
                                     subject=subject)
                            for m in mail_cli]
                mail_err = self._sendmail(mail_cli, dice_stamp)
            else:
                mail_err = 'NOT SENT'

            return mail_err

        def _do_stamp(self):
            recipients = self.validate_stamp_recipients(self.stamp_recipients)
            dreport = self.dice_report.data

            try:
                dice_stamp, dice_decision = self._sign_dreport(dreport,
                                                               recipients)
            except CmdException as ex:
                self._log_client_error("Signing", ex)
                flash(str(ex), 'error')
            except Exception as ex:
                self._log_client_error("Signing", ex, exc_info=1)
                if app.debug:
                    raise
                flash(Markup("Stamp-signing failed due to: %s(%s)"
                      "<br>  Contact JRC for help." % (type(ex).__name__, ex)),
                      'error')
            else:
                mail_err = self._send_stamp_mails(recipients, dice_stamp, dice_decision)

                self.repeat_dice.render_kw['disabled'] = True
                session.update(zip(self._skeys,
                                   [dice_stamp, recipients, dice_decision, mail_err]))
                flash(Markup(
                    "Import the <em>dice-stamp</em> above into your project."))
                project = dice_decision['tag']
                add_project_in_stamps_cookie(request.cookies, project)
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

            return flask.render_template('stamp.html', form=self)

    return StampForm  # class
