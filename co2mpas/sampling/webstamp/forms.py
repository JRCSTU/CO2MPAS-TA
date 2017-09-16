from flask_wtf import FlaskForm
import logging
import os
from pprint import pformat
import re

from flask import flash, session
import flask
from markupsafe import escape, Markup
from validate_email import validate_email
import wtforms

import textwrap as tw
import wtforms.fields as wtff
import wtforms.fields.html5 as wtf5
import wtforms.validators as wtfl


def _validate_email_list(form, field):
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

    class StampForm(FlaskForm):

        dreport_recipients = wtff.TextAreaField(
            label='Dice-report Recipients:',
            description="(separate email-addresses by <kbd>,</kbd>, <kbd>;</kbd>, "
            "<kbd>[Space]</kbd>, <kbd>[Enter]</kbd>, <kbd>[Tab]</kbd> characters)",
            validators=[wtfl.InputRequired(), _validate_email_list],
            default=config.get('DREPORT_CC_DEFAULT'),
            render_kw={'rows': config['MAILIST_WIDGET_NROWS']})

        stamp_recipients = wtff.TextAreaField(
            label='Stamp Recipients:',
            description="(add HERE your own email-address)",
            validators=[wtfl.InputRequired(), _validate_email_list],
            default=config.get('STAMP_RECIPIENTS_DEFAULT'),
            render_kw={'rows': config['MAILIST_WIDGET_NROWS']})

        dice_report = wtff.TextAreaField(
            # label='Dice Report:',  Set in `_mark_as_stamped()`.
            description="(copy-->paste the dice-report above)",
            render_kw={'rows': config['DREPORT_WIDGET_NROWS']})

        submit = wtff.SubmitField(
            'Stamp!',
            description=" This action is irreversible!",
            render_kw={})

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

        def _log_client_errors(self):
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
                        CLient validation-errors while Stamping:
                          dreport_recipients:
                        %s
                          stamp_recipients:
                        %s
                          dice_report:
                        %s
                          errors:
                        %s
                    """),
                    tw.indent(self.dreport_recipients.data, indent),
                    tw.indent(self.stamp_recipients.data, indent),
                    tw.indent(dreport, indent),
                    tw.indent(pformat(self.errors), indent))

        def _mark_as_stamped(self, is_stamped):
            """Update form-widgets if dreport has been stamped or clear session. """
            skeys = 'dice_stamp dreport_recipients stamp_recipients'.split()
            form_disabled = is_stamped

            if is_stamped:
                dice_stamp, dreport_recipients, stamp_recipients = [session[k]
                                                                    for k in skeys]
                self.dice_report.data = dice_stamp
                self.dreport_recipients.data = dreport_recipients
                self.stamp_recipients.data = stamp_recipients
                dreport_label = "Dice Report <em>Stamped</em>:"
                flash(
                    Markup(
                        """
                    <ul>
                      <li><em>Dice-report</em> CC-ed to %i recipient(s): %s</li>
                      <li><em>Dice-stamp</em> sent to %i recipient(s): %s</li>
                    </ul>
                    """ % (
                            len(stamp_recipients),
                            escape('; '.join(stamp_recipients)),
                            len(dreport_recipients),
                            escape('; '.join(dreport_recipients)))))
            else:
                dreport_label = "Dice Report:"
                for k in 'dice_stamp dreport_recipients stamp_recipients'.split():
                    session.pop(k, None)

            self.dreport_recipients.render_kw['readonly'] = form_disabled
            self.stamp_recipients.render_kw['readonly'] = form_disabled
            self.dice_report.render_kw['readonly'] = form_disabled
            self.submit.render_kw['disabled'] = form_disabled
            self.dice_report.label.text = dreport_label

        def _do_stamp(self):
            stamp_recipients = _validate_email_list(self,
                                                    self.stamp_recipients)
            dreport_recipients = _validate_email_list(self,
                                                      self.dreport_recipients)
            dreport = self.validate_dice_report(self.dice_report)

            dice_stamp = '#### STAMPED!\n%s' % dreport  # TODO: stamp!

            session.update({
                'dice_stamp': dice_stamp,
                'dreport_recipients': dreport_recipients,
                'stamp_recipients': stamp_recipients
            })
            flash(Markup("Import the <em>dice-stamp</em> above into your project."))
            self._mark_as_stamped(True)

        def render(self):
            if not self.is_submitted():
                self._mark_as_stamped(False)
            else:
                if 'dice_stamp' in session:
                    flash("Already diced! Click 'New Stamp...' above.", 'error')
                    self._mark_as_stamped(True)
                else:
                    if self.validate():
                        self._do_stamp()
                    else:
                        self._log_client_errors()

            # return flask.redirect(flask.url_for('.index'))
            return flask.render_template('stamp.html', form=self)

    return StampForm  # class
