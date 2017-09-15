import os
from pprint import pformat
import re

import flask
from flask_wtf import FlaskForm
from markupsafe import escape, Markup
from validate_email import validate_email
import wtforms

import wtforms.fields as wtff
import wtforms.validators as wtfl


min_dreport_size = 1000
mailist_widget_nrows = 2
dreport_widget_nrows = 17


def validate_email_list(form, field):
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


class StampForm(FlaskForm):

    dreport_cc = wtff.TextAreaField(
        label='Dice-report CC Recipients:',
        description="(separate email-addresses by <kbd>,</kbd>, <kbd>;</kbd>, "
        "<kbd>[Space]</kbd>, <kbd>[Enter]</kbd>, <kbd>[Tab]</kbd> characters)",
        validators=[wtfl.InputRequired(), validate_email_list],
        render_kw={'rows': mailist_widget_nrows})

    stamp_recipients = wtff.TextAreaField(
        label='Stamp Recipients:',
        description="(add HERE your own email-address)",
        validators=[wtfl.InputRequired(), validate_email_list],
        render_kw={'rows': mailist_widget_nrows})

    dice_report = wtff.TextAreaField(
        label='Dice report:',
        description="(copy-->paste the dice-report above)",
        render_kw={'rows': dreport_widget_nrows})

    submit = wtff.SubmitField('Stamp!', description=" This action is irreversible!")

    def validate_dice_report(self, field):
        data = field.data and field.data.strip()
        if not data:
            raise wtforms.ValidationError("Dice-report is required.")
        if len(data) < min_dreport_size:
            raise wtforms.ValidationError(
                "Dice-report is too short (less than %s char)." % min_dreport_size)

        return data

    def do_stamp(self):
        if self.validate():
            stamp_recipients = validate_email_list(self, self.stamp_recipients)
            dreport_recipients = validate_email_list(self, self.dreport_cc)
            stamp = self.validate_dice_report(self.dice_report)

            # Note that the default flashed messages rendering allows HTML, so
            # we need to escape things if we input user values:
            msgs = [
                Markup("""
                <ul>
                  <li><em>Dice-report</em> CC-ed to %i recipient(s): %s</li>
                  <li><em>Stamp</em> sent to %i recipient(s): %s</li>
                  <li>Select the <em>stamp</em> below and copy it into clipboard:
                      <pre>\n%s\n</pre>
                  </li>
                </ul>
                """ % (
                    len(stamp_recipients), '; '.join(stamp_recipients),
                    len(dreport_recipients), '; '.join(dreport_recipients),
                    escape(stamp)))
            ]
            for msg in msgs:
                print(msg)
                flask.flash(msg)
        else:
            print(pformat(self.errors), vars(self))

        #return flask.redirect(flask.url_for('.index'))
        return flask.render_template('stamp.html', form=self)
