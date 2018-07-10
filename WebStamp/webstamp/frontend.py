# This contains our frontend; since it is a bit messy to use the @app.route
# decorator style when using application factories, all of our routes are
# inside blueprints. This is the front-facing blueprint.
#
# You can find out more about blueprints at
# http://flask.pocoo.org/docs/blueprints/

from flask import request
import flask

from . import forms


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

    StampForm = forms.create_stamp_form_class(app)

    @frontend.route('/stamp/', methods=('GET', 'POST'))
    def stamp():
        log.info("WebStamp URL: %s\n  values: %s",
                 request.url, request.values)
        try:
            return StampForm().render()
        except Exception as ex:
            log.fatal('WebStamp crashed due to: %s\n  %s',
                      ex, request.values, exc_info=1)
            raise
