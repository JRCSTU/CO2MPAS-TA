# This contains our frontend; since it is a bit messy to use the @app.route
# decorator style when using application factories, all of our routes are
# inside blueprints. This is the front-facing blueprint.
#
# You can find out more about blueprints at
# http://flask.pocoo.org/docs/blueprints/

import flask

from . import forms


frontend = flask.Blueprint('frontend', __name__)


# Our index-page just shows a quick explanation. Check out the template
# "templates/index.html" documentation for more details.
@frontend.route('/')
def index():
    return flask.render_template('index.html')


## As a method, for blueprint to access `app` & `config`.
#  See https://stackoverflow.com/a/23037071/548792
#
@frontend.record
def attach_routes(setup_state):
    StampForm = forms.create_stamp_form_class(setup_state.app)

    @frontend.route('/stamp/', methods=('GET', 'POST'))
    def stamp():
        form = StampForm()

        if form.is_submitted():
            page = form.do_stamp()
        else:
            page = flask.render_template('stamp.html', form=form)

        return page
