##################
WebStamper Changes
##################
.. contents::

NEXT RELEASE
======================


v1.9.1a0: 2018-08-05
====================
BIG DICES support to Upload/Download them with files.

- Drop `flask-appconfig` - has been deprecated by `flask >= 1.0`.
  Syntax of launch command `flask` has changed (now need to set :envvar:`FLASK_APP`).
- Stop storing contents of old-dices into cookies
  (cannot retrieve previous one anymore).


v1.9.0a2: 2018-07-11
====================
SPLIT OFF as a separate *polyvers* project.

- Provide *wsgi* sample file.
- Make stamper's name configurable (to run on a *pre-production* server).


v1.9.0a2: 2018-07-10
====================
Fixes since 1.7.4 and for bigger *VFID*.
