import eventlet

#eventlet.monkey_patch()
import zipfile
import tempfile
import subprocess
import os, glob, io
import os.path as osp
from flask import Flask, render_template, request, send_file, jsonify
from flask_socketio import SocketIO
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = '.'
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app, async_mode='eventlet')


@socketio.on('get_log')
def get_log(filename, offset):
    try:
        with open(osp.join('logs', filename)) as file:
            file.seek(offset)
            return file.readlines(), file.tell()
    except FileNotFoundError:
        return '', 0


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/js/<path:path>')
def send_js(path):
    return app.send_static_file('js/%s' % path)


@app.route('/upload', methods=['POST', 'GET'])
def upload():
    if request.method == 'POST':
        os.makedirs('inputs', exist_ok=True)
        for f in request.files.getlist('files'):
            f.save(osp.join('inputs', secure_filename(f.filename)))
        return 'files uploaded successfully'


@app.route('/download', methods=['GET', 'POST'])
def download():
    data = request.json
    folder = osp.join(app.root_path, data['folder'])
    out = io.BytesIO()
    with zipfile.ZipFile(out, mode='w') as zip:
        for filename in data['files']:
            fpath = osp.join(folder, filename)
            if osp.isfile(fpath):
                zip.write(fpath, filename)
    out.seek(0)
    return send_file(out, as_attachment=True, attachment_filename='output.zip', mimetype='application/zip')

@app.route('/download_piz', methods=['GET', 'POST'])
def download_():
    return send_file('file.zip')

@app.route('/delete', methods=['GET', 'POST'])
def delete():
    data = request.json
    folder = osp.join(app.root_path, data['folder'])
    for filename in data['files']:
        fpath = osp.join(folder, filename)
        if osp.isfile(fpath):
            os.remove(fpath)
    return 'files removed successfully'


@app.route('/folder/outputs', methods=['GET', 'POST'])
def outputs():
    return _folder('outputs', 'xlsx', 'co2mpas.ta')


@app.route('/folder/inputs', methods=['GET', 'POST'])
def inputs():
    return _folder('inputs', 'xlsx')


def _folder(path, *exts):
    folder, files = osp.join(app.root_path, path), []
    for ext in exts:
        files.extend(glob.glob(osp.join(folder, '*.%s' % ext)))

    res = {
        'data': [
            {'fname': osp.basename(f), 'fsize': round(osp.getsize(f) / 1e6, 2)}
            for f in sorted(files)
        ]
    }
    return jsonify(res)


def run_cmd(filename, command):
    with open(filename, 'w') as file:
        p = subprocess.Popen(command, stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE)
        k = osp.splitext(osp.basename(filename))[0]
        while p.poll() is None:
            for line in iter(p.stderr.readline, b''):
                line = line.decode('ascii', 'ignore')
                line = line.replace('\r\n', '\n').replace('\r', '\n')
                if line != '\n':
                    file.write(line)
                    socketio.emit('update', (k, file.tell()))


@app.route('/run', methods=['GET', 'POST'])
def run():
    from co2mpas.__main__ import __file__ as co2mpas_cmd
    data = request.json
    folder = osp.join(app.root_path, data['folder'])
    cmd = ['python', co2mpas_cmd, 'batch']
    cmd.extend(osp.join(folder, f) for f in data['files'])
    outputs = osp.join(app.root_path, 'outputs')
    os.makedirs(outputs, exist_ok=True)
    cmd.extend(['-O', outputs, '-D', 'flag.engineering_mode=True'])
    logs = osp.join(app.root_path, 'logs')
    os.makedirs(logs, exist_ok=True)
    filename = tempfile.mktemp('.log', '', logs)
    k = osp.splitext(osp.basename(filename))[0]
    socketio.emit('task', k, broadcast=True)
    run_cmd(filename, cmd)
    return ''


if __name__ == '__main__':
    socketio.run(app, debug=True)
