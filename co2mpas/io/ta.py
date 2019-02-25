# -*- coding: utf-8 -*-
#
# Copyright 2015-2018 European Commission (JRC);
# Licensed under the EUPL (the 'Licence');
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at: http://ec.europa.eu/idabc/eupl

"""
It provides CO2MPAS-TA output model.
"""
import io
import os
import copy
import yaml
import zlib
import json
import lmfit
import secrets
import tarfile
import logging
import functools
import itertools
import numpy as np
import os.path as osp
import schedula as sh
from scipy.interpolate import InterpolatedUnivariateSpline as Spline

log = logging.getLogger(__name__)
SKIP_PARAMETERS = 'models_uuid',


def _json_default(o):
    if isinstance(o, np.ndarray):
        return {'__numpy__': o.tolist()}
    elif isinstance(o, np.generic):
        return o.item()
    elif isinstance(o, lmfit.Parameter):
        return {'__parameter__': o.__getstate__()}
    elif isinstance(o, bytes):
        return {'__bytes__': str(o)}
    raise TypeError("Object of type '%s' is not JSON serializable" %
                    o.__class__.__name__)


def _json_object_hook(dct):
    if '__numpy__' in dct:
        return np.array(dct['__numpy__'])
    elif '__parameter__' in dct:
        _par = lmfit.Parameter()
        _par.__setstate__(dct['__parameter__'])
        return _par
    elif '__bytes__' in dct:
        return eval(dct['__bytes__'])
    return dct


def _filter_data(report):
    report = {k: v for k, v in report.items() if k != 'pipe'}
    for k, v in sh.stack_nested_keys(report):
        if hasattr(v, '__call__') or hasattr(v, 'predict') or \
                (isinstance(v, list) and isinstance(v[0], Spline)) or \
                k[-1] in SKIP_PARAMETERS:
            continue
        yield '.'.join(map(str, k)), v


def write_tar(tar, path, bytes):
    info = tarfile.TarInfo(path)
    info.size = len(bytes)
    tar.addfile(info, io.BytesIO(bytes))


def load_public_RSA_keys(fpath):
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    keys = {}
    with tarfile.open(fpath, 'r') as tar:
        for k in itertools.product(('public',), ('secret', 'server')):
            with tar.extractfile(tar.getmember('%s/%s.pem' % k[::-1])) as f:
                keys[k[-1]] = serialization.load_pem_public_key(
                    f.read(), default_backend()
                )
    return keys


def load_sign_key(sign_key, password=None):
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    if password is None:
        password = os.environ.get('SIGN_KEY_PASSWORD', 'co2mpas') or None

    if not osp.isfile(sign_key):
        generate_sing_key(sign_key, password)

    with open(sign_key) as file:
        d = json.load(file)
    password = d.get('password', password)

    if isinstance(password, str):
        password = password.encode()

    return serialization.load_pem_private_key(
        d['key'].encode(), password, default_backend()
    )


def sign_ta_id(ta_id, sign_key, password=None):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import utils
    from cryptography.hazmat.primitives.asymmetric import padding

    key = load_sign_key(sign_key, password)

    ta_id.pop('signature', None)
    ta_id['pub_sign_key'] = key.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    message = json.dumps(ta_id, default=_json_default, sort_keys=True).encode()
    ta_id['signature'] = key.sign(
        make_hash(message),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        utils.Prehashed(hashes.SHA256())
    )
    return ta_id


def verify_ta_id(ta_id):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import utils
    from cryptography.hazmat.primitives.asymmetric import padding
    message = json.dumps(
        {k: v for k, v in ta_id.items() if k != 'signature'},
        default=_json_default, sort_keys=True
    ).encode()
    serialization.load_pem_public_key(
        ta_id['pub_sign_key'], default_backend()
    ).verify(
        ta_id['signature'], make_hash(message),
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.MAX_LENGTH
        ),
        utils.Prehashed(hashes.SHA256())
    )
    return ta_id


def load_private_RSA_keys(fpath, passwords=None):
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    keys = {}

    with tarfile.open(fpath, 'r') as tar:
        it = itertools.product(('private',), ('secret', 'server'))
        for k, p in itertools.zip_longest(it, passwords or ()):
            p = p is not None and p.encode() or None
            info = tar.getmember('%s/%s.pem' % k[::-1])
            with tar.extractfile(info) as f:
                keys[k[-1]] = serialization.load_pem_private_key(
                    f.read(), p, default_backend()
                )
    return keys


def define_associated_data(public_RSA_keys):
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import hashes, serialization
    digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
    for k in ('secret', 'server'):
        digest.update(public_RSA_keys[k].public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        ))
    return digest.finalize()


def make_hash(*data):
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.backends import default_backend
    digest = hashes.Hash(hashes.SHA256(), backend=default_backend())
    for v in data:
        digest.update(v)
    return digest.finalize()


def generate_sing_key(sign_key, password=None):
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
        backend=default_backend()
    )

    if password is None:
        encrypt_alg = serialization.NoEncryption()
    else:
        encrypt_alg = serialization.BestAvailableEncryption(password.encode())

    key = key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=encrypt_alg
    ).decode()

    with open(sign_key, 'w') as file:
        json.dump({'key': key, 'password': password}, file, sort_keys=1)


def generate_keys(key_folder, passwords=None):
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import rsa

    keys = {'private': {}, 'public': {}}
    for k, p in itertools.zip_longest(('secret', 'server'), passwords or ()):
        key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )

        if p is None:
            encryption_alg = serialization.NoEncryption()
        else:
            encryption_alg = serialization.BestAvailableEncryption(p.encode())

        keys['private'][k] = key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=encryption_alg
        )

        keys['public'][k] = key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo
        )

    it = (
        ('dice.co2mpas.keys', (('public', 'secret'), ('public', 'server'))),
        ('server.co2mpas.keys',
         (('public', 'secret'), ('public', 'server'), ('private', 'server'))),
        ('secret.co2mpas.keys',
         (itertools.product(('public', 'private'), ('secret', 'server'))))
    )

    for fpath, v in it:
        with tarfile.open(osp.join(key_folder, fpath), 'w') as tar:
            for k in v:
                path = '%s.pem' % '/'.join(k[::-1])
                write_tar(tar, path, sh.get_nested_dicts(keys, *k))


def define_rsa_padding():
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    return padding.OAEP(
        mgf=padding.MGF1(algorithm=hashes.SHA256()),
        algorithm=hashes.SHA256(),
        label=None
    )


def rsa_encrypt(rsa, plaintext):
    return rsa.encrypt(plaintext, define_rsa_padding())


def rsa_decrypt(rsa, plaintext):
    return rsa.decrypt(plaintext, define_rsa_padding())


def aes_cipher(key, iv, tag=None):
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
    # Construct a Cipher object, with the key, iv, and additionally the
    # GCM tag used for authenticating the message.
    return Cipher(algorithms.AES(key), modes.GCM(iv, tag), default_backend())


def aes_encrypt(plaintext, associated_data):
    # Generate a random 96-bit IV and 256-bit key.
    iv, key = os.urandom(12), os.urandom(32)
    encryptor = aes_cipher(key, iv).encryptor()

    # associated_data will be authenticated but not encrypted,
    # it must also be passed in on decryption.
    encryptor.authenticate_additional_data(associated_data)

    # Encrypt the plaintext and get the associated ciphertext.
    # GCM does not require padding.
    data = encryptor.update(plaintext) + encryptor.finalize()

    return {'key': key, 'iv': iv, 'tag': encryptor.tag}, data


def aes_decrypt(data, associated_data, key, iv, tag):
    decryptor = aes_cipher(key, iv, tag).decryptor()

    # We put associated_data back in or the tag will fail to verify
    # when we finalize the decryptor.
    decryptor.authenticate_additional_data(associated_data)

    # Decryption gets us the authenticated plaintext.
    # If the tag does not match an InvalidTag exception will be raised.
    return decryptor.update(data) + decryptor.finalize()


def encrypt_data(data, path_keys):
    rsa = load_public_RSA_keys(path_keys)
    plaintext = zlib.compress(yaml.safe_dump(data).encode())
    key, data = aes_encrypt(plaintext, define_associated_data(rsa))
    key = rsa_encrypt(rsa['secret'], yaml.safe_dump(key).encode())
    verify = rsa_encrypt(rsa['server'], make_hash(key, data))
    return {'verify': verify, 'encrypted': {'key': key, 'data': data}}


def verify_AES_key(private_RSA_keys, verify, encrypted):
    verify = rsa_decrypt(private_RSA_keys['server'], verify)
    return make_hash(encrypted['key'], encrypted['data']) == verify


def decrypt_data(encrypted_data, path_keys, passwords=None):
    verify, encrypted = encrypted_data['verify'], encrypted_data['encrypted']
    rsa = load_private_RSA_keys(path_keys, passwords=passwords)
    assert verify_AES_key(rsa, verify, encrypted)
    kw = yaml.load(rsa_decrypt(rsa['secret'], encrypted['key']))
    ad = define_associated_data({k: v.public_key() for k, v in rsa.items()})
    return yaml.load(zlib.decompress(aes_decrypt(encrypted['data'], ad, **kw)))


def _save_data(**data):
    file = io.BytesIO()
    tar = tarfile.open(mode='w:bz2', fileobj=file)
    for k, v in data.items():
        write_tar(tar, k, yaml.safe_dump(v).encode())
    return tar, file


def save_data(
        output_folder, timestamp, ta_id, dice_report, encrypted_data,
        output_file=None, input_file=None):
    kw = dict(
        ta_id=ta_id, dice_report=dice_report, encrypted_data=encrypted_data
    )

    from co2mpas.batch import default_output_file_name
    i = 0
    while True:
        fpath = default_output_file_name(
            output_folder, ta_id['vehicle_family_id'],
            '%s-%02d' % (timestamp, i), 'co2mpas.zip'
        )
        if not osp.isfile(fpath):
            break
        i += 1
    name = osp.splitext(osp.basename(fpath))[0]
    import zipfile
    ta_file = _save_data(**kw)[1]
    ta_hash = make_hash(json.dumps(
        kw, default=_json_default, sort_keys=True
    ).encode()).hex()
    with zipfile.ZipFile(fpath, 'w', zipfile.ZIP_DEFLATED) as zf:
        if input_file:
            input_file.seek(0)
            zf.writestr('%s.input.xlsx' % name, input_file.read())
        if output_file:
            output_file.seek(0)
            zf.writestr('%s.output.xlsx' % name, output_file.read())
        ta_file.seek(0)
        zf.writestr('%s.ta' % name, ta_file.read())
        zf.writestr('%s.hash.txt' % name, ta_hash)
    log.info('Written into correlation-report-file(%s)'
             ' hash: %s.' % (fpath, ta_hash))
    return fpath


def load_data(fpath):
    data = []
    with zipfile.ZipFile(fpath) as zf:
        fname = '%s.ta' % osp.splitext(osp.basename(fpath))[0]
        fileobj = io.BytesIO(zf.read(fname))
        with tarfile.open(mode='r:bz2', fileobj=fileobj) as tar:
            for k in ('ta_id', 'dice_report', 'encrypted_data'):
                try:
                    with tar.extractfile(tar.getmember(k)) as f:
                        data.append(yaml.load(f.read()))
                except KeyError:
                    data.append(sh.NONE)
    return data


def define_ta_id(vehicle_family_id, data, report, dice, meta, dice_report,
                 encrypted_data, output_file, input_file, sign_key):
    output_file.seek(0)
    input_file.seek(0)
    key = {
        'vehicle_family_id': vehicle_family_id,
        'parent_vehicle_family_id': dice.get('parent_vehicle_family_id', ''),
        'hash': {
            'inputs': make_hash(json.dumps(
                data, default=_json_default, sort_keys=True
            ).encode()),
            'meta': make_hash(json.dumps(
                meta, default=_json_default, sort_keys=True
            ).encode()),
            'dice': make_hash(json.dumps(
                dice, default=_json_default, sort_keys=True
            ).encode()),
            'outputs': make_hash(json.dumps(
                dict(_filter_data(report)), default=_json_default,
                sort_keys=True
            ).encode()),
            'dice_report': make_hash(json.dumps(
                dice_report, default=_json_default, sort_keys=True
            ).encode()),
            'encrypted_data': make_hash(json.dumps(
                encrypted_data, default=_json_default, sort_keys=True
            ).encode()),
            'output_file': make_hash(output_file.read()),
            'input_file': make_hash(input_file.read()),
        },
        'user_random': secrets.randbelow(100),
        'extension': int(dice.get('extension', False)),
        'bifuel': int(dice.get('bifuel', False)),
        'wltp_retest': dice.get('wltp_retest', '-'),
        'comments': dice.get('comments', ''),
        'atct_family_correction_factor': dice.get(
            'atct_family_correction_factor', 1),
        'fuel_type': _get_fuel(report),
        'dice': dice
    }
    sign_ta_id(key, sign_key)
    return key


def _get_fuel(d):
    k = ('summary', 'results', 'vehicle', 'nedc_h', 'prediction', 'input',
         'fuel_type')
    return sh.are_in_nested_dicts(d, *k) and sh.get_nested_dicts(d, *k)


def extract_dice_report(vehicle_family_id, start_time, report):
    from co2mpas import version
    res = {
        'info': {
            'vehicle_family_id': vehicle_family_id,
            'CO2MPAS_version': version,
            'datetime': start_time.strftime('%Y/%m/%d-%H:%M:%S')
        }
    }

    # deviation
    keys = 'summary', 'comparison', 'prediction'
    if sh.are_in_nested_dicts(report, *keys):
        deviation = 'declared_co2_emission_value', 'prediction_target_ratio'
        for cycle, d in sh.get_nested_dicts(report, *keys).items():
            if sh.are_in_nested_dicts(d, *deviation):
                v = (sh.get_nested_dicts(d, *deviation) - 1) * 100
                sh.get_nested_dicts(res, 'deviation')[cycle] = v

    # gears
    keys = 'summary', 'comparison', 'calibration'
    if sh.are_in_nested_dicts(report, *keys):
        for cycle, d in sh.get_nested_dicts(report, *keys).items():
            if cycle.startswith('wltp_') and sh.are_in_nested_dicts(d, 'gears'):
                sh.get_nested_dicts(res, 'gears')[cycle] = sh.get_nested_dicts(d, 'gears')

    # vehicle
    keys = [('summary', 'results', 'vehicle'), ('prediction', 'output')]
    vehicle = (
        'fuel_type', 'engine_capacity', 'gear_box_type', 'engine_is_turbo',
        'engine_max_power', 'engine_speed_at_max_power', 'delta_state_of_charge'
    )
    if sh.are_in_nested_dicts(report, *keys[0]):
        for cycle, d in sh.get_nested_dicts(report, *keys[0]).items():
            if sh.are_in_nested_dicts(d, *keys[1]):
                v = sh.selector(
                    vehicle, sh.get_nested_dicts(d, *keys[1]),
                    allow_miss=True
                )
                if v:
                    sh.get_nested_dicts(res, 'vehicle', cycle).update(v)

    # declared
    keys = [
        ('summary', 'results', 'declared_co2_emission'),
        ('prediction', 'target', 'declared_co2_emission_value')
    ]
    declared = {}
    if sh.are_in_nested_dicts(report, *keys[0]):
        for cycle, d in sh.get_nested_dicts(report, *keys[0]).items():
            if sh.are_in_nested_dicts(d, *keys[1]):
                declared[cycle] = sh.get_nested_dicts(d, *keys[1])

    for k in 'hl':
        i, j = 'wltp_%s' % k, 'nedc_%s' % k
        k = 'declared_wltp_%s_vs_declared_nedc_%s' % (k, k)
        if i in declared and j in declared:
            sh.get_nested_dicts(res, 'ratios')[k] = declared[i] / declared[j]

    # corrected
    keys = [
        ('summary', 'results', 'corrected_co2_emission'),
        ('prediction', 'target', 'corrected_co2_emission_value')
    ]
    corrected = {}
    if sh.are_in_nested_dicts(report, *keys[0]):
        for cycle, d in sh.get_nested_dicts(report, *keys[0]).items():
            if sh.are_in_nested_dicts(d, *keys[1]):
                corrected[cycle] = sh.get_nested_dicts(d, *keys[1])
    for k in 'hl':
        i = 'wltp_%s' % k
        k = 'declared_wltp_%s_vs_corrected_wltp_%s' % (k, k)
        if i in declared and i in corrected:
            sh.get_nested_dicts(res, 'ratios')[k] = declared[i] / corrected[i]

    # model scores
    keys = 'data', 'calibration', 'model_scores'
    model_scores = 'model_selections', 'param_selections', 'score_by_model', \
                   'scores'
    if sh.are_in_nested_dicts(report, *keys):
        sh.get_nested_dicts(res, 'model_scores').update(sh.selector(
            model_scores, sh.get_nested_dicts(report, *keys), allow_miss=True
        ))

    res = copy.deepcopy(res)
    for k, v in list(stack(res)):
        if isinstance(v, np.generic):
            get_nested(res, *k[:-1])[k[-1]] = v.item()

    return res


def stack(d, key=()):
    it = ()
    if hasattr(d, 'items'):
        it = d.items()
    elif isinstance(d, list):
        it = enumerate(d)
    else:
        yield key, d
    for k, v in it:
        yield from stack(v, key=key + (k,))


def get_nested(d, *keys):
    if keys:
        return get_nested(d[keys[0]], *keys[1:])
    return d


@functools.lru_cache()
def crypto():
    dsp = sh.Dispatcher()

    dsp.add_function(
        function=sh.bypass,
        inputs=['data', 'meta'],
        outputs=['data2encrypt']
    )

    dsp.add_function(
        function=encrypt_data,
        inputs=['data2encrypt', 'path_keys'],
        outputs=['encrypted_data']
    )

    dsp.add_data('ta_id', filters=[verify_ta_id])

    dsp.add_function(
        function=define_ta_id,
        inputs=['vehicle_family_id', 'data', 'report', 'dice', 'meta',
                'dice_report', 'encrypted_data', 'output_file', 'input_file',
                'sign_key'],
        outputs=['ta_id']
    )

    dsp.add_function(
        function=extract_dice_report,
        inputs=['vehicle_family_id', 'start_time', 'report'],
        outputs=['dice_report']
    )

    dsp.add_function(
        function=save_data,
        inputs=['output_folder', 'timestamp', 'ta_id', 'dice_report',
                'encrypted_data', 'output_file', 'input_file'],
        outputs=['ta_file']
    )

    dsp.add_function(
        function=load_data,
        inputs=['ta_file'],
        outputs=['ta_id', 'dice_report', 'encrypted_data']
    )

    dsp.add_data('passwords', None)

    dsp.add_function(
        function=decrypt_data,
        inputs=['encrypted_data', 'path_keys', 'passwords'],
        outputs=['data2encrypt']
    )

    dsp.add_function(
        function=sh.bypass,
        inputs=['data2encrypt'],
        outputs=['data', 'meta']
    )

    return dsp


def write_ta_output():
    func = sh.SubDispatchFunction(
        crypto(),
        'write_ta_output',
        inputs=['path_keys', 'vehicle_family_id', 'sign_key',
                'start_time', 'timestamp', 'data', 'meta', 'dice', 'report',
                'output_folder', 'output_file', 'input_file'],
        outputs=['ta_file']
    )

    return func


def define_decrypt_function(path_keys, passwords=None):
    dsp = crypto()
    sol = dsp({'path_keys': path_keys, 'passwords': passwords})
    dsp = dsp.shrink_dsp(
        inputs=['ta_file'] + sorted(sol),
        outputs=['ta_id', 'dice_report', 'data', 'meta']
    )
    for k, v in sol.items():
        if k in dsp.nodes:
            dsp.add_data(k, v)

    func = sh.SubDispatchFunction(
        dsp, 'decrypt', ['ta_file'], ['ta_id', 'dice_report', 'data', 'meta']
    )

    func.output_type = 'all'

    return func


if __name__ == '__main__':
    import glob
    import tqdm
    import zipfile
    passwords = ('p_secret', 'p_server')
    #generate_keys('.', passwords)
    func = define_decrypt_function('secret.co2mpas.keys', passwords)
    res = {}
    for k in ('dimi', 'vins', 'vins1'):
        for fpath in tqdm.tqdm(glob.glob('./output/demos/%s/*co2mpas.zip' % k)):
            fname = '%s.ta' % osp.splitext(osp.basename(fpath))[0]
            sh.get_nested_dicts(
                res, '-'.join(fname.split('-')[1:])[:-11], k
            ).update(func(fpath))

    for fname, data in res.items():
        data = [dict(sh.stack_nested_keys(v['ta_id']))
                for k, v in sorted(data.items())]
        for k in data[0]:
            r = [d[k] for d in data]
            s = {json.dumps(v, default=_json_default, sort_keys=1) for v in r}
            try:
                assert len(s) == 1
            except AssertionError:
                print(fname, len(s), k, s)
    c = 0
