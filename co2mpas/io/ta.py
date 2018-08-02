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
                key = serialization.load_pem_public_key(
                    f.read(), default_backend()
                )
                keys[k[-1]] = key
    return keys


def load_private_RSA_keys(fpath, passwords=None):
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives import serialization
    keys = {}

    with tarfile.open(fpath, 'r') as tar:
        it = itertools.product(('private',), ('secret', 'server'))
        for k, p in itertools.zip_longest(it, passwords or ()):
            p = p is not None and p.encode() or None
            with tar.extractfile(tar.getmember('%s/%s.pem' % k[::-1])) as f:
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


def encrypt_data(encrypt_inputs, data, path_keys):
    if encrypt_inputs and osp.isfile(path_keys):
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


def _save_data(fpath, **data):
    with tarfile.open(fpath, 'w') as tar:
        for k, v in data.items():
            write_tar(tar, k, yaml.safe_dump(v).encode())
    log.info('Written into ta-file(%s)...' % fpath)
    return fpath


def save_data(
        output_folder, timestamp, ta_id, dice_report, encrypted_data=None):
    kw = dict(ta_id=ta_id, dice_report=dice_report)
    if encrypted_data is not None:
        kw['encrypted_data'] = encrypted_data
    from co2mpas.batch import default_output_file_name
    tar_file = default_output_file_name(
        output_folder, ta_id['vehicle_family_id'], timestamp, 'co2mpas.ta'
    )
    return _save_data(tar_file, **kw)


def load_data(fpath):
    data = []
    with tarfile.open(fpath, 'r') as tar:
        for k in ('ta_id', 'dice_report', 'encrypted_data'):
            try:
                with tar.extractfile(tar.getmember(k)) as f:
                    data.append(yaml.load(f.read()))
            except KeyError:
                data.append(sh.NONE)
    return data


def define_ta_id(vehicle_family_id, data, report):
    key = {
        'vehicle_family_id': vehicle_family_id,
        'hash': {
            'inputs': make_hash(json.dumps(
                data, default=_json_default, sort_keys=True
            ).encode()),
            'outputs': make_hash(json.dumps(
                dict(_filter_data(report)), default=_json_default,
                sort_keys=True
            ).encode())
        }
    }
    return key


def extract_dice_report(encrypt_inputs, vehicle_family_id, start_time, report):
    from co2mpas import version
    res = {
        'info': {
            'encrypt_inputs': encrypt_inputs,
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

    # vehicle
    keys = [('summary', 'results', 'vehicle'), ('prediction', 'output')]
    vehicle = 'fuel_type', 'engine_capacity', 'gear_box_type', 'engine_is_turbo'
    if sh.are_in_nested_dicts(report, *keys[0]):
        for cycle, d in sh.get_nested_dicts(report, *keys[0]).items():
            if sh.are_in_nested_dicts(d, *keys[1]):
                v = sh.selector(
                    vehicle, sh.get_nested_dicts(d, *keys[1]),
                    allow_miss=True
                )
                if v:
                    sh.get_nested_dicts(res, 'vehicle', cycle).update(v)

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
            sh.get_nested_dicts(res, *k[:-1])[k[-1]] = v.item()

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
        inputs=['encrypt_inputs', 'data2encrypt', 'path_keys'],
        outputs=['encrypted_data']
    )

    dsp.add_function(
        function=define_ta_id,
        inputs=['vehicle_family_id', 'data', 'report'],
        outputs=['ta_id']
    )

    dsp.add_function(
        function=extract_dice_report,
        inputs=['encrypt_inputs', 'vehicle_family_id', 'start_time', 'report'],
        outputs=['dice_report']
    )

    dsp.add_function(
        function=save_data,
        inputs=['output_folder', 'timestamp', 'ta_id', 'dice_report',
                'encrypted_data'],
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
        inputs=['encrypt_inputs', 'path_keys', 'vehicle_family_id',
                'start_time', 'timestamp', 'data', 'meta', 'report',
                'output_folder'],
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
    passwords = None  # ('12345', '67890')
    # generate_keys('.', passwords)
    func = define_decrypt_function('secret.co2mpas.keys', passwords)
    r = func('20180801_185906-IP-TEST_1234567890-AAA-1.co2mpas.ta')
    r = 0
