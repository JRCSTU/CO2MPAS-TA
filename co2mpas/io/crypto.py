import os
import yaml
import zlib
import os.path as osp
import schedula as sh
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_OAEP


def load_public_key(pem_file):
    with open(pem_file, "rb") as file:
        return PKCS1_OAEP.new(RSA.importKey(file.read()))


def load_private_key(pem_file=None):
    if pem_file is None:
        try:
            pem_file = os.environ['PRIVATE_PEM']
        except KeyError:
            return sh.NONE

    with open(pem_file, "rb") as file:
        return PKCS1_OAEP.new(RSA.importKey(file.read()))


def encrypt_raw_data(raw_data, public_key):
    data = zlib.compress(yaml.dump(raw_data).encode())
    chunk_size, encrypted, encrypt = 470, b"", public_key.encrypt
    n = len(data) - len(data) // chunk_size * chunk_size
    if n:
        data += b' ' * (chunk_size - n)

    for i, j in sh.pairwise(range(0, len(data) + 1, chunk_size)):
        encrypted += encrypt(data[i:j])

    return encrypted


def decrypt_raw_data(encrypted_data, private_key):
    chunk_size, data, decrypt = 512, b"", private_key.decrypt
    for i, j in sh.pairwise(range(0, len(encrypted_data) + 1, chunk_size)):
        data += decrypt(encrypted_data[i:j])
    return yaml.load(zlib.decompress(data))


def crypto():
    dsp = sh.Dispatcher()
    dsp.add_data('public_pem', osp.join(osp.dirname(__file__), 'public.pem'))
    dsp.add_function(
        function=load_public_key, inputs=['public_pem'], outputs=['public_key']
    )
    dsp.add_data('private_pem', None)
    dsp.add_function(
        function=load_private_key,
        inputs=['private_pem'], outputs=['private_key']
    )
    dsp.add_function(
        function=encrypt_raw_data,
        inputs=['raw_data', 'public_key'],
        outputs=['encrypted_data']
    )
    dsp.add_function(
        function=decrypt_raw_data,
        inputs=['encrypted_data', 'private_key'],
        outputs=['raw_data']
    )
    return dsp
