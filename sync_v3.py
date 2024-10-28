import os
import hashlib
import base64
import requests
import cbor2
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding as asym_padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
# import io
import struct

method_flags = {
    "GET": 1,
    "POST": 2,
    "PUT": 3,
    "DELETE": 4,
    "HEAD": 5,
    "OPTIONS": 6,
    "CONNECT": 7,
    "PATCH": 8,
    "TRACE": 9,
}

relay_fetch = {"ssls": {}}


class ConnectProcess:
    def __init__(self, target, domain, protocol):
        self._sym_key = None
        self._raw_sym_key = None
        self.target = target
        self.domain = domain
        self.protocol = protocol

    def generate_ssl(self):
        # Generate AES-CTR key
        self._sym_key = os.urandom(32)
        self._raw_sym_key = self._sym_key

    def verify_pb_key(self):
        # print(self.target)
        method_buffer = bytes([method_flags["GET"]])
        # print("before post sync")
        pb_key_req = requests.post(
            f"{self.protocol}://{self.domain}/rl/{self.target}/sync", data=method_buffer, verify=False)
        # print(pb_key_req)
        _pb_key = pb_key_req.content
        # print("pb_key", _pb_key)
        _id = hashlib.sha256(_pb_key).digest()
        _id = base64.urlsafe_b64encode(_id).decode('utf-8').rstrip("=")
        # print("_id", _id)
        if _id != self.target:
            print("Not matched")
            return False

        relay_fetch["ssls"][self.target] = serialization.load_der_public_key(
            _pb_key, backend=default_backend())
        return True

    def encrypter(self, headers, data, public_key):
        counter = os.urandom(16)

        encrypt_sym_key = public_key.encrypt(
            self._raw_sym_key,
            asym_padding.OAEP(
                mgf=asym_padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        _headers = cbor2.dumps(headers) if headers else b""
        # print("_headerslength", len(_headers))

        _headers_index = bytearray(8)

        struct.pack_into('>H', _headers_index,0, len(_headers))
        # print("buffer_index", list(_headers_index))

        body_data = _headers_index + _headers
        # print('before_encrypt',type(data))
        data_encode = cbor2.dumps(data)

        if data is not None:
            body_data += data_encode
        
        # print('body_data', list(body_data))

        encryptor = Cipher(
            algorithms.AES(self._sym_key),
            modes.CTR(counter),
            backend=default_backend()
        ).encryptor()

        _body = encryptor.update(body_data) + encryptor.finalize()

        # print("_body", _body)
        # print("_body_len", len(_body))

        return encrypt_sym_key + counter + _body

    def decrypter(self, data):
        # print("decrypter", list(data))
        if not data:
            return None, None

        _counter = data[:16]
        encrypted_body = data[16:]

        decryptor = Cipher(
            algorithms.AES(self._sym_key),
            modes.CTR(_counter),
            backend=default_backend()
        ).decryptor()

        decrypted_data = decryptor.update(
            encrypted_body) + decryptor.finalize()

        # header_length = int.from_bytes(decrypted_data[:8], byteorder='big')
        
        header_length = struct.unpack_from('>H',buffer=decrypted_data[:8])[0]
        
        _headers = {}
        if header_length > 0:
            _headers = cbor2.loads(decrypted_data[8:8+header_length])

        _body = None
        if len(decrypted_data) > 8 + header_length:
            _body = cbor2.loads(decrypted_data[8 + header_length:])

        return _headers, _body

    def fetch_with_target(self, url, method, data=None):
        method_buffer = bytes([method_flags[method]])

        encrypted_data = self.encrypter(
            # {"key": "value"},
            False,
            data if method not in ["GET", "HEAD"] else None,
            relay_fetch["ssls"][self.target]
        )

        try:
            if method == "POST":
                req_enter = requests.post(
                    f"{self.protocol}://{self.domain}/rl/{self.target}/{url}",
                    data=method_buffer + encrypted_data,
                    verify=False
                )
                # print("req_enter", encrypted_data)

            elif method == "PATCH":
                req_out = requests.patch(
                    f"{self.protocol}://{self.domain}/rl/{self.target}/{url}",
                    data=method_buffer + encrypted_data,
                    verify=False
                )
                # print("req_out", req_out)
        except Exception as e:
            print(e)

        # response = requests.post(
        #     f"https://relay.nexivil.com/rl/{self.target}/{url}",
        #     data=method_buffer + encrypted_data,
        #     verify=False
        # )

        
        # _headers, _body = self.decrypter(response.content)

        # if response.ok:
        #     return _headers, _body
        # else:
        #     print(f"Error: {response.status_code}")
        #     raise Exception({"status": response.status_code,
        #                     "headers": _headers, "body": _body})
