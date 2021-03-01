#import os
#import pickle
import hashlib
import base58
import binascii
#import multiprocessing
#from ellipticcurve.privateKey import PrivateKey
#import itertools
import exrex
import urllib.request, json 
from urllib.error import HTTPError



chars = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
fragments = ['W5M0MpCehiHzreSzNTczkc9d','MaNuD98sHPpAJir']

def sha256(data):
    digest = hashlib.new("sha256")
    digest.update(data)
    return digest.digest()

def wifToPK(wif):
    print(len(wif))
    bites = wif.encode()
    print(bites)
    decode = base58.b58decode(bites)
    print(decode)
    hexify = binascii.hexlify(decode)
    print(hexify)
    return hexify[2:-8]

class Point:
    def __init__(self,
        x=0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
        y=0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
        p=2**256 - 2**32 - 2**9 - 2**8 - 2**7 - 2**6 - 2**4 - 1):
        self.x = x
        self.y = y
        self.p = p

    def __add__(self, other):
        return self.__radd__(other)

    def __mul__(self, other):
        return self.__rmul__(other)

    def __rmul__(self, other):
        n = self
        q = None

        for i in range(256):
            if other & (1 << i):
                q = q + n
            n = n + n

        return q

    def __radd__(self, other):
        if other is None:
            return self
        x1 = other.x
        y1 = other.y
        x2 = self.x
        y2 = self.y
        p = self.p

        if self == other:
            l = pow(2 * y2 % p, p-2, p) * (3 * x2 * x2) % p
        else:
            l = pow(x1 - x2, p-2, p) * (y1 - y2) % p

        newX = (l ** 2 - x2 - x1) % p
        newY = (l * x2 - l * newX - y2) % p

        return Point(newX, newY)

    def toBytes(self):
        x = self.x.to_bytes(32, "big")
        y = self.y.to_bytes(32, "big")
        return b"\x04" + x + y


def getPublicKey(privkey):
    SPEC256k1 = Point()
    pk = int.from_bytes(privkey, "big")
    hash160 = ripemd160(sha256((SPEC256k1 * pk).toBytes()))
    address = b"\x00" + hash160

    address = b58(address + sha256(sha256(address))[:4])
    return address

def checkAddressForBalance(addr):
    req = urllib.request.Request(url="https://blockchain.info/address/"+addr+"?format=json")
    try:
        handler = urllib.request.urlopen(req)
        data = json.loads(handler.read().decode())
        if data['final_balance'] > 0:
            return data['final_balance']
    except HTTPError as e:
        content = e.read()
        return 0
    


# Try making differnt leads

lead_start = "L12w"
lead_end = "ftTD"
combos = exrex.generate('['+chars+']{5}')
for perm in combos:
    lead = lead_start + perm  + lead_end
    # Try ordering the two fragments differently
    guess1 = lead + fragments[0] + fragments[1]
    guess2 = lead + fragments[1] + fragments[0]
    guesses = [guess1, guess2]
    for guess in guesses:
        #privkey = wifToPK(guess)
        #addr = getPubKey(privkey)
        print("Trying WIF: " + guess)
        #print("Private Key: "+privkey )
        #print("Address: "+ addr)
        
        # VALID ADDRESS WITH BALANCE
        addr = "3EoJMDs79b3ztkYvj2E1D98eHnZyCSQKso"
        # INVALID ADDRESS
        #addr = "3EoJMDs79b3ztkYvj2E1D98eHnZyCSQKsR"
        balance = checkAddressForBalance(addr)
        print(balance)



