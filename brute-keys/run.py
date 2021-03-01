#import os
#import pickle
import hashlib
import base58
import binascii
#import multiprocessing
#from ellipticcurve.privateKey import PrivateKey
#import itertools
import exrex

chars = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
fragments = ['W5M0MpCehiHzreSzNTczkc9d','MaNuD98sHPpAJir']

def wifToPK(wif):
    print(len(wif))
    bites = wif.encode()
    print(bites)
    decode = base58.b58decode(bites)
    print(decode)
    hexify = binascii.hexlify(decode)
    print(hexify)
    return hexify[2:-8]


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
        pk = wifToPK(guess)
        print("Trying WIF: " + guess)
        print("Private Key: "+pk )



