# name this file 'solutions.py'
"""Volume II Lab 3: Public Key Encryption (RSA)
Lehner White
Sept. 17, 2015
Math 321
"""
import rsa_tools as rsa
import numpy as np
import math
from Crypto.PublicKey import RSA

def EE(a,b):
    a0 = a
    b0 = b
    x0, x = 1, 0
    y0, y = 0, 1
    while b != 0:
        q=a/b
        x, x0 = x0 - q*x, x
        y, y0 = y0 - q*y, y
        a, b = b, a%b
    return x0,b0

# Problem 1: Implement the following RSA system.
class myRSA(object):
    """RSA String Encryption System. Do not use any external modules except for
    'rsa_tools' and your implementation of the Extended Euclidean Algorithm.
    
    Attributes:
        public_key (tup): the RSA key that is available to everyone, of the
            form (e, n). Used only in encryption.
        _private_key (tup, hidden): the secret RSA key, of the form (d, n).
            Used only in decryption.
    
    Examples:
        >>> r = myRSA()
        >>> r.generate_keys(1000003,608609,1234567891)
        >>> print(r.public_key)
        (1234567891, 608610825827)
        
        >>> r.decrypt(r.encrypt("SECRET MESSAGE"))
        'SECRET MESSAGE'
        
        >>> s = myRSA()
        >>> s.generate_keys(287117,104729,610639)
        >>> s.decrypt(r.encrypt("SECRET MESSAGE",s.public_key))
        'SECRET MESSAGE'
    """
    def __init__(self):
        """Initialize public and private key variables."""
        self.public_key = None
        self._private_key = None
    
    def generate_keys(self, p, q, e):
        """Create a pair of RSA keys.
        
        Inputs:
            p (int): A large prime.
            q (int): A second large prime .
            e (int): The encryption exponent. 
        
        Returns:
            Set the public_key and _private_key attributes.
        """
        n = p*q
        phi_n = (p-1)*(q-1)
        d_prime, temp = EE(e, phi_n)
        d = d_prime % phi_n
        self.public_key = (e,n)
        self._private_key = (d,n)
        pass
    
    def encrypt(self, message, key=None):
        """Encrypt 'message' with a public key and return its encryption as a
        list of integers. If no key is provided, use the 'public_key' attribute
        to encrypt the message.
        
        Inputs:
            message (str): the message to be encrypted.
            key (int tup, opt): the public key to be used in the encryption.
                 Defaults to 'None', in which case 'public_key' is used.
        """
        if key == None:
            key = self.public_key
        
        messages = rsa.partition(message, rsa.string_size(key[1]), '@')
        ciphertext = [] 
        for mess in messages:
            mess = rsa.string_to_int(mess)
            mess = long(mess)
            mess = pow(mess,key[0],key[1])
            ciphertext.append(mess)
        return ciphertext
    
    def decrypt(self, ciphertext):
        """Decrypt 'ciphertext' with the private key and return its decryption
        as a single string. You may assume that the format of 'ciphertext' is
        the same as the output of the encrypt() function. Remember to strip off
        the fill value used in rsa_tools.partition().
        """
        key = self._private_key
        out = []
        for num in ciphertext:
            num = pow(int(num), key[0], key[1])
            num = rsa.int_to_string(num)
            out.append(num)
        
        for l in out[-1]:
            if l == '@':
                out[-1] = out[-1][:-1]
        
        message = ''
        for mess in out:
            message += mess
        return message

"""
s = myRSA()
s.generate_keys(287117,104729,610639)
test_encrypt = s.encrypt("SECRET MESSAGE")
test_out = s.decrypt(test_encrypt)
print test_out

test = 'Abstract Anecdotal evidence has suggested increased fertility rates resulting from catastrophic events in an area. In this paper, we measure this fertility effect using storm advisory data and fertility data for the Atlantic and Gulf-coast counties of the USA. We find that low-severity storm advisories are associated with a positive and significant fertility effect and that high-severity advisories have a significant negative fertility effect.'

s.generate_keys(1000003,608609,1234567891)
test_encrypt = s.encrypt(test)
test_out = s.decrypt(test_encrypt)
print test_out
"""

# Problem 2: Partially test the myRSA class with this function.
#   Use Exceptions to indicate inappropriate arguments or test failure.
def test_myRSA(message, p, q, e):
    """Create a 'myRSA' object. Generate a pair of keys using 'p', 'q', and
    'e'. Encrypt the message, then decrypt the encryption. If the decryption
    is not exactly the same as the original message, raise a ValueError with
    error message "decrypt(encrypt(message)) failed."
    
    If 'message' is not a string, raise a TypeError with error message
    "message must be a string."
    
    If any of p, q, or e are not integers, raise a TypeError with error
    message "p, q, and e must be integers."
    
    Inputs:
        message (str): a message to be encrypted and decrypted.
        p (int): A large prime for key generation.
        q (int): A second large prime for key generation.
        e (int): The encryption exponent.
        
    Returns:
        True if no exception is raised.
    """
    # A NotImplementedError usually indicates that a class method will be
    #   overwritten by children classes, or that the method or function is
    #   still under construction.
    
    if not isinstance(message, str):
        raise TypeError("message must be a string.")
    if not isinstance(p, int) or not isinstance(q,int) or not isinstance(e,int):
        raise TypeError("p, q, and e must be integers.")

    R = myRSA()
    R.generate_keys(p,q,e)
    output = R.decrypt(R.encrypt(message))
    
    if message != output:
        raise ValueError("decrypt(encrypt(message)) failed.")
    
    return True

#print test_myRSA("SECRET MESSAGE", 1000003, 608609, 1234567891) 

# Problem 3: Fermat's test for primality.
def is_prime(n):
    """Use Fermat's test for primality to see if 'n' is probably prime.
    Run the test at most five times, using integers randomly chosen from
    [2, n-1] as possible witnesses. If a witness number is found, return the
    number of tries it took to find the witness. If no witness number is found
    after five tries, return 0.
    
    Inputs:
        n (int): the candidate for primality.
    
    Returns:
        The number of tries it took to find a witness number, up to 5
        (or 0 if no witnesses were found).
    
    """
    for i in range(5):
        a = np.random.randint(2,n)
        if 1 != pow (a, n-1, n):
            return (i+1)
    return 0
#Depending on the try, typically somewhere between 1 and 5 times through the function. 
"""
test = 0
count = 0
while test == 0:
    test = is_prime(340561)
    count += 1
print count
"""

# Problem 4: Implement the following RSA system using PyCrypto.
class PyCrypto(object):
    """RSA String Encryption System. Do not use any external modules except for
    those found in the 'Crypto' package.
    Attributes:
        _keypair (RSA obj, hidden): the RSA key (both public and private).
            Facilitates encrypt() and decrypt().
        public_key (str): A sharable string representation of the public key.
    Examples:
        >>> p = PyCrypto()
        >>> p.decrypt(p.encrypt("SECRET MESSAGE"))
        'SECRET MESSAGE'
        >>> print(p.public_key)
        -----BEGIN PUBLIC KEY-----
        MIIBIjANBgkqhkiG9w0BAQ...
        ...
        ...HwIDAQAB
        -----END PUBLIC KEY-----
        >>> q = PyCrypto()
        >>> q.decrypt(p.encrypt("SECRET MESSAGE",q.public_key))
        'SECRET MESSAGE'
    """
    def __init__(self):
        """Initialize the _keypair and public_key attributes."""
        self._keypair = RSA.generate(2048)
        public_key = self._keypair.publickey()
        self.public_key = public_key.exportKey()
    def encrypt(self, message, key=None):
        """Encrypt 'message' with a public key and return its encryption. If
        no key is provided, use the '_keypair' attribute to encrypt 'message'.
        Inputs:
            message (str): the message to be encrypted.
            key (str, opt): the string representation of the public key to be
                used in the encryption. Defaults to 'None', in which case
                '_keypair' is used to encrypt the message.
        """
        if key==None:
            key = self.public_key
        ciphertext = self._keypair.encrypt(message, key)
        return ciphertext
    def decrypt(self, ciphertext):
        """Decrypt 'ciphertext' with '_keypair' and return the decryption."""
        decrypt = self._keypair.decrypt(ciphertext)
        return decrypt

#newR = PyCrypto()
#print newR.public_key
# ============================== END OF FILE ============================== #
