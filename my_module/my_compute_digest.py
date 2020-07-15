import hashlib

def CalcSha1_str(str_original):
    sha1obj = hashlib.sha1()
    sha1obj.update(str_original.encode('utf-8'))
    hash = sha1obj.hexdigest()

    return hash


if __name__ == '__main__':

    password = 'jsiec'
    encrypt_password = CalcSha1_str(password)
    print(encrypt_password)

    password = 'eyepacs'
    encrypt_password = CalcSha1_str(password)
    print(encrypt_password)
