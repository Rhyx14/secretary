import base64
def encode_base64_str(str:str) -> str:
    '''
    encode string by base64
    '''
    return base64.b64encode(str.encode('utf-8')).decode('utf-8')

def decode_base64_str(base64_str:str) -> str:
    '''
    decode base64 string
    '''
    return base64.b64decode(base64_str.encode('utf-8')).decode('utf-8')