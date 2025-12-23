import autopep8

def autopep8_clean(code: str) -> str:
    formatted_code = autopep8.fix_code(code, options={'aggressive': 2})
    return formatted_code
