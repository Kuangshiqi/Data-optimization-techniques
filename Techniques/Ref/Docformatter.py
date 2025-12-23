import subprocess
import tempfile
import os

def docformatter_clean(code: str) -> str:
    temp_filename = None
    try:
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".py") as temp_file:
            temp_file.write(code)
            temp_file.flush()
            temp_filename = temp_file.name
        subprocess.run(["docformatter", "--in-place", temp_filename], check=True)
        with open(temp_filename, 'r') as formatted_file:
            formatted_code = formatted_file.read()
        return formatted_code
    except Exception:
        return code
    finally:
        if temp_filename and os.path.exists(temp_filename):
            os.remove(temp_filename)