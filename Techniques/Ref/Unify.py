import os
import subprocess
import tempfile

def unify_clean(code: str) -> str:
    try:
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".py") as temp_file:
            temp_file.write(code)
            temp_file.flush()
            temp_filename = temp_file.name

        subprocess.run(["unify", "--in-place", temp_filename], check=True)

        with open(temp_filename, 'r') as formatted_file:
            cleaned_code = formatted_file.read()

        return cleaned_code
    except Exception as e:
        print(f"Error cleaning code with unify: {e}")
        return code
    finally:
        if temp_filename and os.path.exists(temp_filename):
            os.remove(temp_filename)
