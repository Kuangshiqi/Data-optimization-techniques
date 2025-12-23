import subprocess


def autoflake_clean(code: str) -> str:
    try:
        process = subprocess.Popen(
            ["autoflake", "--remove-all-unused-imports", "--remove-unused-variables", "--stdin-display-name", "temp.py", "-"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        cleaned_code, error = process.communicate(input=code)
        if process.returncode != 0:
            print(f"Error cleaning code with autoflake: {error}")
            return code
        return cleaned_code
    except Exception as e:
        print(f"Error during autoflake processing: {e}")
        return code
