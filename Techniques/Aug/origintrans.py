import ast
import astor

def origin_trans(source_code):
    tree = ast.parse(source_code)
    new_source_code = astor.to_source(tree)

    return new_source_code

