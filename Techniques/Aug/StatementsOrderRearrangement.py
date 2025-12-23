import ast
import copy
import astor
def have_common_variable(s1, s2):
    vars_s1 = {node.id for node in ast.walk(s1) if isinstance(node, ast.Name)}
    vars_s2 = {node.id for node in ast.walk(s2) if isinstance(node, ast.Name)}
    return not vars_s1.isdisjoint(vars_s2)

def rearrange_statements_in_block(node):
    for i in range(len(node.body) - 1):
        s1 = node.body[i]
        s2 = node.body[i + 1]
        if not have_common_variable(s1, s2):
            st = copy.deepcopy(s1)
            node.body[i] = s2
            node.body[i + 1] = st
    return node

def rearrange_statements(node):
    if isinstance(node, ast.FunctionDef):
        rearrange_statements_in_block(node)

    elif isinstance(node, (ast.If, ast.For, ast.While)):
        rearrange_statements_in_block(node)

    for child in ast.iter_child_nodes(node):
        rearrange_statements(child)

def rearrange_code(source_code):
    tree = ast.parse(source_code)
    rearrange_statements(tree)
    new_source_code = astor.to_source(tree)
    return new_source_code


# source_code = """
# def example_function(x):
#     a = 1
#     b = 2
#     while a > 0:
#         x = 1
#         y = 0
#     c = a + b
#     return c
# """
#
# modified_code = rearrange_code(source_code)
# print(modified_code)
