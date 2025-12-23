import ast
import astor

class LocalVarRenamer(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.var_mapping = {}
        self.counter = 1

    def generate_temp_name(self):
        temp_name = f"temp_{self.counter}"
        self.counter += 1
        return temp_name

    def visit_FunctionDef(self, node):
        # Reset the variable mapping for each function
        self.var_mapping = {}
        self.counter = 1

        # Rename function arguments
        for arg in node.args.args:
            if arg.arg not in self.var_mapping:
                self.var_mapping[arg.arg] = self.generate_temp_name()
            arg.arg = self.var_mapping[arg.arg]

        self.generic_visit(node)  # Visit the function body
        return node

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Load, ast.Del)):
            if node.id not in self.var_mapping:
                self.var_mapping[node.id] = self.generate_temp_name()
            node.id = self.var_mapping[node.id]
        return node


def rename_local_variables(source_code):
    """
    Takes Python code as input, renames local variables, and returns the modified code.
    """
    tree = ast.parse(source_code)
    renamer = LocalVarRenamer()
    transformed_tree = renamer.visit(tree)
    new_source_code = astor.to_source(transformed_tree)
    if new_source_code == source_code:
        return " "
    else:
        return new_source_code

# if __name__ == "__main__":
#     input_code = "def set_left_most_unset_bit(n): \r\n    if not (n & (n + 1)): \r\n        return n \r\n    pos, temp, count = 0, n, 0 \r\n    while temp: \r\n        if not (temp & 1): \r\n            pos = count      \r\n        count += 1; temp>>=1\r\n    return (n | (1 << (pos))) "
#
#     renamed_code = rename_local_variables(input_code)
#     print("Original Code:")
#     print(input_code)
#     print("\nRenamed Code:")
#     print(renamed_code)