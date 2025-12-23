import ast
import astor

class InfixExpressionDivider(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.counter = 1
        self.temp_assignments = []

    def generate_temp_name(self):
        temp_name = f"temp_{self.counter}"
        self.counter += 1
        return temp_name

    def visit_BinOp(self, node):
        """
        Splits infix expressions into temporary variables.
        """
        self.generic_visit(node)
        temp_name = self.generate_temp_name()
        temp_var = ast.Name(id=temp_name, ctx=ast.Load())
        assignment = ast.Assign(targets=[ast.Name(id=temp_name, ctx=ast.Store())], value=node)
        self.temp_assignments.append(assignment)

        return temp_var

    def visit_FunctionDef(self, node):
        """
        Visit a function definition, transforming its body to split infix expressions.
        """
        new_body = []
        for stmt in node.body:
            self.temp_assignments = []
            transformed_stmt = self.visit(stmt)
            new_body.extend(self.temp_assignments)

            if transformed_stmt is not None:
                new_body.append(transformed_stmt)

        node.body = new_body
        return node

def divide_infix_expressions(source_code):
    """
    Takes Python code as input, splits infix expressions, and returns the modified code.
    """
    tree = ast.parse(source_code)
    divider = InfixExpressionDivider()
    transformed_tree = divider.visit(tree)
    new_source_code = astor.to_source(transformed_tree)
    if new_source_code == source_code:
        return " "
    else:
        return new_source_code

# if __name__ == "__main__":
#     input_code = """
# def example_function(x):
#     y = 1 + 2 * x
#     return y
#     """
#
#     transformed_code = divide_infix_expressions(input_code)
#     print("Original Code:")
#     print(input_code)
#     print("\nTransformed Code:")
#     print(transformed_code)
