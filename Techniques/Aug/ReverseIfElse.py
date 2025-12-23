import ast
import astor


class ReverseIfElseTransformer(ast.NodeTransformer):
    def visit_If(self, node):
        node = self.generic_visit(node)
        if node.orelse:
            node.body, node.orelse = node.orelse, node.body
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)

        return node


def reverse_if_else(source_code):
    tree = ast.parse(source_code)
    transformer = ReverseIfElseTransformer()
    new_tree = transformer.visit(tree)

    new_source_code = astor.to_source(new_tree)
    return new_source_code



# source_code = '''
# x = 10
# if x > 5:
#     print("x is greater than 5")
# else:
#     print("x is 5 or less")
# '''
#
# reversed_code = reverse_if_else(source_code)
# print(reversed_code)
