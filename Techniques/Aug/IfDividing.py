import ast
import astor


class IfDividingTransformer(ast.NodeTransformer):
    def visit_If(self, node):
        if isinstance(node.test, ast.BoolOp) and isinstance(node.test.op, ast.And):

            conditions = node.test.values
            new_if_stmt = None

            for condition in reversed(conditions):
                if new_if_stmt is None:
                    new_if_stmt = ast.If(test=condition, body=node.body, orelse=node.orelse)
                else:
                    new_if_stmt = ast.If(test=condition, body=[new_if_stmt], orelse=[])

            return new_if_stmt

        elif isinstance(node.test, ast.BoolOp) and isinstance(node.test.op, ast.Or):
            conditions = node.test.values
            new_if_stmt = None

            for condition in conditions:
                if new_if_stmt is None:
                    new_if_stmt = ast.If(test=condition, body=node.body, orelse=[])
                else:
                    new_if_stmt = ast.If(test=condition, body=node.body, orelse=[new_if_stmt])

            return new_if_stmt
        return node


def if_dividing(source_code):
    tree = ast.parse(source_code)

    transformer = IfDividingTransformer()
    new_tree = transformer.visit(tree)

    new_source_code = astor.to_source(new_tree)
    return new_source_code


# source_code = '''
# if x > 5 and y < 3:
#     print("x is greater than 5 and y is less than 3")
# '''
#
# new_code = if_dividing(source_code)
# print(new_code)
