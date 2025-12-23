import ast
import astor


class AddAssignToEqualAssignTransformer(ast.NodeTransformer):
    def visit_AugAssign(self, node):
        if isinstance(node.op, ast.Add):
            new_node = ast.Assign(
                targets=[node.target],
                value=ast.BinOp(left=node.target, op=ast.Add(), right=node.value)
            )
            return new_node
        return self.generic_visit(node)


def add_assign_to_equal_assign(source_code):
    tree = ast.parse(source_code)
    transformer = AddAssignToEqualAssignTransformer()
    new_tree = transformer.visit(tree)

    new_source_code = astor.to_source(new_tree)
    if new_source_code == source_code:
        return " "
    else:
        return new_source_code


# source_code = '''
# if x == 5:
#     x = 7
#     x += 1
# else:
#     y = 10
#     y += 2
#     z += 3
# '''
#
# new_code = add_assign_to_equal_assign(source_code)
# print("转换后的代码:")
# print(new_code)
