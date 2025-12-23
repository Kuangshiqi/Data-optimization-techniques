import ast
import astor


class IfToTernaryTransformer(ast.NodeTransformer):
    def visit_If(self, node):
        if (len(node.body) == 1 and isinstance(node.body[0], ast.Assign) and
                len(node.orelse) == 1 and isinstance(node.orelse[0], ast.Assign)):

            true_branch = node.body[0]
            false_branch = node.orelse[0]

            if (isinstance(true_branch.targets[0], ast.Name) and
                    isinstance(false_branch.targets[0], ast.Name) and
                    true_branch.targets[0].id == false_branch.targets[0].id):
                ternary_expr = ast.IfExp(
                    test=node.test,
                    body=true_branch.value,
                    orelse=false_branch.value
                )

                return ast.Assign(
                    targets=[true_branch.targets[0]],
                    value=ternary_expr
                )

        return node


def single_if_to_conditional(source_code):
    tree = ast.parse(source_code)
    transformer = IfToTernaryTransformer()
    new_tree = transformer.visit(tree)

    new_source_code = astor.to_source(new_tree)
    return new_source_code

# source_code = '''
# if x > 5:
#     y = 10
# else:
#     y = 20
# '''
#
# new_code = single_if_to_conditional(source_code)
# print(new_code)
