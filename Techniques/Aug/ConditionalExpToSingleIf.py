import ast
import astor

class TernaryToIfTransformer(ast.NodeTransformer):
    def visit_Assign(self, node):
        if isinstance(node.value, ast.IfExp):
            condition = node.value.test
            true_branch = node.value.body
            false_branch = node.value.orelse
            target = node.targets[0]

            new_if_stmt = ast.If(
                test=condition,
                body=[ast.Assign(targets=[target], value=true_branch)],
                orelse=[ast.Assign(targets=[target], value=false_branch)]
            )

            return new_if_stmt

        return node


def conditional_to_single_if(source_code):
    tree = ast.parse(source_code)

    transformer = TernaryToIfTransformer()
    new_tree = transformer.visit(tree)

    new_source_code = astor.to_source(new_tree)
    if new_source_code == source_code:
        return " "
    else:
        return new_source_code


# source_code = '''
# y = z if x > 5 else 20
# '''
#
# new_code = conditional_to_single_if(source_code)
# print(new_code)
