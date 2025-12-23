import ast
import astor


class VarDeclarationDividingTransformer(ast.NodeTransformer):
    def visit_Assign(self, node):
        if isinstance(node.targets[0], ast.Tuple) and isinstance(node.value, ast.Tuple):
            new_assignments = []
            for target, value in zip(node.targets[0].elts, node.value.elts):
                new_assign = ast.Assign(targets=[target], value=value)
                new_assignments.append(new_assign)
            return new_assignments
        return node


def var_declaration_dividing(source_code):
    source_code = source_code.replace('\t', '    ')
    tree = ast.parse(source_code)
    transformer = VarDeclarationDividingTransformer()
    new_tree = transformer.visit(tree)

    new_source_code = astor.to_source(new_tree)
    if new_source_code == source_code:
        return " "
    else:
        return new_source_code

