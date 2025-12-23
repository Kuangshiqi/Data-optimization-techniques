import ast
import astor


class VarDeclarationMergingTransformer(ast.NodeTransformer):
    def __init__(self):
        self.assignments = []

    def visit_Assign(self, node):
        self.assignments.append(node)
        return None

    def merge_assignments(self):
        if len(self.assignments) > 1:
            targets = [assign.targets[0] for assign in self.assignments]
            values = [assign.value for assign in self.assignments]
            merged_assign = ast.Assign(targets=[ast.Tuple(elts=targets, ctx=ast.Store())],
                                       value=ast.Tuple(elts=values, ctx=ast.Load()))
            return merged_assign
        elif self.assignments:
            return self.assignments[0]
        return None


def var_declaration_merging(source_code):
    source_code = source_code.replace('\t', '    ')
    tree = ast.parse(source_code)

    transformer = VarDeclarationMergingTransformer()
    new_body = []

    for node in tree.body:
        if isinstance(node, ast.Assign):
            transformer.visit_Assign(node)
        else:
            merged_assign = transformer.merge_assignments()
            if merged_assign:
                new_body.append(merged_assign)
            new_body.append(node)
            transformer.assignments.clear()

    merged_assign = transformer.merge_assignments()
    if merged_assign:
        new_body.append(merged_assign)

    tree.body = new_body

    new_source_code = astor.to_source(tree)
    if new_source_code == source_code:
        return " "
    else:
        return new_source_code
