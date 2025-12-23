import ast
import astor

class ForToWhileTransformer(ast.NodeTransformer):
    def visit_For(self, node):
        if isinstance(node.target, ast.Name) and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name) and node.iter.func.id == 'range':
            range_args = node.iter.args
            if len(range_args) == 1:
                start = ast.Constant(value=0)
                stop = range_args[0]
                step = ast.Constant(value=1)
            elif len(range_args) == 2:
                start = range_args[0]
                stop = range_args[1]
                step = ast.Constant(value=1)
            elif len(range_args) == 3:
                start = range_args[0]
                stop = range_args[1]
                step = range_args[2]

            init = ast.Assign(targets=[node.target], value=start)
            condition = ast.Compare(left=node.target, ops=[ast.Lt()], comparators=[stop])
            increment = ast.AugAssign(target=node.target, op=ast.Add(), value=step)

            new_while = ast.While(
                test=condition,
                body=node.body + [increment],
                orelse=node.orelse
            )

            return [init, new_while]

        if isinstance(node.target, ast.Name):
            iter_var = ast.Name(id=node.target.id + '_iter', ctx=ast.Store())
            init = ast.Assign(
                targets=[iter_var],
                value=ast.Call(func=ast.Name(id='iter', ctx=ast.Load()), args=[node.iter], keywords=[])
            )

            next_var = ast.Assign(
                targets=[node.target],
                value=ast.Call(func=ast.Name(id='next', ctx=ast.Load()), args=[iter_var], keywords=[])
            )

            new_while = ast.While(
                test=ast.Constant(value=True),
                body=[next_var] + node.body,
                orelse=node.orelse
            )

            return [init, new_while]

        return node


def for_to_while(source_code):
    tree = ast.parse(source_code)
    transformer = ForToWhileTransformer()
    new_tree = transformer.visit(tree)

    new_source_code = astor.to_source(new_tree)
    if new_source_code == source_code:
        return " "
    else:
        return new_source_code
