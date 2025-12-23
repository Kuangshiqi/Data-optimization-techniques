import ast
import astor

class WhileToForTransformer(ast.NodeTransformer):
    def visit_While(self, node):
        node.body = [self.visit(stmt) for stmt in node.body]
        node.orelse = [self.visit(stmt) for stmt in node.orelse]
        if isinstance(node.test, ast.Compare) and isinstance(node.test.left, ast.Name) and len(node.test.ops) == 1:
            loop_var = node.test.left
            operator = node.test.ops[0]
            stop_value = node.test.comparators[0]

            if isinstance(operator, (ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Eq, ast.NotEq)):
                increment = self._find_increment(node.body, loop_var.id)
                if increment:
                    step_value = self._get_increment_value(increment)
                    init_value, init_statement = self._find_initialization(node, loop_var.id)
                    if init_value is None:
                        return node

                    new_for = ast.For(
                        target=loop_var,
                        iter=ast.Call(func=ast.Name(id='range', ctx=ast.Load()),
                                      args=[init_value, stop_value, step_value], keywords=[]),
                        body=node.body[:-1],
                        orelse=node.orelse
                    )
                    if init_statement:
                        parent_body = getattr(node, 'parent_body', [])
                        if init_statement in parent_body:
                            parent_body.remove(init_statement)

                    return new_for

        return self.generic_visit(node)

    def _find_increment(self, body, var_name):
        """查找增量操作（例如 i += n 或 i = i + n）"""
        if len(body) > 0:
            last_stmt = body[-1]
            if isinstance(last_stmt, ast.AugAssign):
                if isinstance(last_stmt.op, ast.Add) and isinstance(last_stmt.target, ast.Name) and last_stmt.target.id == var_name:
                    return last_stmt
            elif isinstance(last_stmt, ast.Assign):
                if isinstance(last_stmt.targets[0], ast.Name) and last_stmt.targets[0].id == var_name:
                    if isinstance(last_stmt.value, ast.BinOp) and isinstance(last_stmt.value.op, ast.Add) and isinstance(last_stmt.value.left, ast.Name) and last_stmt.value.left.id == var_name:
                        return last_stmt
                    if isinstance(last_stmt.value, ast.BinOp) and isinstance(last_stmt.value.op, ast.Add):
                        return last_stmt
        return None

    def _get_increment_value(self, increment):
        """提取增量值（n），保留表达式"""
        if isinstance(increment, ast.AugAssign):
            return increment.value
        elif isinstance(increment, ast.Assign):
            if isinstance(increment.value, ast.BinOp) and isinstance(increment.value.op, ast.Add):
                add_expr = increment.value
                if isinstance(add_expr.left, ast.Name) and add_expr.left.id == increment.targets[0].id:
                    return add_expr.right
                else:
                    return add_expr
        return None

    def _find_initialization(self, node, var_name):
        """查找循环变量的初始化值"""
        current = getattr(node, 'parent', None)
        while current:
            if hasattr(current, 'body'):
                for stmt in current.body:
                    if isinstance(stmt, ast.Assign):
                        target = stmt.targets[0]
                        if isinstance(target, ast.Name) and target.id == var_name:
                            return stmt.value, stmt
            current = getattr(current, 'parent', None)
        return None, None

    def visit(self, node):
        """为每个节点设置 parent 和 parent_body 属性"""
        for child in ast.iter_child_nodes(node):
            child.parent = node
            child.parent_body = getattr(node, 'body', None)
        return super().visit(node)

def while_to_for(source_code):
    tree = ast.parse(source_code)

    transformer = WhileToForTransformer()
    new_tree = transformer.visit(tree)

    new_source_code = astor.to_source(new_tree)
    return new_source_code

# # 示例代码
# source_code = """
# def get_ludic(n):
#     ludics = []
#     for i in range(1, n + 1):
#         ludics.append(i)
#     index = 2
#     while index != len(ludics):
#         first_ludic = ludics[index]
#         remove_index = index + first_ludic
#         while remove_index < len(ludics):
#             ludics.remove(ludics[remove_index])
#             remove_index = remove_index + first_ludic - 1
#         index = index + 1 + 2
#     return ludics
# """
#
# new_code = while_to_for(source_code)
# print(new_code)
