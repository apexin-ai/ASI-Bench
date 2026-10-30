"""Execute one native mathematics program in a disposable process, never the MCP process."""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path
import sys
import traceback


def backend_namespace(backend):
    if backend == 'sympy':
        import sympy as sp
        return dict(vars(sp), sp=sp, __name__='__main__'), sp.__version__, sp.srepr, sp.latex
    if backend == 'sagemath':
        import sage.all as sage
        from sage.env import SAGE_VERSION
        return dict(vars(sage), sage=sage, __name__='__main__'), SAGE_VERSION, repr, sage.latex
    raise ValueError('unknown mathematics backend')


def main():
    if len(sys.argv) == 3 and sys.argv[1] == '--probe':
        backend = sys.argv[2]
        _, version, _, _ = backend_namespace(backend)
        print(json.dumps({'backend': backend, 'version': version, 'python': sys.version.split()[0]}))
        return 0
    directory = Path(sys.argv[1]).resolve()
    os.chdir(directory)
    request = json.loads((directory / 'request.json').read_text())
    backend = request['backend']
    result = {'backend': backend, 'state': 'failed'}
    try:
        namespace, version, exact, latex = backend_namespace(backend)
        result['version'] = version
        namespace['__file__'] = str(directory / 'input.py')
        code = (directory / 'input.py').read_text(encoding='utf-8')
        if request['syntax'] == 'sage':
            if backend != 'sagemath':
                raise ValueError('sage syntax requires SageMath')
            from sage.repl.preparse import preparse
            code = preparse(code)
        tree = ast.parse(code, filename='input.py')
        # The last expression is the result; scripts without one may set `result`.
        if tree.body and isinstance(tree.body[-1], ast.Expr):
            tree.body[-1] = ast.copy_location(
                ast.Assign(targets=[ast.Name(id='_asibench_result', ctx=ast.Store())],
                           value=tree.body[-1].value), tree.body[-1])
        exec(compile(ast.fix_missing_locations(tree), 'input.py', 'exec'), namespace)
        value = namespace.get('_asibench_result', namespace.get('result'))
        representation = repr(value)
        (directory / 'result.repr.txt').write_text(representation, encoding='utf-8')
        # A bounded preview is not the full expression. Full text is downloadable.
        result.update(state='completed', result_type=type(value).__name__,
                      representation=representation[:2048], truncated=len(representation) > 2048)
        try:
            exact_text = str(exact(value))
            (directory / 'result.exact.txt').write_text(exact_text, encoding='utf-8')
            result['exact'] = exact_text[:2048]
            result['exact_truncated'] = len(exact_text) > 2048
        except Exception as exc:
            result['exact_export_error'] = type(exc).__name__
        try:
            (directory / 'result.latex.txt').write_text(str(latex(value)), encoding='utf-8')
        except Exception as exc:
            result['latex_export_error'] = type(exc).__name__
    except BaseException as exc:
        result.update(state='failed', error_type=type(exc).__name__, error=str(exc)[:4096])
        traceback.print_exc(file=sys.stderr)
    temporary = directory / '.result.json.tmp'
    temporary.write_text(json.dumps(result, ensure_ascii=True), encoding='utf-8')
    temporary.replace(directory / 'result.json')
    return 0 if result['state'] == 'completed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
