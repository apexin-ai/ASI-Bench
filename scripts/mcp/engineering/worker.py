"""Native engine launcher. Inputs execute with container privileges, without a shell."""
from __future__ import annotations
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ENGINES = {'ngspice': ('ngspice', ['--version']), 'iverilog': ('iverilog', ['-V']),
           'verilator': ('verilator', ['--version']), 'opensta': ('sta', ['-version']),
           'calculix': ('ccx', ['-v']), 'openmodelica': ('omc', ['--version'])}
SUFFIX = dict(ngspice='cir', iverilog='sv', verilator='sv', opensta='tcl', calculix='inp', openmodelica='mos')


def probe(backend):
    binary, args = ENGINES[backend]
    path = shutil.which(binary)
    if not path:
        raise RuntimeError(f'{binary} is not installed')
    result = subprocess.run([path, *args], capture_output=True, text=True, timeout=30)
    text = (result.stdout + result.stderr).strip()
    # ccx -v returns 201 on supported distro versions.
    if not text or (result.returncode and backend != 'calculix'):
        raise RuntimeError(f'{binary} version probe failed: {text[:1000]}')
    return dict(backend=backend, executable=path, version=text[:4000])


def run(directory):
    os.chdir(directory)
    request = json.loads(Path('request.json').read_text())
    backend, top = request['backend'], request['top']
    code = Path('input.txt').read_text()
    name = 'input.' + SUFFIX[backend]
    if backend == 'openmodelica':
        # getErrorString reports semantic/compiler errors even when omc exits zero.
        code += '\ngetErrorString();\n'
    Path(name).write_text(code)
    sources = [name] + [n for n in request['input_files'] if Path(n).suffix in {'.v', '.sv'}]
    commands = {
        'ngspice': [['ngspice', '-b', name]],
        'iverilog': [['iverilog', '-g2012', '-s', top, '-o', 'sim.vvp', *sources], ['vvp', 'sim.vvp']],
        'verilator': [['verilator', '--binary', '--timing', '-Wno-fatal', '--top-module', top,
                       '--Mdir', 'obj_dir', '-o', 'simulation', *sources], ['obj_dir/simulation']],
        'opensta': [['sta', '-no_init', '-exit', name]],
        'calculix': [['ccx', '-i', 'input']],
        'openmodelica': [['omc', name]],
    }[backend]
    stages = []
    for index, command in enumerate(commands):
        # Parent owns external timeout/cancellation and kills this whole process group.
        with Path(f'stage-{index}.stdout.log').open('wb') as out, Path(f'stage-{index}.stderr.log').open('wb') as err:
            result = subprocess.run(command, stdin=subprocess.DEVNULL, stdout=out, stderr=err)
        stages.append(dict(command=command, exit_code=result.returncode))
        text = ''.join(Path(f'stage-{index}.{stream}.log').read_text(errors='replace')[-1048576:]
                       for stream in ('stdout', 'stderr'))
        print(text, flush=True)
        error = result.returncode != 0
        if backend == 'ngspice':
            error |= bool(re.search(r'(?im)^\s*(?:fatal error|error(?:\s+on\s+line|\s*:|\s+line)|doanalyses:|run simulation\(s\) aborted)', text))
        elif backend == 'opensta':
            error |= bool(re.search(r'(?m)^Error:|\binvalid command name\b', text))
        elif backend == 'calculix':
            error |= '*ERROR' in text or 'Job finished' not in text
        elif backend == 'openmodelica':
            error |= 'Error:' in text or 'Failed to build model' in text
        if error:
            return dict(backend=backend, state='failed', stages=stages, detail='native engine error; inspect stage logs')
    return dict(backend=backend, state='completed', stages=stages,
                detail='native command completed; scientific convergence and task constraints require separate checks')


if __name__ == '__main__':
    if sys.argv[1] == '--probe':
        print(json.dumps(probe(sys.argv[2])))
    else:
        directory = Path(sys.argv[1]).resolve()
        try:
            result = run(directory)
        except Exception as exc:
            result = dict(state='failed', detail=str(exc))
        (directory / 'result.json').write_text(json.dumps(result))
        sys.exit(0 if result['state'] == 'completed' else 1)
