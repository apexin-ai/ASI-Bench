"""Actual installed Claude + Docker + local fake API, never a paid provider.

Opt in with ASIBENCH_SESSION_DOCKER=1 and optionally ASIBENCH_CLAUDE_BINARY.
Exercises native CLI run/resume, dry-run without credentials, independent scoring,
and hard cumulative deadlines. No experiment repository imports or lock files.
"""
import http.server
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import yaml
from click.testing import CliRunner


@unittest.skipUnless(os.environ.get('ASIBENCH_SESSION_DOCKER') == '1', 'opt-in local Claude/Docker integration')
class NativeRecoveryTests(unittest.TestCase):
    def test_native_run_resume_score_and_deadline(self):
        from ai4sci_bench.cli import cli
        from ai4sci_bench.runner.claude_checkpoint import docker, checkpoint_key
        from ai4sci_bench.runner.session_checkpoint import load_checkpoint
        from ai4sci_bench.runner.task_image import TaskImageBuilder
        binary = os.environ.get('ASIBENCH_CLAUDE_BINARY') or shutil.which('claude')
        self.assertTrue(binary, 'Provide ASIBENCH_CLAUDE_BINARY')
        self.assertIn('Claude Code', subprocess.check_output([binary, '--version'], text=True))
        observed = []
        phase = {'resume': False, 'initial_success': False, 'resumed_tool': False, 'stall': False}
        class API(http.server.BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path.startswith('/v1/messages/count_tokens'):
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(b'{"input_tokens":100}')
                    return
                observed.append(body)
                if phase['stall']:
                    time.sleep(3)
                if not phase['resume'] and phase['initial_success']:
                    self.send_response(403)
                    self.send_header('Content-Type', 'application/json')
                    self.send_header('request-id', 'local-quota-failure')
                    self.end_headers()
                    self.wfile.write(b'{"type":"error","error":{"type":"permission_error","message":"token quota is not enough"}}')
                    return
                if not phase['resume']:
                    phase['initial_success'] = True
                    command = 'printf checkpoint-kept > before.txt; printf rootfs-kept > /home/agent/rootfs-kept'
                    content = {'type': 'tool_use', 'id': 'tool-before', 'name': 'Bash', 'input': {'command': command}}
                    stop, number = 'tool_use', 1
                elif not phase['resumed_tool']:
                    phase['resumed_tool'] = True
                    command = 'test "$(cat before.txt)" = checkpoint-kept && test "$(cat /home/agent/rootfs-kept)" = rootfs-kept && printf resumed > after.txt'
                    content = {'type': 'tool_use', 'id': 'tool-after', 'name': 'Bash', 'input': {'command': command}}
                    stop, number = 'tool_use', 2
                else:
                    content = {'type': 'text', 'text': 'Finished using the preserved workspace.'}
                    stop, number = 'end_turn', 3
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.end_headers()
                message = {'id': f'msg-local-{number}', 'type': 'message', 'role': 'assistant',
                           'model': 'claude-opus-5-5', 'content': [], 'stop_reason': None,
                           'stop_sequence': None, 'usage': {'input_tokens': number * 10,
                           'output_tokens': 1, 'cache_creation_input_tokens': 0, 'cache_read_input_tokens': 0}}
                events = [{'type': 'message_start', 'message': message},
                          {'type': 'content_block_start', 'index': 0, 'content_block':
                              ({**content, 'input': {}} if stop == 'tool_use' else {'type': 'text', 'text': ''})},
                          {'type': 'content_block_delta', 'index': 0, 'delta':
                              ({'type': 'input_json_delta', 'partial_json': json.dumps(content['input'])}
                               if stop == 'tool_use' else {'type': 'text_delta', 'text': content['text']})},
                          {'type': 'content_block_stop', 'index': 0},
                          {'type': 'message_delta', 'delta': {'stop_reason': stop, 'stop_sequence': None},
                           'usage': {'output_tokens': number + 2}}, {'type': 'message_stop'}]
                for event in events:
                    try:
                        self.wfile.write(f'event: {event["type"]}\ndata: {json.dumps(event)}\n\n'.encode())
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return

        with tempfile.TemporaryDirectory(prefix='asibench-native-resume-test-') as tmp:
            directory = Path(tmp)
            build_context = directory / 'build'
            build_context.mkdir()
            shutil.copyfile(binary, build_context / 'claude')
            (build_context / 'Dockerfile').write_text('FROM python:3.12-slim\n'
                'COPY --chmod=755 claude /usr/local/bin/claude\n'
                'RUN mkdir -p /opt/venv/bin && ln -s /usr/local/bin/python /opt/venv/bin/python '
                '&& useradd -m -u 1000 agent\nUSER agent\n')
            tag = 'asibench-native-session-test:' + str(os.getpid())
            built = subprocess.run(['docker', 'build', '--network=none', '-t', tag, str(build_context)],
                                   capture_output=True, text=True, timeout=120)
            self.assertEqual(built.returncode, 0, built.stderr)
            image = docker('image', 'inspect', '--format', '{{.Id}}', tag)
            tasks = directory / 'tasks'
            task_dir = tasks / 'test/resume'
            task_dir.mkdir(parents=True)
            (task_dir / 'task.yaml').write_text(yaml.safe_dump({
                'id': 'test.resume', 'name': 'Local checkpoint test', 'version': '1.0', 'status': 'test',
                'domain': 'test', 'subdomain': 'test', 'input': {'files': []},
                'output': {'files': [{'name': 'after.txt', 'type': 'data'}]},
                'evaluation': {'gates': [], 'scoring': [{'scorer': 'file_match', 'weight': 100,
                    'config': {'checks': [{'file': 'after.txt'}]}}]}}))
            instances = directory / 'instances'
            bundle = instances / 'test.resume__seed31415'
            (bundle / 'reference').mkdir(parents=True)
            (bundle / 'reference/after.txt').write_text('resumed')
            (bundle / 'prompt_b4.md').write_text('Create the requested files with Bash, then complete the task.')
            api = http.server.ThreadingHTTPServer(('127.0.0.1', 0), API)
            thread = threading.Thread(target=api.serve_forever, daemon=True)
            thread.start()
            runner = CliRunner()
            key = checkpoint_key('test.resume', bundle.name, 'b4')
            env = {'LOCAL_CHECKPOINT_TEST_KEY': 'local-fake-key-only',
                   'LOCAL_CHECKPOINT_TEST_URL': f'http://127.0.0.1:{api.server_port}',
                   'PYTHON_DOTENV_DISABLED': '1'}
            config = {'model': 'claude-opus-5-5', 'effort': 'max', 'api_protocol': 'anthropic',
                      'api_key_env': 'LOCAL_CHECKPOINT_TEST_KEY', 'api_base_env': 'LOCAL_CHECKPOINT_TEST_URL',
                      'network_policy': 'model-only'}
            common = ['--tasks', 'test.resume', '--tasks-dir', str(tasks), '--instances-dir', str(instances)]
            first, second = directory / 'first', directory / 'second'
            try:
                with patch.dict(os.environ, env), \
                        patch.object(TaskImageBuilder, 'ensure_image', return_value=tag), \
                        patch.object(TaskImageBuilder, 'get_image_identity', return_value=image):
                    run_args = ['run', *common, '--include-test', '--prompt-levels', 'b4',
                                '--agent', 'claude_code_cli', '--agent-config', json.dumps(config),
                                '--sandbox', 'os', '--checkpoint-sessions', '--timeout', '120',
                                '--fail-on-agent-error']
                    initial = runner.invoke(cli, [*run_args, '--output-dir', str(first)])
                    self.assertEqual(initial.exit_code, 1, initial.output + repr(initial.exception))
                    source = first / 'checkpoints' / key / 'checkpoint.json'
                    self.assertTrue(source.is_file(), initial.output + repr(initial.exception))
                    saved = load_checkpoint(source)[1]
                    self.assertEqual(saved['failure_class'], 'external_api')
                    self.assertEqual(saved['accounting']['observed_tokens']['input_tokens'], 10)
                    before_calls = len(observed)
                    resume_args = ['resume', '--from', str(first), *common, '--levels', 'b4',
                                   '--output-dir', str(second)]
                    os.environ.pop('LOCAL_CHECKPOINT_TEST_KEY')
                    with patch('ai4sci_bench.cli._ensure_run_sandbox_available', side_effect=AssertionError('dry-run called Docker')):
                        preview = runner.invoke(cli, [*resume_args, '--dry-run'])
                    self.assertEqual(preview.exit_code, 0, preview.output + repr(preview.exception))
                    self.assertFalse(second.exists())
                    self.assertEqual(len(observed), before_calls)
                    self.assertFalse(source.with_name('continuation.json').exists())
                    os.environ['LOCAL_CHECKPOINT_TEST_KEY'] = 'local-fake-key-only'
                    phase['resume'] = True
                    result = runner.invoke(cli, resume_args)
                    self.assertEqual(result.exit_code, 0, result.output + repr(result.exception))
                    final = json.loads((second / 'checkpoints' / key / 'checkpoint.json').read_text())
                    self.assertEqual(final['session_id'], saved['session_id'])
                    self.assertEqual(len(final['segments']), 2)
                    self.assertGreater(final['budget']['charged_seconds'], saved['budget']['charged_seconds'])
                    self.assertEqual(final['budget']['limit_seconds'], 120)
                    self.assertEqual(final['accounting']['observed_tokens']['input_tokens'], 60)
                    self.assertEqual(final['accounting']['observed_tokens']['output_tokens'], 12)
                    self.assertIsNone(final['accounting']['actual_cost_usd'])
                    self.assertTrue(source.with_name('continuation.json').exists())
                    self.assertNotIn('local-fake-key-only', json.dumps(final))
                    self.assertTrue((second / 'session-accounting.json').is_file())
                    continued = next(b for b in observed if 'Continue the interrupted task' in json.dumps(b['messages']))
                    self.assertIn('tool-before', json.dumps(continued['messages']))
                    # The resumed artifact is accepted by the real independent scorer.
                    score_path = directory / 'scores/report.json'
                    score = runner.invoke(cli, ['score', '--repo', 'seed31415', '--results-dir', str(second),
                        '--instances-dir', str(instances), '--tasks-dir', str(tasks), '--output', str(score_path)])
                    self.assertEqual(score.exit_code, 0, score.output + repr(score.exception))
                    report = json.loads(score_path.read_text())
                    self.assertEqual(report['scorer_error_count'], 0)
                    self.assertEqual(report['mean_percent'], 100)
                    # A slow upstream cannot turn cleanup grace into generation time.
                    phase['stall'] = True
                    deadline = directory / 'deadline'
                    deadline_args = [*run_args, '--output-dir', str(deadline)]
                    deadline_args[deadline_args.index('--timeout') + 1] = '2'
                    exhausted = runner.invoke(cli, deadline_args)
                    self.assertEqual(exhausted.exit_code, 1, exhausted.output + repr(exhausted.exception))
                    limited = json.loads((deadline / 'checkpoints' / key / 'checkpoint.json').read_text())
                    self.assertEqual(limited['failure_class'], 'model_budget_or_process')
                    self.assertLess(limited['budget']['charged_seconds'], 4)
            finally:
                api.shutdown()
                api.server_close()
                thread.join()
                docker('image', 'rm', tag)
