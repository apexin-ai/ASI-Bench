"""Recovery policy, corruption, selection and accounting tests (no paid API)."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import uuid
from argparse import Namespace
from unittest.mock import patch

from ai4sci_bench.runner.session_checkpoint import (
    SCHEMA, Claim, ResponseUsage, atomic_json, classify, digest, estimate_cost,
    fingerprint, inventory, load_checkpoint, token_prices, usage_totals)
from ai4sci_bench.runner.session_resume import plan_resume
from ai4sci_bench.runner.claude_checkpoint import checkpoint_key


def checkpoint(root, task='math.example', level='b4', failure='external_api'):
    root.mkdir(parents=True)
    session = str(uuid.uuid4())
    (root / 'home/.claude/projects/-workspace').mkdir(parents=True)
    (root / 'home/.claude/projects/-workspace' / (session + '.jsonl')).write_text('{"type":"user"}\n')
    (root / 'workspace').mkdir()
    (root / 'workspace/kept.py').write_text('print(42)\n')
    (root / 'history').mkdir()
    (root / 'rootfs-image.tar').write_bytes(b'fake image archive')
    data = {'schema': SCHEMA, 'complete': True, 'session_id': session,
            'config': {'task_id': task, 'level': level, 'instance_id': task + '__seed31415'},
            'failure_class': failure,
            'budget': {'limit_seconds': 10800, 'charged_seconds': 100, 'remaining_seconds': 10700},
            'files': {key: inventory(root / key) for key in ('home', 'workspace', 'history')},
            'rootfs': {'sha256': digest(root / 'rootfs-image.tar')},
            'accounting': {'actual_cost_usd': None}, 'segments': []}
    data['config_sha256'] = fingerprint(data['config'])
    atomic_json(root / 'checkpoint.json', data)
    return root / 'checkpoint.json', data


class RecoveryTests(unittest.TestCase):
    def test_native_cli_preview_needs_no_credentials_or_dispatch(self):
        from click.testing import CliRunner
        from ai4sci_bench.cli import cli
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _ = checkpoint(root / 'source/checkpoints/cell')
            args = ['resume', '--from', str(root / 'source'), '--tasks-dir', str(root),
                    '--tasks', 'math.example', '--levels', 'b4', '--output-dir', str(root / 'preview'),
                    '--api-key-env', 'UNSET_CHECKPOINT_KEY', '--dry-run']
            with patch('ai4sci_bench.runner.session_resume.preflight_inputs', return_value=root), \
                    patch('ai4sci_bench.runner.session_resume.execute_resume', side_effect=AssertionError('dispatch')), \
                    patch('ai4sci_bench.cli._ensure_run_sandbox_available', side_effect=AssertionError('Docker')):
                result = CliRunner().invoke(cli, args)
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertFalse((root / 'preview').exists())
            self.assertFalse(source.with_name('continuation.json').exists())

    def test_native_cli_rejects_legacy_logs_and_unsupported_harness(self):
        from click.testing import CliRunner
        from ai4sci_bench.cli import cli
        with tempfile.TemporaryDirectory() as tmp:
            result = CliRunner().invoke(cli, ['resume', '--from', tmp, '--tasks-dir', tmp,
                                              '--output-dir', str(Path(tmp) / 'new'), '--dry-run'])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn('old trajectory logs', result.output)
            with patch('ai4sci_bench.cli._ensure_run_sandbox_available', side_effect=AssertionError('Docker')):
                invalid = CliRunner().invoke(cli, ['run', '--agent', 'codex_cli', '--sandbox', 'os',
                    '--checkpoint-sessions', '--output-dir', str(Path(tmp) / 'unsupported')])
            self.assertNotEqual(invalid.exit_code, 0)
            self.assertIn('Claude Code', invalid.output)

    def test_instance_identity_is_part_of_checkpoint_selection(self):
        self.assertNotEqual(checkpoint_key('math.a', 'math.a__seed42', 'b4'),
                            checkpoint_key('math.a', 'math.a__seed31415', 'b4'))

    def test_terminal_policy_does_not_retry_model_failure(self):
        request = {'id': '1', 'route': '/v1/messages', 'status': 503}
        self.assertEqual(classify('failed', 'API Error: unavailable', [request]), 'external_api')
        self.assertEqual(classify('completed', 'old API error', [request]), 'completed')
        self.assertEqual(classify('timeout', 'API Error', [request]), 'model_budget_or_process')
        self.assertEqual(classify('failed', 'API Error', [request], {'OOMKilled': True}), 'model_budget_or_process')
        self.assertEqual(classify('failed', 'API Error', [request], {'ExitCode': 137}), 'unknown')
        self.assertEqual(classify('failed', 'killed own shell', [request]), 'unknown')
        self.assertEqual(classify('failed', 'API Error: 400', [dict(request, status=400)]), 'unknown')
        self.assertEqual(classify('failed', 'API Error', [dict(request, route='/v1/messages/count_tokens', status=403)]), 'unknown')
        self.assertEqual(classify('failed', 'Docker daemon is not responsive'), 'external_infrastructure')

    def test_checkpoint_integrity_and_remaining_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, data = checkpoint(Path(tmp) / 'cell')
            self.assertEqual(load_checkpoint(path)[1]['session_id'], data['session_id'])
            (path.parent / 'workspace/kept.py').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                load_checkpoint(path)
            data['budget']['charged_seconds'] = 10800
            atomic_json(path, data)
            with self.assertRaisesRegex(ValueError, 'exhausted'):
                load_checkpoint(path)

    def test_missing_session_and_archive_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, data = checkpoint(Path(tmp) / 'cell')
            (path.parent / 'rootfs-image.tar').write_bytes(b'corrupted')
            with self.assertRaisesRegex(ValueError, 'image integrity'):
                load_checkpoint(path)
            (path.parent / 'rootfs-image.tar').write_bytes(b'fake image archive')
            session = next((path.parent / 'home').rglob('*.jsonl'))
            session.unlink()
            data['files']['home'] = inventory(path.parent / 'home')
            atomic_json(path, data)
            with self.assertRaisesRegex(ValueError, 'native Claude session'):
                load_checkpoint(path)

    def test_incomplete_and_model_failure_cannot_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            path, data = checkpoint(Path(tmp) / 'cell', failure='model_budget_or_process')
            with self.assertRaisesRegex(ValueError, 'not an external'):
                load_checkpoint(path)
            data['complete'] = False
            atomic_json(path, data)
            with self.assertRaisesRegex(ValueError, 'incomplete'):
                load_checkpoint(path)

    def test_claim_prevents_concurrency_and_reuse(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, _ = checkpoint(Path(tmp) / 'source')
            with Claim(source, Path(tmp) / 'child') as claim:
                claim.begin_execution()
                with self.assertRaises((ValueError, BlockingIOError)):
                    with Claim(source, Path(tmp) / 'second'):
                        self.fail('must not branch')
            with self.assertRaisesRegex(ValueError, 'already claimed'):
                with Claim(source, Path(tmp) / 'second'):
                    self.fail('must not replay')

    def test_preparation_failure_releases_unused_claim(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, _ = checkpoint(Path(tmp) / 'source')
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                with Claim(source, Path(tmp) / 'child'):
                    raise ValueError('configuration changed')
            self.assertFalse(source.with_name('continuation.json').exists())

    def test_sparse_selection_and_invalid_explicit_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            checkpoint(root / 'checkpoints/a', task='math.a', level='b1')
            checkpoint(root / 'checkpoints/b', task='math.b', level='b4')
            checkpoint(root / 'checkpoints/c', task='math.a', level='b2', failure='unknown')
            chosen, rejected = plan_resume(root)
            self.assertEqual(set(chosen), {checkpoint_key('math.a', 'math.a__seed31415', 'b1'), checkpoint_key('math.b', 'math.b__seed31415', 'b4')})
            self.assertEqual(len(rejected), 1)
            with self.assertRaisesRegex(ValueError, 'no checkpoint'):
                plan_resume(root, 'math.a,math.b', 'b1,b4')

    def test_legacy_logs_are_not_presented_as_resumable(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, 'old trajectory logs'):
                plan_resume(tmp)

    def test_runtime_configuration_mismatch_does_not_touch_fresh_workspace(self):
        from ai4sci_bench.runner.claude_checkpoint import CheckpointExecution
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _ = checkpoint(root / 'source')
            work = root / 'fresh-workspace'
            work.mkdir()
            (work / 'prompt.md').write_text('original input')
            instance = Namespace(task_id='math.example', instance_id='math.example__seed31415',
                                 prompt_level=Namespace(value='b4'), parameters={}, metadata={},
                                 workspace_dir=work, effective_timeout_seconds=10800)
            adapter = Namespace(model='claude-opus-5-5', effort='max', sandbox='os',
                                tool_mode=Namespace(value='restricted'), effective_network_policy='model-only',
                                api_protocol='anthropic', permission_mode=None, supports_image_input=False,
                                _os_sandbox=Namespace(cpu_limit='4', memory_limit='8g'))
            with self.assertRaisesRegex(ValueError, 'frozen experiment parameters changed'):
                CheckpointExecution(adapter, instance, root / 'new-checkpoint', source)
            self.assertEqual((work / 'prompt.md').read_text(), 'original input')
            self.assertFalse((root / 'new-checkpoint').exists())



class AccountingTests(unittest.TestCase):
    def test_chunked_sse_usage_is_not_counted_twice(self):
        parser = ResponseUsage('/v1/messages')
        parser.sse = True
        parser.record['status'] = 200
        events = [
            {'type': 'message_start', 'message': {'usage': {'input_tokens': 10, 'output_tokens': 1,
                         'cache_read_input_tokens': 20, 'cache_creation_input_tokens': 3}}},
            {'type': 'message_delta', 'usage': {'output_tokens': 5}},
            {'type': 'message_delta', 'usage': {'output_tokens': 7}},
            {'type': 'message_stop'}]
        raw = b''.join(b'data: ' + json.dumps(v).encode() + b'\n\n' for v in events)
        for byte in raw:
            parser.feed(bytes([byte]))
        record = parser.finish()
        self.assertTrue(record['usage_complete'])
        totals = usage_totals([record])
        self.assertEqual(totals['observed_tokens']['output_tokens'], 7)
        self.assertEqual(totals['observed_tokens']['cache_read_input_tokens'], 20)
        self.assertIsNone(totals['actual_cost_usd'])
        prices = token_prices(dict.fromkeys(totals['observed_tokens'], 1))
        self.assertAlmostEqual(estimate_cost(totals, prices)['estimated_cost_usd'], 40 / 1e6)

    def test_missing_usage_failure_and_stream_truncation_are_unknown(self):
        parser = ResponseUsage('/v1/messages')
        parser.sse = True
        parser.record['status'] = 200
        parser.feed(b'data: {"type":"message_start","message":{"usage":{"input_tokens":10}}}\n\n')
        record = parser.finish()
        self.assertFalse(record['usage_complete'])
        self.assertTrue(record['stream_error'])
        self.assertFalse(usage_totals([record])['usage_complete'])
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            usage_totals([record, record])

    def test_json_usage_and_count_tokens_are_separate(self):
        parser = ResponseUsage('/v1/messages')
        parser.record['status'] = 200
        parser.feed(b'{"type":"message","usage":{"input_tokens":4,"output_tokens":2}}')
        record = parser.finish()
        count = {'id': 'count', 'route': '/v1/messages/count_tokens', 'usage': {'input_tokens': 999}}
        self.assertTrue(record['usage_complete'])
        self.assertEqual(usage_totals([record, count])['observed_tokens']['input_tokens'], 4)

    def test_bad_prices_rejected(self):
        for value in ({'input_tokens': 1}, dict.fromkeys(('input_tokens', 'output_tokens',
                           'cache_creation_input_tokens', 'cache_read_input_tokens'), float('nan'))):
            with self.assertRaises(ValueError):
                token_prices(value)


if __name__ == '__main__':
    unittest.main()
