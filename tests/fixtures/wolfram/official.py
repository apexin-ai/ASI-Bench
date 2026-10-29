"""Protocol-only fixture. NEVER substitutes for a Wolfram scientific test."""
import base64
import json
import sys
import time

for line in sys.stdin:
    request = json.loads(line)
    if 'id' not in request:
        continue
    method = request['method']; params = request.get('params', {})
    if method == 'initialize':
        result = {'protocolVersion': params['protocolVersion'], 'capabilities': {'tools': {}, 'prompts': {}, 'resources': {}},
                  'serverInfo': {'name': 'protocol-fixture-not-Wolfram', 'version': '1'}}
    elif method == 'tools/list':
        name = 'WolframLanguageEvaluator' if not params.get('cursor') else 'FixtureSecond'
        result = {'tools': [{'name': name, 'inputSchema': {'type': 'object', 'properties': {'code': {'type': 'string'}}}}]}
        if name == 'WolframLanguageEvaluator': result['nextCursor'] = 'second'
    elif method == 'tools/call':
        code = params.get('arguments', {}).get('code', '')
        if code == 'hang': time.sleep(20)
        if code == 'exit': sys.exit(1)
        result = {'content': [{'type': 'text', 'text': code},
                   {'type': 'image', 'data': base64.b64encode(b'image-fixture').decode(), 'mimeType': 'image/png'},
                   {'type': 'resource', 'resource': {'uri': 'file:///workspace/fixture.txt', 'text': 'fixture', 'mimeType': 'text/plain'}}],
                  'structuredContent': {'echo': code}, 'isError': code == 'error', '_meta': {'fixture': True}}
    elif method == 'prompts/list':
        result = {'prompts': [{'name': 'fixture-prompt', 'description': 'fixture'}]}
    elif method == 'prompts/get':
        result = {'messages': [{'role': 'user', 'content': {'type': 'text', 'text': 'fixture prompt'}}]}
    elif method == 'resources/list':
        result = {'resources': [{'name': 'fixture', 'uri': 'file:///workspace/fixture.txt'}]}
    elif method == 'resources/templates/list':
        result = {'resourceTemplates': []}
    elif method == 'resources/read':
        result = {'contents': [{'uri': params['uri'], 'text': 'fixture resource'}]}
    elif method == 'ping': result = {}
    else:
        print(json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'error': {'code': -32601, 'message': 'unsupported fixture method'}}), flush=True)
        continue
    print(json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': result}), flush=True)
