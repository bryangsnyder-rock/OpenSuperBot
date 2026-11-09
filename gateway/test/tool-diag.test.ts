import { afterEach, describe, expect, it, vi } from 'vitest'
import { CodedError } from '../src/coded-error.js'
import { createFakeComputer } from '../src/computer.js'
import { loadConfig } from '../src/config.js'
import { ensureDmThread, openDb, upsertBot } from '../src/db.js'
import { runTurn, type AgentEvents } from '../src/agent.js'
import { createLLM, type ToolDef } from '../src/llm.js'
import { buildTools } from '../src/tools.js'
import {
  classifyArguments,
  contentForm,
  peekToolDiagId,
  sanitizeEndpoint,
  sanitizeFinishReason,
  sanitizeModel,
  sanitizeToolName,
  setToolDiagSink,
  toolDiagEnabled,
} from '../src/tool-diag.js'

const FLAG = 'OPENSUPERBOT_TOOL_DIAG'
const SECRET = 'SECRET_DIAG_VALUE'
const LLM_KEYS = new Set([
  'v', 'kind', 'id', 'step', 'model', 'endpoint_class', 'endpoint_scheme', 'endpoint_host',
  'tools_advertised', 'tool_names', 'tool_choice', 'outcome', 'http_status', 'attempts',
  'finish_reason', 'content_form', 'tool_calls_shape', 'tool_call_count', 'calls',
])
const ROUTE_KEYS = new Set(['v', 'kind', 'id', 'watch', 'has_computer', 'tools_advertised', 'tool_names'])
const DISPATCH_KEYS = new Set(['v', 'kind', 'id', 'step', 'tool', 'dispatch', 'error_class', 'error_code'])
const CALL_KEYS = new Set(['name', 'arg_type', 'parse'])

const lines: string[] = []

function records(): Record<string, unknown>[] {
  return lines.map((line) => {
    expect(line.startsWith('tool-diag ')).toBe(true)
    expect(line.trimEnd()).toBe(line)
    return JSON.parse(line.slice('tool-diag '.length)) as Record<string, unknown>
  })
}

function ofKind(kind: string): Record<string, unknown>[] {
  return records().filter((record) => record.kind === kind)
}

function assertClosed(record: Record<string, unknown>, allowed: Set<string>) {
  for (const key of Object.keys(record)) expect(allowed.has(key)).toBe(true)
  expect(record.v).toBe(1)
  if (record.id !== 'missing') expect(record.id).toMatch(/^td_[0-9a-f]{12}$/)
}

function assertNoLeak(secrets: string[] = [SECRET]) {
  const blob = lines.join('\n')
  for (const secret of secrets) expect(blob).not.toContain(secret)
}

afterEach(() => {
  lines.length = 0
  setToolDiagSink(null)
  vi.unstubAllEnvs()
  vi.restoreAllMocks()
})

describe('tool diagnostic gate', () => {
  it('is disabled unless the flag is one of the accepted on values', () => {
    expect(toolDiagEnabled({})).toBe(false)
    for (const value of ['', '0', 'false', 'off', 'enabled', 'yes please']) {
      expect(toolDiagEnabled({ [FLAG]: value })).toBe(false)
    }
    for (const value of ['1', 'true', 'yes', 'on', 'TRUE', ' On ']) {
      expect(toolDiagEnabled({ [FLAG]: value })).toBe(true)
    }
  })

  it('does not emit or change a parsed tool call when the flag is off', async () => {
    const spy = vi.spyOn(console, 'error').mockImplementation(() => {})
    const llm = createLLM(loadConfig({ OPENSUPERBOT_API_KEY: 'sk-SECRETKEY' }), fakeFetch(structuredBody()))
    const turn = await llm.chat(
      [{ role: 'user', content: SECRET }],
      [tool('message_user', SECRET)],
    )
    expect(turn.toolCalls).toEqual([{ id: 'c1', name: 'message_user', args: { kind: 'report', password: SECRET } }])
    expect(lines).toEqual([])
    expect(spy.mock.calls.flat().join(' ')).not.toContain('tool-diag')
    expect(peekToolDiagId()).toBeUndefined()
  })
})

describe('sanitizers', () => {
  it('classifies argument wire types without returning the value', () => {
    expect(classifyArguments('{"password":"' + SECRET + '"}')).toEqual({ arg_type: 'string', parse: 'ok' })
    expect(classifyArguments('{')).toEqual({ arg_type: 'string', parse: 'fail' })
    expect(classifyArguments({ password: SECRET })).toEqual({ arg_type: 'object', parse: 'skipped' })
    expect(classifyArguments(['x'])).toEqual({ arg_type: 'array', parse: 'skipped' })
    expect(classifyArguments(1)).toEqual({ arg_type: 'number', parse: 'skipped' })
    expect(classifyArguments(false)).toEqual({ arg_type: 'boolean', parse: 'skipped' })
    expect(classifyArguments(null)).toEqual({ arg_type: 'null', parse: 'skipped' })
    expect(classifyArguments(undefined)).toEqual({ arg_type: 'absent', parse: 'skipped' })
    expect(classifyArguments(Symbol(SECRET))).toEqual({ arg_type: 'other', parse: 'skipped' })
  })

  it('keeps safe model and tool tokens and redacts everything else', () => {
    expect(sanitizeModel('qwen3:14b')).toBe('qwen3:14b')
    expect(sanitizeModel('grok-4.7')).toBe('grok-4.7')
    expect(sanitizeModel('org/model_name')).toBe('org/model_name')
    expect(sanitizeModel('stub')).toBe('stub')
    expect(sanitizeModel(`http://127.0.0.1/${SECRET}`)).toBe('redacted')
    expect(sanitizeModel(`user:pass@${SECRET}`)).toBe('redacted')
    expect(sanitizeModel(`bad model ${SECRET}`)).toBe('redacted')
    expect(sanitizeModel({ toString: () => SECRET })).toBe('redacted')
    expect(sanitizeToolName('message_user')).toBe('message_user')
    expect(sanitizeToolName('browser_goto')).toBe('browser_goto')
    expect(sanitizeToolName(`../${SECRET}`)).toBe('redacted')
    expect(sanitizeToolName(`tool\n${SECRET}`)).toBe('redacted')
    expect(sanitizeToolName('')).toBe('invalid')
    expect(sanitizeToolName(12)).toBe('invalid')
  })

  it('keeps only a short finish_reason token', () => {
    expect(sanitizeFinishReason('tool_calls')).toBe('tool_calls')
    expect(sanitizeFinishReason('stop')).toBe('stop')
    expect(sanitizeFinishReason(undefined)).toBe('missing')
    expect(sanitizeFinishReason(null)).toBe('missing')
    expect(sanitizeFinishReason(`stop ${SECRET}`)).toBe('other')
    expect(sanitizeFinishReason(3)).toBe('other')
  })

  it('describes content shape without the text', () => {
    expect(contentForm(undefined)).toBe('absent')
    expect(contentForm(null)).toBe('null')
    expect(contentForm('')).toBe('empty')
    expect(contentForm(`hello ${SECRET}`)).toBe('text')
    expect(contentForm([{ text: SECRET }])).toBe('other')
  })

  it('strips userinfo, query, fragment, port, and path, and hides non-public hosts', () => {
    expect(sanitizeEndpoint(`https://user:${SECRET}@api.example.com:8443/v1/models?api_key=${SECRET}#${SECRET}`)).toEqual({
      endpoint_class: 'public', endpoint_scheme: 'https', endpoint_host: 'api.example.com',
    })
    expect(sanitizeEndpoint('https://API.Example.COM/v1')).toEqual({
      endpoint_class: 'public', endpoint_scheme: 'https', endpoint_host: 'api.example.com',
    })
    expect(sanitizeEndpoint(`http://127.0.0.1:11434/v1?key=${SECRET}`)).toEqual({
      endpoint_class: 'loopback', endpoint_scheme: 'http',
    })
    expect(sanitizeEndpoint(`http://2130706433/${SECRET}`)).toEqual({
      endpoint_class: 'loopback', endpoint_scheme: 'http',
    })
    expect(sanitizeEndpoint(`http://[::1]:11434/${SECRET}`)).toEqual({
      endpoint_class: 'loopback', endpoint_scheme: 'http',
    })
    expect(sanitizeEndpoint(`http://[::ffff:127.0.0.1]/${SECRET}`)).toEqual({
      endpoint_class: 'loopback', endpoint_scheme: 'http',
    })
    expect(sanitizeEndpoint('http://localhost./v1')).toEqual({
      endpoint_class: 'loopback', endpoint_scheme: 'http',
    })
    expect(sanitizeEndpoint('http://gateway.localhost/v1')).toEqual({
      endpoint_class: 'loopback', endpoint_scheme: 'http',
    })
    for (const host of ['10.1.2.3', '172.16.5.5', '192.168.1.20', '169.254.1.1', '100.64.1.1']) {
      expect(sanitizeEndpoint(`http://${host}:11434/${SECRET}`)).toEqual({
        endpoint_class: 'private', endpoint_scheme: 'http',
      })
    }
    expect(sanitizeEndpoint(`http://[fd00::1]/${SECRET}`)).toEqual({
      endpoint_class: 'private', endpoint_scheme: 'http',
    })
    expect(sanitizeEndpoint(`http://[fe80::1]/${SECRET}`)).toEqual({
      endpoint_class: 'private', endpoint_scheme: 'http',
    })
    expect(sanitizeEndpoint(`http://nas/${SECRET}`)).toEqual({
      endpoint_class: 'private', endpoint_scheme: 'http',
    })
    expect(sanitizeEndpoint(`http://db.internal/${SECRET}`)).toEqual({
      endpoint_class: 'private', endpoint_scheme: 'http',
    })
    expect(sanitizeEndpoint(`http://printer.local/${SECRET}`)).toEqual({
      endpoint_class: 'private', endpoint_scheme: 'http',
    })
    expect(sanitizeEndpoint(`http://192.168.0.5.nip.io/${SECRET}`)).toEqual({
      endpoint_class: 'private', endpoint_scheme: 'http',
    })
    expect(sanitizeEndpoint(`http://203.0.113.10:8080/${SECRET}`)).toEqual({
      endpoint_class: 'public-ip', endpoint_scheme: 'http',
    })
    expect(sanitizeEndpoint(`http://8.8.8.8/${SECRET}`)).toEqual({
      endpoint_class: 'public-ip', endpoint_scheme: 'http',
    })
    expect(sanitizeEndpoint(`file:///home/${SECRET}/model`)).toEqual({
      endpoint_class: 'other', endpoint_scheme: 'other',
    })
    expect(sanitizeEndpoint(`not a url ${SECRET}`)).toEqual({ endpoint_class: 'invalid' })
    expect(sanitizeEndpoint('')).toEqual({ endpoint_class: 'invalid' })
    expect(sanitizeEndpoint(undefined)).toEqual({ endpoint_class: 'invalid' })
    const evil = { toString() { throw new Error(SECRET) } }
    expect(sanitizeEndpoint(evil)).toEqual({ endpoint_class: 'invalid' })
    const blob = JSON.stringify([
      sanitizeEndpoint(`https://user:${SECRET}@api.example.com:8443/v1?x=${SECRET}#${SECRET}`),
      sanitizeEndpoint(`http://127.0.0.1:11434/${SECRET}`),
      sanitizeEndpoint(`http://192.168.1.20/${SECRET}`),
      sanitizeEndpoint(`file:///home/${SECRET}`),
      sanitizeEndpoint(`not a url ${SECRET}`),
    ])
    expect(blob).not.toContain(SECRET)
    expect(blob).not.toContain('11434')
    expect(blob).not.toContain('8443')
    expect(blob).not.toContain('192.168')
    expect(blob).not.toContain('/v1')
    expect(blob).not.toContain('user')
  })
})

describe('createLLM diagnostic', () => {
  it('records structured calls, omitted tool_choice, and redacts prompts, args, headers, and descriptions', async () => {
    vi.stubEnv(FLAG, '1')
    setToolDiagSink((line) => lines.push(line))
    let body: Record<string, unknown> | null = null
    const llm = createLLM(loadConfig({
      OPENSUPERBOT_API_BASE: 'http://127.0.0.1:11434/v1',
      OPENSUPERBOT_API_KEY: 'sk-SECRETKEY',
      OPENSUPERBOT_MODEL: 'qwen3:14b',
    }), (async (_url, init) => {
      body = JSON.parse(String(init?.body))
      return new Response(JSON.stringify(structuredBody({
        reasoning: SECRET,
        content: `I should call a tool ${SECRET}`,
      })))
    }) as typeof fetch)
    const tools: ToolDef[] = [{
      type: 'function',
      function: { name: 'message_user', description: SECRET, parameters: { token: SECRET } },
    }]
    const turn = await llm.chat([{ role: 'user', content: SECRET, images: [`data:image/png;base64,${SECRET}`] }], tools)
    expect(turn.toolCalls).toEqual([{ id: 'c1', name: 'message_user', args: { kind: 'report', password: SECRET } }])
    expect(body).toMatchObject({ model: 'qwen3:14b' })
    expect(body).not.toHaveProperty('tool_choice')
    const [record] = ofKind('llm')
    expect(ofKind('llm')).toHaveLength(1)
    assertClosed(record!, LLM_KEYS)
    expect(record).toMatchObject({
      kind: 'llm',
      step: 1,
      model: 'qwen3:14b',
      endpoint_class: 'loopback',
      endpoint_scheme: 'http',
      tools_advertised: 1,
      tool_names: ['message_user'],
      tool_choice: 'omitted',
      outcome: 'completed',
      attempts: 1,
      finish_reason: 'tool_calls',
      content_form: 'text',
      tool_calls_shape: 'array',
      tool_call_count: 1,
      calls: [{ name: 'message_user', arg_type: 'string', parse: 'ok' }],
    })
    expect(record).not.toHaveProperty('endpoint_host')
    assertNoLeak([SECRET, 'sk-SECRETKEY', 'Bearer', 'data:image'])
    expect(lines.join('\n')).not.toContain('/v1')
    expect(lines.join('\n')).not.toContain('11434')
  })

  it('preserves parser behavior for argument shapes and records only type and parse result', async () => {
    vi.stubEnv(FLAG, '1')
    setToolDiagSink((line) => lines.push(line))
    const cases = [
      { arguments: '{"ok":true}', args: { ok: true }, arg_type: 'string', parse: 'ok' },
      { arguments: '{', args: null, arg_type: 'string', parse: 'fail' },
      { arguments: { password: SECRET }, args: null, arg_type: 'object', parse: 'skipped' },
      { arguments: [SECRET], args: null, arg_type: 'array', parse: 'skipped' },
      { arguments: 7, args: null, arg_type: 'number', parse: 'skipped' },
      { arguments: false, args: null, arg_type: 'boolean', parse: 'skipped' },
      { arguments: null, args: null, arg_type: 'null', parse: 'skipped' },
    ]
    for (const item of cases) {
      lines.length = 0
      const llm = createLLM(loadConfig({ OPENSUPERBOT_MODEL: 'demo-model' }), fakeFetch({
        choices: [{
          finish_reason: 'tool_calls',
          message: { role: 'assistant', content: null, tool_calls: [{ id: 'c1', function: { name: 'message_user', arguments: item.arguments } }] },
        }],
      }))
      const turn = await llm.chat([{ role: 'user', content: 'hi' }], [tool('message_user')])
      expect(turn.toolCalls).toEqual([{ id: 'c1', name: 'message_user', args: item.args }])
      expect(ofKind('llm')[0]?.calls).toEqual([{ name: 'message_user', arg_type: item.arg_type, parse: item.parse }])
    }
    const missing = createLLM(loadConfig({ OPENSUPERBOT_MODEL: 'demo-model' }), fakeFetch({
      choices: [{ finish_reason: 'stop', message: { role: 'assistant', content: 'plain prose', tool_calls: [{ id: 'c1', function: { name: 'message_user' } }] } }],
    }))
    const turn = await missing.chat([{ role: 'user', content: 'hi' }], [tool('message_user')])
    expect(turn.toolCalls[0]?.args).toBeNull()
    expect(ofKind('llm').at(-1)?.calls).toEqual([{ name: 'message_user', arg_type: 'absent', parse: 'skipped' }])
    assertNoLeak()
  })

  it('records no structured calls when the model returns prose', async () => {
    vi.stubEnv(FLAG, '1')
    setToolDiagSink((line) => lines.push(line))
    const llm = createLLM(loadConfig({
      OPENSUPERBOT_API_BASE: 'https://api.example.com/v1',
      OPENSUPERBOT_MODEL: 'demo-model',
    }), fakeFetch({
      choices: [{ finish_reason: 'stop', message: { role: 'assistant', content: `tool call \`\`\` ${SECRET}` } }],
    }))
    const turn = await llm.chat([{ role: 'user', content: SECRET }], [tool('message_user', SECRET)])
    expect(turn.toolCalls).toEqual([])
    expect(turn.content).toContain(SECRET)
    expect(ofKind('llm')[0]).toMatchObject({
      endpoint_class: 'public',
      endpoint_scheme: 'https',
      endpoint_host: 'api.example.com',
      tool_choice: 'omitted',
      finish_reason: 'stop',
      content_form: 'text',
      tool_calls_shape: 'absent',
      tool_call_count: 0,
      outcome: 'completed',
    })
    assertNoLeak()
    expect(lines.join('\n')).not.toContain('/v1')
  })

  it('records a malformed tool_calls payload and still throws', async () => {
    vi.stubEnv(FLAG, '1')
    setToolDiagSink((line) => lines.push(line))
    const llm = createLLM(loadConfig({ OPENSUPERBOT_LLM_RETRIES: '0', OPENSUPERBOT_MODEL: 'demo-model' }), fakeFetch({
      choices: [{ finish_reason: `bad ${SECRET}`, message: { role: 'assistant', content: null, tool_calls: { leaked: SECRET } } }],
    }))
    await expect(llm.chat([{ role: 'user', content: SECRET }], [])).rejects.toThrow(TypeError)
    expect(ofKind('llm')[0]).toMatchObject({
      outcome: 'malformed_tool_calls',
      finish_reason: 'other',
      tool_calls_shape: 'other',
      tool_call_count: 0,
    })
    assertNoLeak()
  })

  it('records none when tool_choice is sent and auto when that is the runtime value', async () => {
    vi.stubEnv(FLAG, '1')
    setToolDiagSink((line) => lines.push(line))
    const seen: Record<string, unknown>[] = []
    const fetchImpl = (async (_url: string | URL | Request, init?: RequestInit) => {
      seen.push(JSON.parse(String(init?.body)))
      return new Response(JSON.stringify({ choices: [{ finish_reason: 'stop', message: { role: 'assistant', content: 'ok' } }] }))
    }) as typeof fetch
    const llm = createLLM(loadConfig({ OPENSUPERBOT_MODEL: 'demo-model' }), fetchImpl)
    const tools = [tool('message_user')]
    await llm.chat([{ role: 'user', content: 'hi' }], tools, { toolChoice: 'none' })
    await llm.chat([{ role: 'user', content: 'hi' }], tools, { toolChoice: 'auto' as 'none' })
    expect(seen[0]).toMatchObject({ tool_choice: 'none' })
    expect(seen[1]).toMatchObject({ tool_choice: 'auto' })
    expect(ofKind('llm').map((record) => record.tool_choice)).toEqual(['none', 'auto'])
  })

  it('bounds advertised names and redacts unsafe ones', async () => {
    vi.stubEnv(FLAG, '1')
    setToolDiagSink((line) => lines.push(line))
    const tools = Array.from({ length: 40 }, (_, index) => tool(index === 0 ? `bad ${SECRET}` : `tool_${index}`))
    const llm = createLLM(loadConfig({ OPENSUPERBOT_MODEL: 'demo-model' }), fakeFetch({
      choices: [{ finish_reason: 'stop', message: { role: 'assistant', content: '' } }],
    }))
    await llm.chat([{ role: 'user', content: 'hi' }], tools)
    const record = ofKind('llm')[0]!
    expect(record.tools_advertised).toBe(40)
    expect(record.tool_names).toHaveLength(32)
    expect((record.tool_names as string[])[0]).toBe('redacted')
    expect(record.content_form).toBe('empty')
    assertNoLeak()
  })

  it('classifies http, timeout, unreachable, abort, and empty choices without provider text', async () => {
    vi.stubEnv(FLAG, '1')
    setToolDiagSink((line) => lines.push(line))
    const http = createLLM(loadConfig({ OPENSUPERBOT_LLM_RETRIES: '0' }), (async () => new Response(SECRET, { status: 400 })) as typeof fetch, async () => {})
    await expect(http.chat([{ role: 'user', content: SECRET }], [])).rejects.toThrow(/400/)
    expect(ofKind('llm')[0]).toMatchObject({ outcome: 'http_error', http_status: 400, attempts: 1 })
    assertNoLeak()

    lines.length = 0
    let calls = 0
    const retry = createLLM(loadConfig({}), (async () => {
      calls += 1
      return calls === 1 ? new Response(SECRET, { status: 503 }) : new Response(JSON.stringify({
        choices: [{ finish_reason: 'stop', message: { role: 'assistant', content: 'fine' } }],
      }))
    }) as typeof fetch, async () => {})
    expect((await retry.chat([{ role: 'user', content: 'hi' }], [])).content).toBe('fine')
    expect(ofKind('llm')).toHaveLength(1)
    expect(ofKind('llm')[0]).toMatchObject({ outcome: 'completed', attempts: 2 })
    assertNoLeak()

    lines.length = 0
    const hang = ((_url: string | URL | Request, init?: RequestInit) => new Promise<Response>((_resolve, reject) => {
      init?.signal?.addEventListener('abort', () => reject(new DOMException(SECRET, 'AbortError')))
    })) as typeof fetch
    const timeout = createLLM(loadConfig({ OPENSUPERBOT_LLM_TIMEOUT_MS: '20', OPENSUPERBOT_LLM_RETRIES: '0' }), hang, async () => {})
    await expect(timeout.chat([{ role: 'user', content: 'hi' }], [])).rejects.toThrow(/timed out/i)
    expect(ofKind('llm')[0]).toMatchObject({ outcome: 'timeout' })
    assertNoLeak()

    lines.length = 0
    const down = createLLM(loadConfig({ OPENSUPERBOT_LLM_RETRIES: '0' }), (async () => {
      throw new Error(`connect ${SECRET}`)
    }) as typeof fetch, async () => {})
    await expect(down.chat([{ role: 'user', content: 'hi' }], [])).rejects.toThrow(/unreachable/i)
    expect(ofKind('llm')[0]).toMatchObject({ outcome: 'unreachable' })
    assertNoLeak()

    lines.length = 0
    const stop = new AbortController()
    const aborted = createLLM(loadConfig({ OPENSUPERBOT_LLM_RETRIES: '0' }), hang, async () => {})
    const pending = aborted.chat([{ role: 'user', content: 'hi' }], [], { signal: stop.signal })
    stop.abort()
    await expect(pending).rejects.toThrow()
    expect(ofKind('llm')[0]).toMatchObject({ outcome: 'aborted' })
    assertNoLeak()

    lines.length = 0
    const empty = createLLM(loadConfig({ OPENSUPERBOT_LLM_RETRIES: '0' }), fakeFetch({ choices: [] }))
    await expect(empty.chat([{ role: 'user', content: SECRET }], [])).rejects.toThrow(/empty choices/)
    expect(ofKind('llm')[0]).toMatchObject({ outcome: 'empty_choices' })
    assertNoLeak()
  })

  it('still returns the provider result when the diagnostic sink throws', async () => {
    vi.stubEnv(FLAG, '1')
    setToolDiagSink(() => { throw new Error(SECRET) })
    const llm = createLLM(loadConfig({ OPENSUPERBOT_MODEL: 'demo-model' }), fakeFetch(structuredBody()))
    await expect(llm.chat([{ role: 'user', content: SECRET }], [tool('message_user')])).resolves.toMatchObject({
      toolCalls: [{ id: 'c1', name: 'message_user', args: { kind: 'report', password: SECRET } }],
    })
  })

  it('records stub routing without claiming a wire argument format', async () => {
    vi.stubEnv(FLAG, '1')
    setToolDiagSink((line) => lines.push(line))
    const llm = createLLM(loadConfig({ OPENSUPERBOT_MODEL: 'stub' }))
    const turn = await llm.chat([{ role: 'user', content: 'https://example.com' }], [tool('browser_goto')])
    expect(turn.toolCalls[0]?.name).toBe('browser_goto')
    expect(ofKind('llm')[0]).toMatchObject({
      model: 'stub',
      endpoint_class: 'stub',
      tool_choice: 'omitted',
      outcome: 'completed',
      tool_call_count: 1,
      calls: [{ name: 'browser_goto', arg_type: 'unobserved', parse: 'skipped' }],
    })
    expect(ofKind('llm')[0]).not.toHaveProperty('endpoint_host')
    expect(lines.join('\n')).not.toContain('example.com')
  })
})

describe('runTurn diagnostic', () => {
  const bot = { id: 'researcher', name: 'Scout', role: 'research', emoji: '🔎', soul_path: '' }

  function harness(responses: unknown[], opts: { baseURL?: string, model?: string } = {}) {
    const bodies: Record<string, unknown>[] = []
    const fetchImpl = (async (_url: string | URL | Request, init?: RequestInit) => {
      bodies.push(JSON.parse(String(init?.body)))
      const next = responses.shift()
      if (!next) return new Response('no script', { status: 500 })
      return new Response(JSON.stringify(next))
    }) as typeof fetch
    const llm = createLLM(loadConfig({
      OPENSUPERBOT_API_BASE: opts.baseURL ?? 'https://api.example.com/v1',
      OPENSUPERBOT_MODEL: opts.model ?? 'demo-model',
      OPENSUPERBOT_LLM_RETRIES: '0',
    }), fetchImpl)
    const db = openDb(':memory:')
    upsertBot(db, bot)
    const threadId = ensureDmThread(db, bot.id)
    const events: AgentEvents = { onMessage: () => {}, onStatus: () => {} }
    return { llm, db, threadId, events, bodies }
  }

  it('correlates route, model call, and dispatch, then drops the marker', async () => {
    vi.stubEnv(FLAG, '1')
    setToolDiagSink((line) => lines.push(line))
    const { llm, db, threadId, events, bodies } = harness([
      {
        choices: [{
          finish_reason: 'tool_calls',
          message: {
            role: 'assistant',
            content: null,
            tool_calls: [{
              id: 'c1',
              function: {
                name: 'message_user',
                arguments: JSON.stringify({
                  kind: 'report',
                  payload: { lines: [{ system: 'Web', result: 'scanned' }], closing: SECRET },
                }),
              },
            }],
          },
        }],
      },
      { choices: [{ finish_reason: 'stop', message: { role: 'assistant', content: 'Done.' } }] },
    ])
    const outcome = await runTurn({ db, llm, bot, soul: '', threadId }, `please ${SECRET}`, events)
    expect(outcome).toBe('replied')
    expect(bodies[0]).not.toHaveProperty('tool_choice')
    const route = ofKind('route')
    const llmRecords = ofKind('llm')
    const dispatch = ofKind('dispatch')
    expect(route).toHaveLength(1)
    expect(llmRecords).toHaveLength(2)
    expect(dispatch).toHaveLength(1)
    for (const record of route) assertClosed(record, ROUTE_KEYS)
    for (const record of llmRecords) assertClosed(record, LLM_KEYS)
    for (const record of dispatch) assertClosed(record, DISPATCH_KEYS)
    for (const call of llmRecords[0]!.calls as Record<string, unknown>[]) {
      for (const key of Object.keys(call)) expect(CALL_KEYS.has(key)).toBe(true)
    }
    const id = route[0]!.id
    expect(llmRecords.every((record) => record.id === id)).toBe(true)
    expect(dispatch[0]).toMatchObject({ id, step: 1, tool: 'message_user', dispatch: 'ok' })
    expect(llmRecords.map((record) => record.step)).toEqual([1, 2])
    expect(route[0]).toMatchObject({ watch: false, has_computer: false })
    expect(route[0]!.tool_names).toContain('message_user')
    expect(route[0]!.tool_names).not.toContain('shell')
    expect(llmRecords[0]!.tool_names).toEqual(route[0]!.tool_names)
    expect(peekToolDiagId()).toBeUndefined()
    assertNoLeak([SECRET, 'UNIQUE'])
    expect(lines.join('\n')).not.toContain('/workspace')
    expect(lines.join('\n')).not.toContain('Salesforce')
  })

  it('shows the watch filter and a denied dispatch without logging the command', async () => {
    vi.stubEnv(FLAG, '1')
    setToolDiagSink((line) => lines.push(line))
    const { llm, db, threadId, events } = harness([
      {
        choices: [{
          finish_reason: 'tool_calls',
          message: {
            role: 'assistant',
            content: null,
            tool_calls: [{ id: 'c1', function: { name: 'shell', arguments: JSON.stringify({ cmd: SECRET }) } }],
          },
        }],
      },
      { choices: [{ finish_reason: 'stop', message: { role: 'assistant', content: 'Stopped.' } }] },
    ], { baseURL: 'http://10.0.0.8:11434/v1' })
    const outcome = await runTurn({
      db, llm, bot, soul: '', threadId, watch: true,
      getComputer: async () => createFakeComputer(),
    }, 'watch', events)
    expect(outcome).toBe('replied')
    const names = ofKind('route')[0]!.tool_names as string[]
    expect(names).toContain('read_file')
    expect(names).toContain('message_user')
    expect(names).not.toContain('shell')
    expect(names).not.toContain('write_file')
    expect(names).not.toContain('save_memory')
    expect(ofKind('llm')[0]!.tool_names).toEqual(names)
    expect(ofKind('llm')[0]).toMatchObject({ endpoint_class: 'private', endpoint_scheme: 'http' })
    expect(ofKind('llm')[0]).not.toHaveProperty('endpoint_host')
    expect(ofKind('dispatch')[0]).toMatchObject({ tool: 'shell', dispatch: 'denied' })
    expect(ofKind('route')[0]).toMatchObject({ watch: true, has_computer: true })
    assertNoLeak([SECRET, '10.0.0.8', '11434'])
    expect(lines.join('\n')).not.toContain('not available')
  })

  it('omits shell when the attached computer has no shell', async () => {
    vi.stubEnv(FLAG, '1')
    setToolDiagSink((line) => lines.push(line))
    const { llm, db, threadId, events } = harness([
      { choices: [{ finish_reason: 'stop', message: { role: 'assistant', content: 'ok' } }] },
    ])
    await runTurn({
      db, llm, bot, soul: '', threadId,
      getComputer: async () => createFakeComputer(),
      computer: { kind: 'local', shell: false },
    }, 'hi', events)
    const names = ofKind('route')[0]!.tool_names as string[]
    expect(names).toContain('read_file')
    expect(names).not.toContain('shell')
    expect(ofKind('llm')[0]!.tool_names).toEqual(names)
    expect(buildTools({ hasComputer: true, shell: false }).map((item) => item.function.name)).toEqual(names)
  })

  it('classifies rejected, unknown, unavailable, and thrown dispatch without exception text', async () => {
    vi.stubEnv(FLAG, '1')
    setToolDiagSink((line) => lines.push(line))
    const badReport = {
      choices: [{
        finish_reason: 'tool_calls',
        message: {
          role: 'assistant', content: null,
          tool_calls: [{ id: 'c1', function: { name: 'message_user', arguments: '{"kind":"report","payload":{"lines":[]}}' } }],
        },
      }],
    }
    const text = { choices: [{ finish_reason: 'stop', message: { role: 'assistant', content: 'Done.' } }] }
    const rejected = harness([badReport, text])
    await runTurn({ db: rejected.db, llm: rejected.llm, bot, soul: '', threadId: rejected.threadId }, 'go', rejected.events)
    expect(ofKind('dispatch')[0]).toMatchObject({ tool: 'message_user', dispatch: 'rejected' })

    lines.length = 0
    const unknown = harness([
      { choices: [{ finish_reason: 'tool_calls', message: { role: 'assistant', content: null, tool_calls: [{ id: 'c1', function: { name: `../${SECRET}`, arguments: '{}' } }] } }] },
      text,
    ])
    await runTurn({ db: unknown.db, llm: unknown.llm, bot, soul: '', threadId: unknown.threadId }, 'go', unknown.events)
    expect(ofKind('dispatch')[0]).toMatchObject({ tool: 'redacted', dispatch: 'unknown_tool' })
    assertNoLeak()

    lines.length = 0
    const unavailable = harness([
      { choices: [{ finish_reason: 'tool_calls', message: { role: 'assistant', content: null, tool_calls: [{ id: 'c1', function: { name: 'read_file', arguments: '{"path":"' + SECRET + '"}' } }] } }] },
      text,
    ])
    await runTurn({ db: unavailable.db, llm: unavailable.llm, bot, soul: '', threadId: unavailable.threadId }, 'go', unavailable.events)
    expect(ofKind('dispatch')[0]).toMatchObject({ tool: 'read_file', dispatch: 'unavailable' })
    assertNoLeak()

    lines.length = 0
    const thrown = harness([
      { choices: [{ finish_reason: 'tool_calls', message: { role: 'assistant', content: null, tool_calls: [{ id: 'c1', function: { name: 'shell', arguments: '{"cmd":"echo hi"}' } }] } }] },
      text,
    ])
    await runTurn({
      db: thrown.db, llm: thrown.llm, bot, soul: '', threadId: thrown.threadId,
      getComputer: async () => createFakeComputer({
        shell: async () => { throw new Error(`disk ${SECRET}`) },
      }),
    }, 'go', thrown.events)
    expect(ofKind('dispatch')[0]).toMatchObject({ tool: 'shell', dispatch: 'error', error_class: 'error' })
    expect(ofKind('dispatch')[0]).not.toHaveProperty('error_code')
    assertNoLeak()

    lines.length = 0
    const coded = harness([
      { choices: [{ finish_reason: 'tool_calls', message: { role: 'assistant', content: null, tool_calls: [{ id: 'c1', function: { name: 'shell', arguments: '{}' } }] } }] },
      text,
    ])
    await runTurn({
      db: coded.db, llm: coded.llm, bot, soul: '', threadId: coded.threadId,
      getComputer: async () => createFakeComputer({
        shell: async () => { throw new CodedError('browser_missing', SECRET, { path: SECRET }) },
      }),
    }, 'go', coded.events)
    expect(ofKind('dispatch')[0]).toMatchObject({ dispatch: 'error', error_class: 'coded', error_code: 'browser_missing' })
    assertNoLeak()
  })

  it('keeps dispatch behavior when recording throws', async () => {
    vi.stubEnv(FLAG, '1')
    setToolDiagSink(() => { throw new Error(SECRET) })
    const { llm, db, threadId, events } = harness([
      structuredBody(),
      { choices: [{ finish_reason: 'stop', message: { role: 'assistant', content: 'Done.' } }] },
    ])
    await expect(runTurn({ db, llm, bot, soul: '', threadId }, 'go', events)).resolves.toBe('replied')
  })
})

function tool(name: string, description = 'desc'): ToolDef {
  return { type: 'function', function: { name, description, parameters: {} } }
}

function structuredBody(extra: Record<string, unknown> = {}) {
  return {
    choices: [{
      finish_reason: 'tool_calls',
      message: {
        role: 'assistant',
        content: null,
        tool_calls: [{
          id: 'c1',
          type: 'function',
          function: { name: 'message_user', arguments: JSON.stringify({ kind: 'report', password: SECRET }) },
        }],
        ...extra,
      },
    }],
  }
}

function fakeFetch(body: unknown): typeof fetch {
  return (async () => new Response(JSON.stringify(body))) as typeof fetch
}
