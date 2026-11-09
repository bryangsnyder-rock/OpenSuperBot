/**
 * Opt-in metadata for tool-call routing. Disabled unless OPENSUPERBOT_TOOL_DIAG is 1, true, yes, or on.
 * Records may include only the fields built below. Never pass messages, prompts, content, arguments,
 * headers, credentials, provider bodies, reasoning, or exception text into a record.
 * Every public function swallows its own failures.
 */
import { AsyncLocalStorage } from 'node:async_hooks'
import { randomBytes } from 'node:crypto'
import { isIPv4, isIPv6 } from 'node:net'
import { CodedError } from './coded-error.js'

const ENV = 'OPENSUPERBOT_TOOL_DIAG'
const MAX_TOOLS = 32
const MAX_CALLS = 16
const MODEL = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,79}$/
const MODEL_SCOPED = /^[A-Za-z0-9][A-Za-z0-9_.:-]{0,40}\/[A-Za-z0-9][A-Za-z0-9_.:-]{0,40}$/
const TOOL_NAME = /^[A-Za-z_][A-Za-z0-9_.:-]{0,63}$/
const FINISH = /^[a-z0-9_]{1,32}$/
const ERROR_CODE = /^[a-z0-9_]{1,40}$/
const PUBLIC_HOST = /^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$/
const LOOPBACK_NAMES = new Set(['localhost', 'ip6-localhost', 'ip6-loopback', 'lvh.me', 'localtest.me', 'localhost.localdomain'])
const PRIVATE_SUFFIXES = ['.localhost', '.local', '.internal', '.lan', '.home', '.home.arpa', '.corp', '.intranet', '.localdomain']

export type EndpointMeta = {
  endpoint_class: 'loopback' | 'private' | 'public' | 'public-ip' | 'stub' | 'invalid' | 'other'
  endpoint_scheme?: 'http' | 'https' | 'other'
  endpoint_host?: string
}

export type ArgClass = {
  arg_type: 'string' | 'object' | 'array' | 'number' | 'boolean' | 'null' | 'absent' | 'other' | 'unobserved'
  parse: 'ok' | 'fail' | 'skipped'
}

type Frame = { id: string; step: number }
type Sink = (line: string) => void
type CallNote = { name: string; arg_type: ArgClass['arg_type']; parse: ArgClass['parse'] }
type FailOutcome = 'empty_choices' | 'malformed_tool_calls' | 'http_error' | 'timeout' | 'unreachable' | 'aborted' | 'error'

export type ChatMeta = {
  attempts?: number
  finishReason?: unknown
  rawToolCalls?: unknown
  content?: unknown
  /** false: the wire arguments were not visible, so parse is reported as skipped */
  observedArgs?: boolean
}

export type ChatDiag = {
  noteCall(name: unknown, rawArgs: unknown): void
  complete(meta: ChatMeta): void
  fail(outcome: FailOutcome, meta?: ChatMeta & { httpStatus?: number }): void
}

export type DispatchClass = 'ok' | 'rejected' | 'denied' | 'unavailable' | 'unknown_tool' | 'error'

const frames = new AsyncLocalStorage<Frame>()
const defaultSink: Sink = (line) => { console.error(line) }
let sink: Sink = defaultSink
const noop: ChatDiag = { noteCall() {}, complete() {}, fail() {} }

export function setToolDiagSink(next: Sink | null): void {
  sink = next ?? defaultSink
}

export function toolDiagEnabled(env: NodeJS.ProcessEnv | Record<string, string | undefined> = process.env): boolean {
  return ['1', 'true', 'yes', 'on'].includes(String(env[ENV] ?? '').trim().toLowerCase())
}

function active(): boolean {
  try { return toolDiagEnabled() } catch { return false }
}

export function peekToolDiagId(): string | undefined {
  try { return frames.getStore()?.id } catch { return undefined }
}

export function openToolDiagFrame(): Frame {
  return { id: `td_${randomBytes(6).toString('hex')}`, step: 0 }
}

export function runWithToolDiag<T>(frame: Frame, fn: () => Promise<T>): Promise<T> {
  return frames.run(frame, fn)
}

export function classifyArguments(raw: unknown): ArgClass {
  if (raw === undefined) return { arg_type: 'absent', parse: 'skipped' }
  if (raw === null) return { arg_type: 'null', parse: 'skipped' }
  if (Array.isArray(raw)) return { arg_type: 'array', parse: 'skipped' }
  switch (typeof raw) {
    case 'string':
      try {
        JSON.parse(raw)
        return { arg_type: 'string', parse: 'ok' }
      } catch {
        return { arg_type: 'string', parse: 'fail' }
      }
    case 'object':
      return { arg_type: 'object', parse: 'skipped' }
    case 'number':
      return { arg_type: 'number', parse: 'skipped' }
    case 'boolean':
      return { arg_type: 'boolean', parse: 'skipped' }
    default:
      return { arg_type: 'other', parse: 'skipped' }
  }
}

export function sanitizeModel(raw: unknown): string {
  if (typeof raw !== 'string' || raw.includes('..') || raw.includes('//') || raw.includes('@')) return 'redacted'
  return MODEL.test(raw) || MODEL_SCOPED.test(raw) ? raw : 'redacted'
}

export function sanitizeToolName(raw: unknown): string {
  if (typeof raw !== 'string' || raw.length === 0) return 'invalid'
  return TOOL_NAME.test(raw) ? raw : 'redacted'
}

export function sanitizeFinishReason(raw: unknown): string {
  if (raw == null) return 'missing'
  return typeof raw === 'string' && FINISH.test(raw) ? raw : 'other'
}

export function contentForm(raw: unknown): 'absent' | 'null' | 'empty' | 'text' | 'other' {
  if (raw === undefined) return 'absent'
  if (raw === null) return 'null'
  if (typeof raw !== 'string') return 'other'
  return raw.length === 0 ? 'empty' : 'text'
}

function ipv4Class(host: string): 'loopback' | 'private' | 'public-ip' | null {
  if (!isIPv4(host)) return null
  const [a, b] = host.split('.').map(Number)
  if (a === 127 || host === '0.0.0.0') return 'loopback'
  if (
    a === 10 || a === 0 || a === 255
    || (a === 172 && b >= 16 && b <= 31)
    || (a === 192 && b === 168)
    || (a === 169 && b === 254)
    || (a === 100 && b >= 64 && b <= 127)
  ) return 'private'
  return 'public-ip'
}

function mappedV4(host: string): string | null {
  const dotted = /^::ffff:(\d{1,3}(?:\.\d{1,3}){3})$/i.exec(host)
  if (dotted?.[1] && isIPv4(dotted[1])) return dotted[1]
  const hex = /^::ffff:([0-9a-f]{1,4}):([0-9a-f]{1,4})$/i.exec(host)
  if (!hex?.[1] || !hex[2]) return null
  const hi = Number.parseInt(hex[1], 16)
  const lo = Number.parseInt(hex[2], 16)
  if (!Number.isFinite(hi) || !Number.isFinite(lo)) return null
  return `${hi >> 8}.${hi & 255}.${lo >> 8}.${lo & 255}`
}

function ipv6Class(host: string): 'loopback' | 'private' | 'public-ip' {
  const value = host.toLowerCase()
  if (value === '::1') return 'loopback'
  if (value === '::') return 'private'
  const mapped = mappedV4(value)
  if (mapped) return ipv4Class(mapped) ?? 'public-ip'
  const head = value.startsWith('::') ? 0 : Number.parseInt(value.split(':')[0] || '0', 16)
  if (head >= 0xfe80 && head <= 0xfebf) return 'private'
  if (head >= 0xfc00 && head <= 0xfdff) return 'private'
  if (head >= 0xfec0 && head <= 0xfeff) return 'private'
  return 'public-ip'
}

function hostClass(hostname: string): 'loopback' | 'private' | 'public' | 'public-ip' | 'other' {
  const host = hostname.toLowerCase().replace(/\.$/, '').replace(/^\[|\]$/g, '')
  if (!host || /[%@/\s]/.test(host)) return 'other'
  const v4 = ipv4Class(host)
  if (v4) return v4
  if (isIPv6(host)) return ipv6Class(host)
  if (LOOPBACK_NAMES.has(host) || host.endsWith('.localhost')) return 'loopback'
  const embedded = /(?:^|\.)(\d{1,3}(?:\.\d{1,3}){3})(?:\.|$)/.exec(host)
  if (embedded?.[1]) {
    const inner = ipv4Class(embedded[1])
    if (inner === 'loopback') return 'loopback'
    if (inner === 'private' || inner === 'public-ip') return 'private'
  }
  if (!host.includes('.')) return 'private'
  if (PRIVATE_SUFFIXES.some((suffix) => host.endsWith(suffix))) return 'private'
  return PUBLIC_HOST.test(host) ? 'public' : 'other'
}

/** Classifies a base URL. The raw string is never copied into the result. */
export function sanitizeEndpoint(raw: unknown, mode: 'remote' | 'stub' = 'remote'): EndpointMeta {
  if (mode === 'stub') return { endpoint_class: 'stub' }
  if (typeof raw !== 'string' || raw.trim() === '') return { endpoint_class: 'invalid' }
  let url: URL
  try { url = new URL(raw) } catch { return { endpoint_class: 'invalid' } }
  const endpoint_scheme = url.protocol === 'http:' ? 'http' : url.protocol === 'https:' ? 'https' : 'other'
  if (endpoint_scheme === 'other') return { endpoint_class: 'other', endpoint_scheme }
  const endpoint_class = hostClass(url.hostname)
  if (endpoint_class === 'public') {
    const endpoint_host = url.hostname.toLowerCase().replace(/\.$/, '')
    return PUBLIC_HOST.test(endpoint_host)
      ? { endpoint_class, endpoint_scheme, endpoint_host }
      : { endpoint_class: 'other', endpoint_scheme }
  }
  if (endpoint_class === 'other') return { endpoint_class, endpoint_scheme }
  return { endpoint_class, endpoint_scheme }
}

function sanitizeToolList(names: unknown): { tools_advertised: number; tool_names: string[] } {
  const list = Array.isArray(names) ? names : []
  return { tools_advertised: list.length, tool_names: list.slice(0, MAX_TOOLS).map((name) => sanitizeToolName(name)) }
}

function sanitizeToolChoice(value: unknown): 'omitted' | 'none' | 'auto' | 'required' | 'other' {
  if (value == null || value === '') return 'omitted'
  if (value === 'none' || value === 'auto' || value === 'required') return value
  return 'other'
}

function toolCallsShape(raw: unknown): { tool_calls_shape: 'absent' | 'array' | 'other'; tool_call_count: number } {
  if (raw == null) return { tool_calls_shape: 'absent', tool_call_count: 0 }
  if (Array.isArray(raw)) return { tool_calls_shape: 'array', tool_call_count: raw.length }
  return { tool_calls_shape: 'other', tool_call_count: 0 }
}

function attemptsOf(value: unknown): number {
  return typeof value === 'number' && Number.isInteger(value) && value > 0 && value < 100 ? value : 1
}

function httpStatus(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isInteger(value) && value >= 100 && value <= 599 ? value : undefined
}

function emit(record: Record<string, unknown>): void {
  try {
    if (!active()) return
    sink(`tool-diag ${JSON.stringify(record)}`)
  } catch { /* diagnostic failures must not affect the turn */ }
}

function responseFields(meta: ChatMeta, noted: CallNote[]): Record<string, unknown> {
  const shape = toolCallsShape(meta.rawToolCalls)
  let calls = noted.slice(0, MAX_CALLS)
  if (meta.observedArgs === false && shape.tool_calls_shape === 'array' && Array.isArray(meta.rawToolCalls)) {
    calls = []
    for (const item of meta.rawToolCalls) {
      if (calls.length >= MAX_CALLS) break
      const name = item && typeof item === 'object' && 'name' in item ? (item as { name?: unknown }).name : undefined
      calls.push({ name: sanitizeToolName(name), arg_type: 'unobserved', parse: 'skipped' })
    }
  }
  return {
    outcome: 'completed',
    attempts: attemptsOf(meta.attempts),
    finish_reason: sanitizeFinishReason(meta.finishReason),
    content_form: contentForm(meta.content),
    ...shape,
    ...(shape.tool_calls_shape === 'array' ? { calls } : {}),
  }
}

export function openChatDiag(input: {
  baseURL: unknown
  model: unknown
  toolNames: unknown
  toolChoice: unknown
  mode?: 'remote' | 'stub'
}): ChatDiag {
  try {
    if (!active()) return noop
    const noted: CallNote[] = []
    let sent = false
    const current = frames.getStore()
    const step = current ? ++current.step : 1
    const id = current?.id ?? openToolDiagFrame().id
    const base = {
      v: 1 as const,
      kind: 'llm' as const,
      id,
      step,
      model: sanitizeModel(input.model),
      ...sanitizeEndpoint(input.baseURL, input.mode ?? 'remote'),
      ...sanitizeToolList(input.toolNames),
      tool_choice: sanitizeToolChoice(input.toolChoice),
    }
    const send = (extra: Record<string, unknown>) => {
      if (sent) return
      sent = true
      emit({ ...base, ...extra })
    }
    return {
      noteCall(name, rawArgs) {
        try {
          if (noted.length >= MAX_CALLS) return
          noted.push({ name: sanitizeToolName(name), ...classifyArguments(rawArgs) })
        } catch { /* ignore */ }
      },
      complete(meta) {
        try { send(responseFields(meta, noted)) } catch { /* ignore */ }
      },
      fail(outcome, meta = {}) {
        try {
          const status = httpStatus(meta.httpStatus)
          send({ ...responseFields(meta, noted), outcome, ...(status === undefined ? {} : { http_status: status }) })
        } catch { /* ignore */ }
      },
    }
  } catch {
    return noop
  }
}

export function recordRoute(input: { watch: boolean; hasComputer: boolean; toolNames: unknown }): void {
  try {
    if (!active()) return
    const current = frames.getStore()
    emit({
      v: 1,
      kind: 'route',
      id: current?.id ?? 'missing',
      watch: input.watch === true,
      has_computer: input.hasComputer === true,
      ...sanitizeToolList(input.toolNames),
    })
  } catch { /* ignore */ }
}

export function recordDispatch(tool: unknown, dispatch: DispatchClass, error?: unknown): void {
  try {
    if (!active()) return
    const current = frames.getStore()
    const record: Record<string, unknown> = {
      v: 1,
      kind: 'dispatch',
      id: current?.id ?? 'missing',
      step: current?.step ?? 0,
      tool: sanitizeToolName(tool),
      dispatch,
    }
    if (dispatch === 'error') Object.assign(record, classifyError(error))
    emit(record)
  } catch { /* ignore */ }
}

function classifyError(error: unknown): { error_class: 'abort' | 'coded' | 'error' | 'other'; error_code?: string } {
  try {
    if (error instanceof CodedError) {
      return ERROR_CODE.test(error.code) ? { error_class: 'coded', error_code: error.code } : { error_class: 'coded' }
    }
    if (error instanceof Error && error.name === 'AbortError') return { error_class: 'abort' }
    if (error instanceof Error) return { error_class: 'error' }
    return { error_class: 'other' }
  } catch {
    return { error_class: 'other' }
  }
}
