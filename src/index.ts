import type { Context } from '@deepseek-ai/cordis'
import z from '@deepseek-ai/schemastery'
import type {} from '@deepseek-ai/dsh-shell'

/** MLX server process manager for DSH. */
export const name = 'mlx-local'
export const inject = ['shell']

/** Configuration for the MLX FastAPI server process. */
export interface Config {
  /** Server base URL used to validate configuration. */
  baseURL?: string
  /** Shell command used to start the server. */
  command?: string
  /** Working directory for the server command. */
  workdir?: string
}

export const Config: z<Config> = z.object({
  baseURL: z.string().default('http://127.0.0.1:8000'),
  command: z.string().default('uv run uvicorn app.main:app --host 127.0.0.1 --port 8000'),
  workdir: z.string().default('/Users/dylan/Workspace/AI/mlx-lm-local-demo'),
})

type ResolvedConfig = Required<Config>

/** Start the MLX server and let the shell service own its process lifetime. */
export function apply(ctx: Context, config: Config): void {
  const resolved = config as ResolvedConfig
  const baseURL = resolved.baseURL.trim().replace(/\/$/, '')
  if (!baseURL || !/^https?:\/\//.test(baseURL)) throw new Error('mlx-local: baseURL must be a non-empty http(s) URL')
  if (!resolved.command.trim()) throw new Error('mlx-local: command must not be empty')
  if (!resolved.workdir.trim()) throw new Error('mlx-local: workdir must not be empty')

  const server = ctx.shell.resolve({
    command: resolved.command,
    workdir: resolved.workdir,
    timeoutMs: 2147000000,
    onExpiry: 'none',
  })
  const executionPromise = ctx.shell.execute(server)
  ctx.effect(() => async () => {
    try {
      const process = await executionPromise
      process.kill()
      await process.done
    } catch (error) {
      ctx.logger.error('mlx-local server cleanup failed: ' + String(error))
    }
  })
  void executionPromise.then(async execution => {
    try {
      const result = await execution.result()
      if (result.exitCode !== 0) ctx.logger.error('mlx-local server exited with code ' + String(result.exitCode))
    } catch (error) {
      ctx.logger.error('mlx-local server process failed: ' + String(error))
    }
  }).catch(error => {
    ctx.logger.error('mlx-local server failed to start: ' + String(error))
  })
}
