import z from '@deepseek-ai/schemastery';
import { spawn } from 'node:child_process';
/** MLX server process manager for DSH. */
export const name = 'mlx-local';
export const inject = [];
export const Config = z.object({
    baseURL: z.string().default('http://127.0.0.1:8000'),
    command: z.string().default('uv run uvicorn app.main:app --host 127.0.0.1 --port 8000'),
    workdir: z.string().default('/Users/dylan/Workspace/AI/mlx-lm-local-demo'),
});
/** Start the MLX server as a native child process, outside DSH shell execution. */
export function apply(ctx, config) {
    if (process.platform !== 'darwin') {
        ctx.logger.warn(`mlx-local: unsupported platform ${process.platform}; MLX requires macOS, server not started`);
        return;
    }
    const resolved = config;
    const baseURL = resolved.baseURL.trim().replace(/\/$/, '');
    if (!baseURL || !/^https?:\/\//.test(baseURL))
        throw new Error('mlx-local: baseURL must be a non-empty http(s) URL');
    if (!resolved.command.trim())
        throw new Error('mlx-local: command must not be empty');
    if (!resolved.workdir.trim())
        throw new Error('mlx-local: workdir must not be empty');
    const child = spawn('/bin/zsh', ['-lc', resolved.command], {
        cwd: resolved.workdir,
        stdio: ['ignore', 'pipe', 'pipe'],
        env: { ...process.env },
    });
    child.stdout?.on('data', data => ctx.logger.info('mlx-local: ' + String(data).trimEnd()));
    child.stderr?.on('data', data => ctx.logger.warn('mlx-local: ' + String(data).trimEnd()));
    child.on('error', error => ctx.logger.error('mlx-local native server failed to start: ' + String(error)));
    child.on('exit', (code, signal) => {
        if (code !== 0)
            ctx.logger.error(`mlx-local native server exited (code=${code}, signal=${signal})`);
    });
    ctx.effect(() => async () => {
        if (!child.killed)
            child.kill('SIGTERM');
    });
}
//# sourceMappingURL=index.js.map
