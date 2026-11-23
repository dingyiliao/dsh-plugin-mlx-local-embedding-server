import z from '@deepseek-ai/schemastery';
/** MLX server process manager for DSH. */
export const name = 'mlx-local';
export const inject = ['shell'];
export const Config = z.object({
    baseURL: z.string().default('http://127.0.0.1:8000'),
    command: z.string().default('uv run uvicorn app.main:app --host 127.0.0.1 --port 8000'),
    workdir: z.string().default('/Users/dylan/Workspace/AI/mlx-lm-local-demo'),
});
/** Start the MLX server and let the shell service own its process lifetime. */
export function apply(ctx, config) {
    const resolved = config;
    const baseURL = resolved.baseURL.trim().replace(/\/$/, '');
    if (!baseURL || !/^https?:\/\//.test(baseURL))
        throw new Error('mlx-local: baseURL must be a non-empty http(s) URL');
    if (!resolved.command.trim())
        throw new Error('mlx-local: command must not be empty');
    if (!resolved.workdir.trim())
        throw new Error('mlx-local: workdir must not be empty');
    const server = ctx.shell.resolve({
        command: resolved.command,
        workdir: resolved.workdir,
        timeoutMs: 2147000000,
        onExpiry: 'none',
    });
    const executionPromise = ctx.shell.execute(server);
    ctx.effect(() => async () => {
        try {
            const process = await executionPromise;
            process.kill();
            await process.done;
        }
        catch (error) {
            ctx.logger.error('mlx-local server cleanup failed: ' + String(error));
        }
    });
    void executionPromise.then(async (execution) => {
        try {
            const result = await execution.result();
            if (result.exitCode !== 0)
                ctx.logger.error('mlx-local server exited with code ' + String(result.exitCode));
        }
        catch (error) {
            ctx.logger.error('mlx-local server process failed: ' + String(error));
        }
    }).catch(error => {
        ctx.logger.error('mlx-local server failed to start: ' + String(error));
    });
}
//# sourceMappingURL=index.js.map