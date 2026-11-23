import type { Context } from '@deepseek-ai/cordis';
import z from '@deepseek-ai/schemastery';
/** MLX server process manager for DSH. */
export declare const name = "mlx-local";
export declare const inject: string[];
/** Configuration for the MLX FastAPI server process. */
export interface Config {
    /** Server base URL used to validate configuration. */
    baseURL?: string;
    /** Shell command used to start the server. */
    command?: string;
    /** Working directory for the server command. */
    workdir?: string;
}
export declare const Config: z<Config>;
/** Start the MLX server and let the shell service own its process lifetime. */
export declare function apply(ctx: Context, config: Config): void;
//# sourceMappingURL=index.d.ts.map