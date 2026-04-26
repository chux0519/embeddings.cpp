export type SnowflakeRuntime = "auto" | "webgpu" | "pthread" | "wasm";
export type SnowflakeRunnerMode = "ephemeral" | "persistent";
type ResolvedRuntime = Exclude<SnowflakeRuntime, "auto">;
export interface SnowflakeStatusEvent {
    stage: string;
    detail?: string;
    runtime?: ResolvedRuntime;
    atMs?: number;
}
export interface SnowflakeEmbedderOptions {
    modelUrl: string;
    runtime?: SnowflakeRuntime;
    runnerMode?: SnowflakeRunnerMode;
    runtimeBaseUrl?: string;
    tokenizerUrl?: string;
    tokenizerScriptUrl?: string;
    threads?: number;
    cache?: boolean;
    onStatus?: (event: SnowflakeStatusEvent) => void;
}
export interface SnowflakeEmbedding {
    vector: Float32Array;
    tokenCount: number;
    runtime: ResolvedRuntime;
    runnerMode: SnowflakeRunnerMode;
}
export interface SnowflakeEmbedderInfo {
    modelUrl: string;
    runtime: ResolvedRuntime | "pending";
    runnerMode: SnowflakeRunnerMode;
    dim: 768;
    normalized: true;
}
export interface SnowflakeEmbedder {
    prefetch(): Promise<void>;
    embed(text: string): Promise<SnowflakeEmbedding>;
    embedAll(texts: string[]): Promise<SnowflakeEmbedding[]>;
    info(): SnowflakeEmbedderInfo;
    dispose(): Promise<void>;
}
interface TokenizerLike {
    encode(text: string): Int32Array;
    dispose?: () => void;
}
interface BrowserTokenizersNamespace {
    Tokenizer: {
        fromJSON(json: ArrayBuffer): Promise<TokenizerLike>;
    };
}
declare global {
    interface Navigator {
        gpu?: {
            requestAdapter(): Promise<unknown>;
        };
    }
    interface Window {
        tokenizers?: BrowserTokenizersNamespace;
    }
}
export declare const DEFAULT_SNOWFLAKE_MODEL_URL = "https://huggingface.co/chux0519/snowflake-arctic-embed-m-v2.0-gguf-embeddings-cpp/resolve/main/snowflake-arctic-embed-m-v2.0.q4_k_mlp_q8_attn.gguf";
export declare function createSnowflakeEmbedder(options: SnowflakeEmbedderOptions): Promise<SnowflakeEmbedder>;
export declare function createSnowflakeDefaults(overrides?: Partial<SnowflakeEmbedderOptions>): SnowflakeEmbedderOptions;
export {};
