export type SnowflakeRuntime = "auto" | "webgpu" | "pthread" | "wasm";
type ResolvedRuntime = Exclude<SnowflakeRuntime, "auto">;
export interface SnowflakeEmbedderOptions {
    modelUrl: string;
    runtime?: SnowflakeRuntime;
    runtimeBaseUrl?: string;
    tokenizerUrl?: string;
    tokenizerScriptUrl?: string;
    threads?: number;
    cache?: boolean;
}
export interface SnowflakeEmbedding {
    vector: Float32Array;
    tokenCount: number;
    runtime: ResolvedRuntime;
}
export interface SnowflakeEmbedderInfo {
    modelUrl: string;
    runtime: ResolvedRuntime | "pending";
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
