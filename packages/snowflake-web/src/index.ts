export type SnowflakeRuntime = "auto" | "webgpu" | "pthread" | "wasm";

export interface SnowflakeEmbedderOptions {
  modelUrl: string;
  runtime?: SnowflakeRuntime;
  runtimeBaseUrl?: string;
  tokenizerUrl?: string;
  threads?: number;
  cache?: boolean;
}

export interface SnowflakeEmbedding {
  vector: Float32Array;
  tokenCount: number;
  runtime: Exclude<SnowflakeRuntime, "auto">;
}

export interface SnowflakeEmbedderInfo {
  modelUrl: string;
  runtime: Exclude<SnowflakeRuntime, "auto"> | "pending";
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

export const DEFAULT_SNOWFLAKE_MODEL_URL =
  "https://huggingface.co/chux0519/snowflake-arctic-embed-m-v2.0-gguf-embeddings-cpp/resolve/main/snowflake-arctic-embed-m-v2.0.q4_k_mlp_q8_attn.gguf";

export const DEFAULT_SNOWFLAKE_TOKENIZER_URL =
  "https://raw.githubusercontent.com/chux0519/embeddings.cpp/main/demo/browser-wasm/assets/snowflake-tokenizer.json";

export function createSnowflakeDefaults(
  overrides: Partial<SnowflakeEmbedderOptions> = {},
): SnowflakeEmbedderOptions {
  return {
    modelUrl: DEFAULT_SNOWFLAKE_MODEL_URL,
    runtime: "auto",
    cache: true,
    ...overrides,
  };
}
