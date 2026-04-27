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
  effectiveRunnerMode?: SnowflakeRunnerMode;
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

interface EncodeResultPayload {
  vectors: number[][];
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

export const DEFAULT_SNOWFLAKE_MODEL_URL =
  "https://huggingface.co/chux0519/snowflake-arctic-embed-m-v2.0-gguf-embeddings-cpp/resolve/main/snowflake-arctic-embed-m-v2.0.q4_k_mlp_q8_attn.gguf";

const DEFAULT_RUNTIME_BASE_PATH = "/";
const DEFAULT_TOKENIZER_JSON_PATH = "/demo/browser-wasm/assets/snowflake-tokenizer.json";
const DEFAULT_TOKENIZER_SCRIPT_PATH = "/demo/browser-wasm/vendor/web-tokenizers.js";
const DEFAULT_FILE_CACHE = "embeddings-browser-files-v1";
const DEFAULT_MODEL_DB = "embeddings-browser-models-v1";
const DEFAULT_MODEL_STORE = "files";
const RUNTIME_ASSET_VERSION = "webpkg21";

const BUILD_DIRS: Record<ResolvedRuntime, string> = {
  wasm: "build-wasm-web-dyn",
  pthread: "build-wasm-web-pthread-dyn",
  webgpu: "build-wasm-webgpu-browser-dyn",
};

function browserOrigin(): string {
  if (typeof window === "undefined" || !window.location?.origin) {
    return "";
  }
  return window.location.origin;
}

function nowMs(): number {
  if (typeof performance !== "undefined" && typeof performance.now === "function") {
    return performance.now();
  }
  return Date.now();
}

async function readResponseBytes(
  response: Response,
  onProgress?: (loaded: number, total: number | null) => void,
): Promise<Uint8Array> {
  if (!response.body) {
    const bytes = new Uint8Array(await response.arrayBuffer());
    onProgress?.(bytes.byteLength, bytes.byteLength);
    return bytes;
  }

  const totalHeader = response.headers.get("content-length");
  const total = totalHeader ? Number.parseInt(totalHeader, 10) : null;
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;

  while (true) {
    const { value, done } = await reader.read();
    if (done) {
      break;
    }
    if (!value) {
      continue;
    }
    chunks.push(value);
    loaded += value.byteLength;
    onProgress?.(loaded, Number.isFinite(total) ? total : null);
  }

  const out = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out;
}

function joinUrl(base: string, path: string): string {
  return new URL(path, base.endsWith("/") ? base : `${base}/`).toString();
}

function withVersion(url: string, version = RUNTIME_ASSET_VERSION): string {
  const next = new URL(url, browserOrigin() || undefined);
  next.searchParams.set("v", version);
  return next.toString();
}

function ensureBrowser(): void {
  if (typeof window === "undefined" || typeof document === "undefined") {
    throw new Error("@embeddings-cpp/web is browser-only");
  }
}

async function detectRuntime(preferred: SnowflakeRuntime): Promise<ResolvedRuntime> {
  if (preferred !== "auto") {
    return preferred;
  }
  return "wasm";
}

let tokenizerScriptPromise: Promise<void> | null = null;

function loadScriptOnce(sourceUrl: string, scriptText: string): Promise<void> {
  if (tokenizerScriptPromise) {
    return tokenizerScriptPromise;
  }

  tokenizerScriptPromise = new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[data-snowflake-tokenizers="1"]`);
    if (existing) {
      resolve();
      return;
    }

    const script = document.createElement("script");
    const blob = new Blob([scriptText], { type: "application/javascript" });
    const blobUrl = URL.createObjectURL(blob);
    script.src = blobUrl;
    script.async = false;
    script.dataset.snowflakeTokenizers = "1";
    script.onload = () => {
      URL.revokeObjectURL(blobUrl);
      resolve();
    };
    script.onerror = () => {
      URL.revokeObjectURL(blobUrl);
      reject(new Error(`failed to evaluate tokenizer script: ${sourceUrl}`));
    };
    document.head.appendChild(script);
  });

  return tokenizerScriptPromise;
}

function fileCacheKey(kind: string, url: string): string {
  return `/__cache__/${kind}/${encodeURIComponent(url)}`;
}

function openModelDb(): Promise<IDBDatabase | null> {
  if (typeof indexedDB === "undefined") {
    return Promise.resolve(null);
  }

  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DEFAULT_MODEL_DB, 1);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(DEFAULT_MODEL_STORE)) {
        db.createObjectStore(DEFAULT_MODEL_STORE);
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error("failed to open model db"));
  });
}

async function readModelCache(url: string): Promise<Uint8Array | null> {
  const db = await openModelDb();
  if (!db) {
    return null;
  }

  try {
    return await new Promise<Uint8Array | null>((resolve, reject) => {
      const tx = db.transaction(DEFAULT_MODEL_STORE, "readonly");
      const store = tx.objectStore(DEFAULT_MODEL_STORE);
      const req = store.get(url);
      req.onsuccess = () => {
        const value = req.result;
        if (value instanceof ArrayBuffer) {
          resolve(new Uint8Array(value));
          return;
        }
        if (value?.buffer instanceof ArrayBuffer) {
          resolve(new Uint8Array(value.buffer));
          return;
        }
        resolve(null);
      };
      req.onerror = () => reject(req.error ?? new Error("failed to read model cache"));
    });
  } finally {
    db.close();
  }
}

async function writeModelCache(url: string, bytes: Uint8Array): Promise<void> {
  const db = await openModelDb();
  if (!db) {
    return;
  }

  try {
    await new Promise<void>((resolve, reject) => {
      const tx = db.transaction(DEFAULT_MODEL_STORE, "readwrite");
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error ?? new Error("failed to write model cache"));
      tx.objectStore(DEFAULT_MODEL_STORE).put(bytes.buffer.slice(0), url);
    });
  } finally {
    db.close();
  }
}

async function fetchMaybeCached(url: string, cacheEnabled: boolean, kind: string): Promise<Response> {
  if (!cacheEnabled || typeof caches === "undefined") {
    return fetch(url);
  }

  const cache = await caches.open(DEFAULT_FILE_CACHE);
  const key = fileCacheKey(kind, url);
  const cached = await cache.match(key);
  if (cached) {
    return cached;
  }

  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`failed to fetch ${url}: ${response.status}`);
  }
  await cache.put(key, response.clone());
  return response;
}

class BrowserSnowflakeEmbedder implements SnowflakeEmbedder {
  private readonly options: Required<SnowflakeEmbedderOptions>;
  private readonly runtime: ResolvedRuntime;
  private tokenizer: TokenizerLike | null = null;
  private iframe: HTMLIFrameElement | null = null;
  private queue: Promise<void> = Promise.resolve();
  private requestId = 0;

  private effectiveRunnerMode(): SnowflakeRunnerMode {
    if (this.options.runnerMode !== "persistent") {
      return this.options.runnerMode;
    }
    if (this.runtime === "wasm" || this.runtime === "webgpu") {
      return "persistent";
    }
    return "ephemeral";
  }

  private usesRunnerApiTransport(): boolean {
    return this.runtime !== "wasm" || this.effectiveRunnerMode() === "persistent";
  }

  private shouldRecycleRunnerPerRequest(): boolean {
    return this.options.runnerMode === "persistent" && this.effectiveRunnerMode() !== "persistent";
  }

  constructor(options: Required<SnowflakeEmbedderOptions>, runtime: ResolvedRuntime) {
    this.options = options;
    this.runtime = runtime;
  }

  private emit(stage: string, detail?: string): void {
    this.options.onStatus({ stage, detail, runtime: this.runtime, atMs: nowMs() });
  }

  private progressDetail(url: string, loaded: number, total: number | null): string {
    const percent = total && total > 0 ? ((loaded / total) * 100).toFixed(1) : null;
    const parts = [url, `${(loaded / (1024 * 1024)).toFixed(2)} MiB`];
    if (total && total > 0) {
      parts.push(`/ ${(total / (1024 * 1024)).toFixed(2)} MiB`);
    }
    if (percent) {
      parts.push(`(${percent}%)`);
    }
    return parts.join(" ");
  }

  private async prefetchModelToIndexedDb(): Promise<void> {
    if (!this.options.cache) {
      return;
    }

    const cached = await readModelCache(this.options.modelUrl);
    if (cached) {
      this.emit("model-idb-hit", `${(cached.byteLength / (1024 * 1024)).toFixed(2)} MiB`);
      return;
    }

    this.emit("model-idb-miss", this.options.modelUrl);
    const response = await fetch(this.options.modelUrl);
    if (!response.ok) {
      throw new Error(`failed to prefetch model ${this.options.modelUrl}: ${response.status}`);
    }
    const bytes = await readResponseBytes(response, (loaded, total) => {
      this.emit("model-idb-progress", this.progressDetail(this.options.modelUrl, loaded, total));
    });
    await writeModelCache(this.options.modelUrl, bytes);
    this.emit("model-idb-ready", `${(bytes.byteLength / (1024 * 1024)).toFixed(2)} MiB`);
  }

  info(): SnowflakeEmbedderInfo {
    return {
      modelUrl: this.options.modelUrl,
      runtime: this.runtime,
      runnerMode: this.options.runnerMode,
      effectiveRunnerMode: this.effectiveRunnerMode(),
      dim: 768,
      normalized: true,
    };
  }

  async prefetch(): Promise<void> {
    this.emit("prefetch-start");
    if (this.options.runnerMode !== this.effectiveRunnerMode()) {
      this.emit("runner-mode-fallback", `${this.options.runnerMode} -> ${this.effectiveRunnerMode()}`);
    }
    await this.ensureTokenizer();
    await this.prefetchFiles();
    this.emit("prefetch-ready");
  }

  async embed(text: string): Promise<SnowflakeEmbedding> {
    const [result] = await this.embedAll([text]);
    return result;
  }

  async embedAll(texts: string[]): Promise<SnowflakeEmbedding[]> {
    if (texts.length === 0) {
      return [];
    }

    return this.enqueue(async () => {
      if (this.options.runnerMode !== this.effectiveRunnerMode()) {
        this.emit("runner-mode-fallback", `${this.options.runnerMode} -> ${this.effectiveRunnerMode()}`);
      }
      this.emit("tokenizer-encode-start");
      const tokenizer = await this.ensureTokenizer();
      const lines: string[] = [];
      const tokenCounts: number[] = [];

      for (const text of texts) {
        const trimmed = text.trim();
        if (!trimmed) {
          throw new Error("embed text is empty");
        }
        const ids = tokenizer.encode(trimmed);
        tokenCounts.push(ids.length);
        lines.push(this.encodeBatchLine(ids));
      }

      this.emit("tokenizer-encode-ready", `${texts.length} item(s)`);
      const result = await this.runEncode(lines.join("\n"));
      if (!result.vectors || result.vectors.length !== texts.length) {
        throw new Error("unexpected embedding result shape");
      }

      this.emit("embed-done", `${result.vectors.length} item(s)`);
      return result.vectors.map((vector, index) => ({
        vector: Float32Array.from(vector),
        tokenCount: tokenCounts[index],
        runtime: this.runtime,
        runnerMode: this.effectiveRunnerMode(),
      }));
    });
  }

  async dispose(): Promise<void> {
    this.tokenizer?.dispose?.();
    this.tokenizer = null;

    if (this.iframe) {
      this.iframe.remove();
      this.iframe = null;
    }
    this.emit("disposed");
  }

  private async enqueue<T>(fn: () => Promise<T>): Promise<T> {
    const previous = this.queue;
    let release: () => void = () => {};
    this.queue = new Promise<void>((resolve) => {
      release = resolve;
    });

    await previous;
    try {
      return await fn();
    } finally {
      release();
    }
  }

  private runtimeBaseUrl(): string {
    return this.options.runtimeBaseUrl;
  }

  private tokenizerJsonUrl(): string {
    return this.options.tokenizerUrl;
  }

  private tokenizerScriptUrl(): string {
    return this.options.tokenizerScriptUrl;
  }

  private buildDir(): string {
    return BUILD_DIRS[this.runtime];
  }

  private runtimeAssetUrl(path: string): string {
    return withVersion(joinUrl(this.runtimeBaseUrl(), path));
  }

  private iframeUrl(): string {
    const params = new URLSearchParams({
      build: this.buildDir(),
      model_url: this.options.modelUrl,
      pooling: "cls",
      v: RUNTIME_ASSET_VERSION,
    });

    if (this.runtime === "webgpu") {
      params.set("backend", "webgpu");
    }
    if (this.runtime === "pthread") {
      params.set("threads", String(this.options.threads));
    }

    const runnerPage =
      this.usesRunnerApiTransport()
        ? "/scripts/wasm_persistent_encode_page.html"
        : "/scripts/wasm_encode_page.html";

    return `${joinUrl(this.runtimeBaseUrl(), runnerPage)}?${params.toString()}`;
  }

  private async ensureTokenizer(): Promise<TokenizerLike> {
    if (this.tokenizer) {
      this.emit("tokenizer-ready", "reused");
      return this.tokenizer;
    }

    this.emit("tokenizer-script-loading", this.tokenizerScriptUrl());
    const scriptResponse = await fetchMaybeCached(this.tokenizerScriptUrl(), this.options.cache, "tokenizer-script");
    if (!scriptResponse.ok) {
      throw new Error(`failed to fetch tokenizer script: ${scriptResponse.status}`);
    }
    const scriptBytes = await readResponseBytes(scriptResponse, (loaded, total) => {
      this.emit("tokenizer-script-progress", this.progressDetail(this.tokenizerScriptUrl(), loaded, total));
    });
    this.emit("tokenizer-script-fetched");
    const scriptText = new TextDecoder().decode(scriptBytes);
    this.emit("tokenizer-script-injecting");
    await loadScriptOnce(this.tokenizerScriptUrl(), scriptText);
    this.emit("tokenizer-script-ready");
    this.emit("tokenizer-json-loading", this.tokenizerJsonUrl());
    const response = await fetchMaybeCached(this.tokenizerJsonUrl(), this.options.cache, "tokenizer");
    if (!response.ok) {
      throw new Error(`failed to load tokenizer JSON: ${response.status}`);
    }
    const tokenizers = window.tokenizers;
    if (!tokenizers?.Tokenizer) {
      throw new Error("web-tokenizers runtime did not initialize");
    }
    const jsonBytes = await readResponseBytes(response, (loaded, total) => {
      this.emit("tokenizer-json-progress", this.progressDetail(this.tokenizerJsonUrl(), loaded, total));
    });
    this.emit("tokenizer-json-ready");
    this.emit("tokenizer-fromjson-start");
    const jsonBuffer = jsonBytes.slice().buffer;
    this.tokenizer = await tokenizers.Tokenizer.fromJSON(jsonBuffer);
    this.emit("tokenizer-fromjson-ready");
    this.emit("tokenizer-ready");
    return this.tokenizer;
  }

  private async prefetchFiles(): Promise<void> {
    const assets = [
      this.runtimeAssetUrl("/scripts/wasm_encode_page.html"),
      this.runtimeAssetUrl("/scripts/wasm_persistent_encode_page.html"),
      this.tokenizerScriptUrl(),
      this.tokenizerJsonUrl(),
      this.runtimeAssetUrl(
        `/${this.buildDir()}/${this.usesRunnerApiTransport() ? "embedding_wasm_model_runner" : "embedding_wasm_model_encode"}.js`,
      ),
      this.runtimeAssetUrl(
        `/${this.buildDir()}/${this.usesRunnerApiTransport() ? "embedding_wasm_model_runner" : "embedding_wasm_model_encode"}.wasm`,
      ),
      this.options.modelUrl,
    ];

    for (const asset of assets) {
      if (asset === this.options.modelUrl) {
        await this.prefetchModelToIndexedDb();
        continue;
      }
      const kind = "asset";
      this.emit("prefetch-asset", asset);
      const response = await fetchMaybeCached(asset, this.options.cache, kind);
      if (!response.ok) {
        throw new Error(`failed to prefetch ${asset}: ${response.status}`);
      }
    }
  }

  private encodeBatchLine(ids: Int32Array): string {
    const fullIds = [0, ...Array.from(ids), 2];
    const mask = fullIds.map(() => 1);
    return `${fullIds.join(",")}\t${mask.join(",")}`;
  }

  private async ensureIframe(): Promise<HTMLIFrameElement> {
    if (!this.iframe) {
      this.emit("iframe-create");
      this.iframe = document.createElement("iframe");
      this.iframe.hidden = true;
      this.iframe.title = "snowflake-web-runner";
      document.body.appendChild(this.iframe);
    }

    const targetUrl = this.iframeUrl();
    if (this.iframe.src !== targetUrl) {
      this.emit("runtime-page-loading", targetUrl);
      this.iframe.src = targetUrl;
      await new Promise<void>((resolve) => {
        this.iframe!.onload = () => resolve();
      });
      this.emit("runtime-page-ready");
    }

    return this.iframe;
  }

  private resetIframe(): void {
    if (this.iframe) {
      this.iframe.remove();
      this.iframe = null;
      this.emit("iframe-reset");
    }
  }

  private async runEncode(batchLine: string): Promise<EncodeResultPayload> {
    const iframe = await this.ensureIframe();
    return new Promise<EncodeResultPayload>((resolve, reject) => {
      const self = this;
      const requestId = ++this.requestId;
      const effectiveRunnerMode = this.effectiveRunnerMode();
      let lastStage = "";
      let lastDetail = "";
      let requestReceived = false;
      let sendLogged = false;
      const sendRequest = () => {
        if (!sendLogged) {
          sendLogged = true;
          self.emit("encode-request-sent");
        }
        iframe.contentWindow?.postMessage({ type: "encode-request", batchLine, requestId }, "*");
      };
      const resend = window.setInterval(() => {
        if (!requestReceived) {
          sendRequest();
        }
      }, 500);
      const timeout = window.setTimeout(() => {
        window.clearInterval(resend);
        window.removeEventListener("message", onMessage);
        self.resetIframe();
        const suffix = lastStage ? ` after ${lastStage}${lastDetail ? `: ${lastDetail}` : ""}` : "";
        reject(new Error(`encode request timed out${suffix}`));
      }, 120000);

      function onMessage(event: MessageEvent): void {
        if (event.source !== iframe.contentWindow) {
          return;
        }
        const data = event.data as
          | {
              type?: string;
              result?: EncodeResultPayload;
              error?: string;
              stage?: string;
              detail?: string;
              atMs?: number;
              requestId?: number;
            }
          | undefined;
        if (!data) {
          return;
        }
        if (data.type === "encode-status") {
          if (data.requestId != null && data.requestId !== requestId) {
            return;
          }
          lastStage = data.stage || "encode-status";
          lastDetail = data.detail || "";
          self.emit(lastStage, lastDetail);
          if (lastStage === "encode-request-received") {
            requestReceived = true;
            window.clearInterval(resend);
          }
          return;
        }
        if (data.type !== "encode-result") {
          return;
        }
        if (data.requestId !== requestId && !(data.requestId == null && data.error)) {
          return;
        }
        window.clearTimeout(timeout);
        window.clearInterval(resend);
        window.removeEventListener("message", onMessage);
        if (effectiveRunnerMode === "ephemeral" || self.shouldRecycleRunnerPerRequest()) {
          self.resetIframe();
        }
        if (data.error || !data.result) {
          self.resetIframe();
          reject(new Error(data.error || `encode failed after ${lastStage || "unknown-stage"}${lastDetail ? `: ${lastDetail}` : ""}`));
          return;
        }
        resolve(data.result);
      }

      window.addEventListener("message", onMessage);
      sendRequest();
    });
  }
}

export async function createSnowflakeEmbedder(
  options: SnowflakeEmbedderOptions,
): Promise<SnowflakeEmbedder> {
  ensureBrowser();

  const runtimeBaseUrl = options.runtimeBaseUrl ?? (browserOrigin() || DEFAULT_RUNTIME_BASE_PATH);
  const tokenizerUrl =
    options.tokenizerUrl ?? joinUrl(runtimeBaseUrl, DEFAULT_TOKENIZER_JSON_PATH);
  const tokenizerScriptUrl =
    options.tokenizerScriptUrl ?? joinUrl(runtimeBaseUrl, DEFAULT_TOKENIZER_SCRIPT_PATH);
  const detectedThreads = options.threads ?? navigator.hardwareConcurrency ?? 4;

  const resolved: Required<SnowflakeEmbedderOptions> = {
    modelUrl: options.modelUrl || DEFAULT_SNOWFLAKE_MODEL_URL,
    runtime: options.runtime ?? "auto",
    runnerMode: options.runnerMode ?? "ephemeral",
    runtimeBaseUrl,
    tokenizerUrl,
    tokenizerScriptUrl,
    threads: Math.max(1, Math.min(12, detectedThreads)),
    cache: options.cache ?? true,
    onStatus: options.onStatus ?? (() => {}),
  };

  const runtime = await detectRuntime(resolved.runtime);
  resolved.onStatus({ stage: "runtime-selected", runtime, detail: runtime });
  return new BrowserSnowflakeEmbedder(resolved, runtime);
}

export function createSnowflakeDefaults(
  overrides: Partial<SnowflakeEmbedderOptions> = {},
): SnowflakeEmbedderOptions {
  return {
    modelUrl: DEFAULT_SNOWFLAKE_MODEL_URL,
    runtime: "auto",
    runnerMode: "ephemeral",
    runtimeBaseUrl: browserOrigin() || DEFAULT_RUNTIME_BASE_PATH,
    cache: true,
    ...overrides,
  };
}
