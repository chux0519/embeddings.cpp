export const DEFAULT_SNOWFLAKE_MODEL_URL = "https://huggingface.co/chux0519/snowflake-arctic-embed-m-v2.0-gguf-embeddings-cpp/resolve/main/snowflake-arctic-embed-m-v2.0.q4_k_mlp_q8_attn.gguf";
const DEFAULT_RUNTIME_BASE_PATH = "/";
const DEFAULT_TOKENIZER_JSON_PATH = "/demo/browser-wasm/assets/snowflake-tokenizer.json";
const DEFAULT_TOKENIZER_SCRIPT_PATH = "/demo/browser-wasm/vendor/web-tokenizers.js";
const DEFAULT_FILE_CACHE = "embeddings-browser-files-v1";
const BUILD_DIRS = {
    wasm: "build-wasm-web-dyn",
    pthread: "build-wasm-web-pthread-dyn",
    webgpu: "build-wasm-webgpu-browser-dyn",
};
function browserOrigin() {
    if (typeof window === "undefined" || !window.location?.origin) {
        return "";
    }
    return window.location.origin;
}
function nowMs() {
    if (typeof performance !== "undefined" && typeof performance.now === "function") {
        return performance.now();
    }
    return Date.now();
}
function joinUrl(base, path) {
    return new URL(path, base.endsWith("/") ? base : `${base}/`).toString();
}
function ensureBrowser() {
    if (typeof window === "undefined" || typeof document === "undefined") {
        throw new Error("@embeddings-cpp/web is browser-only");
    }
}
async function detectRuntime(preferred) {
    if (preferred !== "auto") {
        return preferred;
    }
    if (navigator.gpu) {
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (adapter) {
                return "webgpu";
            }
        }
        catch {
            // ignore and fall back
        }
    }
    if (window.crossOriginIsolated) {
        return "pthread";
    }
    return "wasm";
}
let tokenizerScriptPromise = null;
function loadScriptOnce(src) {
    if (tokenizerScriptPromise) {
        return tokenizerScriptPromise;
    }
    tokenizerScriptPromise = new Promise((resolve, reject) => {
        const existing = document.querySelector(`script[data-snowflake-tokenizers="1"][src="${src}"]`);
        if (existing) {
            resolve();
            return;
        }
        const script = document.createElement("script");
        script.src = src;
        script.async = true;
        script.dataset.snowflakeTokenizers = "1";
        script.onload = () => resolve();
        script.onerror = () => reject(new Error(`failed to load tokenizer script: ${src}`));
        document.head.appendChild(script);
    });
    return tokenizerScriptPromise;
}
function fileCacheKey(kind, url) {
    return `/__cache__/${kind}/${encodeURIComponent(url)}`;
}
async function fetchMaybeCached(url, cacheEnabled, kind) {
    if (!cacheEnabled || typeof caches === "undefined") {
        return fetch(url, { cache: "reload" });
    }
    const cache = await caches.open(DEFAULT_FILE_CACHE);
    const key = fileCacheKey(kind, url);
    const cached = await cache.match(key);
    if (cached) {
        return cached;
    }
    const response = await fetch(url, { cache: "reload" });
    if (!response.ok) {
        throw new Error(`failed to fetch ${url}: ${response.status}`);
    }
    await cache.put(key, response.clone());
    return response;
}
class BrowserSnowflakeEmbedder {
    options;
    runtime;
    tokenizer = null;
    iframe = null;
    queue = Promise.resolve();
    constructor(options, runtime) {
        this.options = options;
        this.runtime = runtime;
    }
    emit(stage, detail) {
        this.options.onStatus({ stage, detail, runtime: this.runtime, atMs: nowMs() });
    }
    info() {
        return {
            modelUrl: this.options.modelUrl,
            runtime: this.runtime,
            dim: 768,
            normalized: true,
        };
    }
    async prefetch() {
        this.emit("prefetch-start");
        await this.ensureTokenizer();
        await this.prefetchFiles();
        this.emit("prefetch-ready");
    }
    async embed(text) {
        const [result] = await this.embedAll([text]);
        return result;
    }
    async embedAll(texts) {
        if (texts.length === 0) {
            return [];
        }
        return this.enqueue(async () => {
            this.emit("tokenizer-encode-start");
            const tokenizer = await this.ensureTokenizer();
            const lines = [];
            const tokenCounts = [];
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
            }));
        });
    }
    async dispose() {
        this.tokenizer?.dispose?.();
        this.tokenizer = null;
        if (this.iframe) {
            this.iframe.remove();
            this.iframe = null;
        }
        this.emit("disposed");
    }
    async enqueue(fn) {
        const previous = this.queue;
        let release = () => { };
        this.queue = new Promise((resolve) => {
            release = resolve;
        });
        await previous;
        try {
            return await fn();
        }
        finally {
            release();
        }
    }
    runtimeBaseUrl() {
        return this.options.runtimeBaseUrl;
    }
    tokenizerJsonUrl() {
        return this.options.tokenizerUrl;
    }
    tokenizerScriptUrl() {
        return this.options.tokenizerScriptUrl;
    }
    buildDir() {
        return BUILD_DIRS[this.runtime];
    }
    iframeUrl() {
        const params = new URLSearchParams({
            build: this.buildDir(),
            model_url: this.options.modelUrl,
            pooling: "cls",
        });
        if (this.runtime === "webgpu") {
            params.set("backend", "webgpu");
        }
        if (this.runtime === "pthread") {
            params.set("threads", String(this.options.threads));
        }
        return `${joinUrl(this.runtimeBaseUrl(), "/scripts/wasm_encode_page.html")}?${params.toString()}`;
    }
    async ensureTokenizer() {
        if (this.tokenizer) {
            this.emit("tokenizer-ready", "reused");
            return this.tokenizer;
        }
        this.emit("tokenizer-script-loading", this.tokenizerScriptUrl());
        await loadScriptOnce(this.tokenizerScriptUrl());
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
        this.emit("tokenizer-json-ready");
        const json = await response.arrayBuffer();
        this.emit("tokenizer-fromjson-start");
        this.tokenizer = await tokenizers.Tokenizer.fromJSON(json);
        this.emit("tokenizer-fromjson-ready");
        this.emit("tokenizer-ready");
        return this.tokenizer;
    }
    async prefetchFiles() {
        const assets = [
            joinUrl(this.runtimeBaseUrl(), "/scripts/wasm_encode_page.html"),
            this.tokenizerScriptUrl(),
            this.tokenizerJsonUrl(),
            joinUrl(this.runtimeBaseUrl(), `/${this.buildDir()}/embedding_wasm_model_encode.js`),
            joinUrl(this.runtimeBaseUrl(), `/${this.buildDir()}/embedding_wasm_model_encode.wasm`),
            this.options.modelUrl,
        ];
        for (const asset of assets) {
            const kind = asset === this.options.modelUrl ? "model" : "asset";
            this.emit("prefetch-asset", asset);
            const response = await fetchMaybeCached(asset, this.options.cache, kind);
            if (!response.ok) {
                throw new Error(`failed to prefetch ${asset}: ${response.status}`);
            }
        }
    }
    encodeBatchLine(ids) {
        const fullIds = [0, ...Array.from(ids), 2];
        const mask = fullIds.map(() => 1);
        return `${fullIds.join(",")}\t${mask.join(",")}`;
    }
    async ensureIframe() {
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
            await new Promise((resolve) => {
                this.iframe.onload = () => resolve();
            });
            this.emit("runtime-page-ready");
        }
        return this.iframe;
    }
    async runEncode(batchLine) {
        const iframe = await this.ensureIframe();
        return new Promise((resolve, reject) => {
            const self = this;
            const timeout = window.setTimeout(() => {
                window.removeEventListener("message", onMessage);
                reject(new Error("encode request timed out"));
            }, 120000);
            function onMessage(event) {
                const data = event.data;
                if (!data) {
                    return;
                }
                if (data.type === "encode-status") {
                    self.emit(data.stage || "encode-status", data.detail);
                    return;
                }
                if (data.type !== "encode-result") {
                    return;
                }
                window.clearTimeout(timeout);
                window.removeEventListener("message", onMessage);
                if (data.error || !data.result) {
                    reject(new Error(data.error || "encode failed"));
                    return;
                }
                resolve(data.result);
            }
            window.addEventListener("message", onMessage);
            this.emit("encode-request-sent");
            iframe.contentWindow?.postMessage({ type: "encode-request", batchLine }, "*");
        });
    }
}
export async function createSnowflakeEmbedder(options) {
    ensureBrowser();
    const runtimeBaseUrl = options.runtimeBaseUrl ?? (browserOrigin() || DEFAULT_RUNTIME_BASE_PATH);
    const tokenizerUrl = options.tokenizerUrl ?? joinUrl(runtimeBaseUrl, DEFAULT_TOKENIZER_JSON_PATH);
    const tokenizerScriptUrl = options.tokenizerScriptUrl ?? joinUrl(runtimeBaseUrl, DEFAULT_TOKENIZER_SCRIPT_PATH);
    const detectedThreads = options.threads ?? navigator.hardwareConcurrency ?? 4;
    const resolved = {
        modelUrl: options.modelUrl || DEFAULT_SNOWFLAKE_MODEL_URL,
        runtime: options.runtime ?? "auto",
        runtimeBaseUrl,
        tokenizerUrl,
        tokenizerScriptUrl,
        threads: Math.max(1, Math.min(12, detectedThreads)),
        cache: options.cache ?? true,
        onStatus: options.onStatus ?? (() => { }),
    };
    const runtime = await detectRuntime(resolved.runtime);
    resolved.onStatus({ stage: "runtime-selected", runtime, detail: runtime });
    return new BrowserSnowflakeEmbedder(resolved, runtime);
}
export function createSnowflakeDefaults(overrides = {}) {
    return {
        modelUrl: DEFAULT_SNOWFLAKE_MODEL_URL,
        runtime: "auto",
        runtimeBaseUrl: browserOrigin() || DEFAULT_RUNTIME_BASE_PATH,
        cache: true,
        ...overrides,
    };
}
