const CACHE_NAME = 'embeddings-browser-demo-v4';
const ALLOWED_PREFIXES = [
  '/build-wasm-web/',
  '/build-wasm-web-dyn/',
  '/build-wasm-web-pthread/',
  '/build-wasm-web-pthread-dyn/',
  '/build-wasm-webgpu-browser/',
  '/build-wasm-webgpu-browser-dyn/',
  '/scripts/wasm_bench_page.html',
  '/scripts/wasm_encode_page.html',
  '/demo/browser-wasm/',
];

function shouldHandle(url) {
  if (url.origin !== self.location.origin) {
    return false;
  }
  return ALLOWED_PREFIXES.some((prefix) => url.pathname.startsWith(prefix));
}

self.addEventListener('install', (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener('activate', (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== 'GET' || !shouldHandle(url)) {
    return;
  }

  event.respondWith((async () => {
    const cache = await caches.open(CACHE_NAME);
    const cached = await cache.match(event.request);
    if (cached) {
      return cached;
    }

    const response = await fetch(event.request);
    if (response.ok) {
      cache.put(event.request, response.clone());
    }
    return response;
  })());
});
