struct Params {
    offset_qkv: u32,
    offset_cos: u32,
    offset_sin: u32,
    offset_dst: u32,
    stride_qkv1: u32,
    stride_cos1: u32,
    stride_sin1: u32,
    stride_dst1: u32,
    stride_dst2: u32,
    stride_dst3: u32,
    n_pairs: u32,
    hidden_size: u32,
    n_head: u32,
    d_head: u32,
    half: u32,
    n_token: u32,
    n_part: u32,
};

@group(0) @binding(0)
var<storage, read_write> qkv: array<f32>;

@group(0) @binding(1)
var<storage, read_write> rope_cos: array<f32>;

@group(0) @binding(2)
var<storage, read_write> rope_sin: array<f32>;

@group(0) @binding(3)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(4)
var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.n_pairs) {
        return;
    }

    var idx = gid.x;
    let i = idx % params.half;
    idx = idx / params.half;
    let head = idx % params.n_head;
    idx = idx / params.n_head;
    let token = idx % params.n_token;
    let part = idx / params.n_token;

    if (part >= params.n_part) {
        return;
    }

    let src_base = params.offset_qkv +
        token * params.stride_qkv1 +
        part * params.hidden_size +
        head * params.d_head;
    let dst_base = params.offset_dst +
        token * params.stride_dst1 +
        head * params.stride_dst2 +
        part * params.stride_dst3;

    let x0 = qkv[src_base + i];
    let x1 = qkv[src_base + params.half + i];

    if (part == 2u) {
        dst[dst_base + i] = x0;
        dst[dst_base + params.half + i] = x1;
        return;
    }

    let cv = rope_cos[params.offset_cos + token * params.stride_cos1 + i];
    let sv = rope_sin[params.offset_sin + token * params.stride_sin1 + i];
    dst[dst_base + i] = x0 * cv - x1 * sv;
    dst[dst_base + params.half + i] = x1 * cv + x0 * sv;
}
