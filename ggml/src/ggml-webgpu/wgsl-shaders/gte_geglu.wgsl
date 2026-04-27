struct Params {
    offset_src0: u32,
    offset_dst: u32,
    stride_src01: u32,
    stride_dst1: u32,
    ne: u32,
    hidden_features: u32,
};

@group(0) @binding(0)
var<storage, read_write> src0: array<f32>;

@group(0) @binding(1)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(2)
var<uniform> params: Params;

const SQRT_2_OVER_PI: f32 = 0.79788456080286535587989211986876;
const GELU_COEF_A: f32 = 0.044715;

fn gelu(x: f32) -> f32 {
    let v = SQRT_2_OVER_PI * x * (1.0 + GELU_COEF_A * x * x);
    return 0.5 * x * tanh(v) + 0.5 * x;
}

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.ne) {
        return;
    }

    let token = gid.x / params.hidden_features;
    let i = gid.x % params.hidden_features;
    let src_base = params.offset_src0 + token * params.stride_src01;
    let dst_idx = params.offset_dst + token * params.stride_dst1 + i;
    let up = src0[src_base + i];
    let gate = src0[src_base + params.hidden_features + i];
    dst[dst_idx] = up * gelu(gate);
}
