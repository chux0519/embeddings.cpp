struct Params {
    offset_src0: u32,
    offset_positions: u32,
    offset_dst: u32,
    stride_src01: u32,
    stride_dst1: u32,
    ne: u32,
    hidden: u32,
};

@group(0) @binding(0)
var<storage, read_write> src0: array<f32>;

@group(0) @binding(1)
var<storage, read_write> positions: array<i32>;

@group(0) @binding(2)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(3)
var<uniform> params: Params;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (gid.x >= params.ne) {
        return;
    }

    let batch = gid.x / params.hidden;
    let i = gid.x % params.hidden;
    let dst_idx = params.offset_dst + batch * params.stride_dst1 + i;
    let pos = positions[params.offset_positions + batch];
    if (pos < 0) {
        dst[dst_idx] = 0.0;
        return;
    }

    dst[dst_idx] = src0[params.offset_src0 + u32(pos) * params.stride_src01 + i];
}
