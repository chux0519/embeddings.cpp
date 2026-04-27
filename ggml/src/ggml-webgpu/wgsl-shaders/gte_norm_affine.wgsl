struct Params {
    offset_src0: u32,
    offset_weight: u32,
    offset_bias: u32,
    offset_dst: u32,
    stride_src01: u32,
    stride_dst1: u32,
    hidden: u32,
    eps: f32,
};

@group(0) @binding(0)
var<storage, read_write> src0: array<f32>;

@group(0) @binding(1)
var<storage, read_write> weight: array<f32>;

@group(0) @binding(2)
var<storage, read_write> bias: array<f32>;

@group(0) @binding(3)
var<storage, read_write> dst: array<f32>;

@group(0) @binding(4)
var<uniform> params: Params;

var<workgroup> sums: array<f32, WG_SIZE>;
var<workgroup> sqs: array<f32, WG_SIZE>;

@compute @workgroup_size(WG_SIZE)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    let tid = lid.x;
    let src_base = params.offset_src0 + row * params.stride_src01;
    let dst_base = params.offset_dst + row * params.stride_dst1;

    var sum = 0.0;
    var sq = 0.0;
    var i = tid;
    loop {
        if (i >= params.hidden) {
            break;
        }
        let x = src0[src_base + i];
        sum += x;
        sq += x * x;
        i += WG_SIZE;
    }

    sums[tid] = sum;
    sqs[tid] = sq;
    workgroupBarrier();

    var stride = WG_SIZE / 2u;
    loop {
        if (stride == 0u) {
            break;
        }
        if (tid < stride) {
            sums[tid] += sums[tid + stride];
            sqs[tid] += sqs[tid + stride];
        }
        workgroupBarrier();
        stride = stride / 2u;
    }

    let mean = sums[0] / f32(params.hidden);
    let varv = sqs[0] / f32(params.hidden) - mean * mean;
    let inv_std = inverseSqrt(varv + params.eps);

    i = tid;
    loop {
        if (i >= params.hidden) {
            break;
        }
        let x = src0[src_base + i];
        let w = weight[params.offset_weight + i];
        let b = bias[params.offset_bias + i];
        dst[dst_base + i] = ((x - mean) * inv_std) * w + b;
        i += WG_SIZE;
    }
}
