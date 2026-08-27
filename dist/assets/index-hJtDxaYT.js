const ka={add:`// Elementwise residual add: y[i] = a[i] + b[i].
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;
@group(0) @binding(2) var<storage, read> b: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (i >= p.n) { return; }
  y[i] = a[i] + b[i];
}
`,argmax:`// GPU argmax over the logits, writing one token id into a GPU buffer so the token never leaves the
// GPU (enables the deferred-sync decode loop). Single workgroup, WG threads strided-scan the N
// logits tracking (maxVal, maxIdx), then a shared-mem tree reduction. Tie-break = LOWEST index, to
// match the CPU argmax (strict > keeps the first max). No subgroup ops -> all devices.
override WG: u32 = 256u;
struct Params { N: u32, outIdx: u32, _0: u32, _1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> outTok: array<u32>;   // outTok[p.outIdx] = argmax

var<workgroup> sval: array<f32, 256>;
var<workgroup> sidx: array<u32, 256>;

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  var bv = -3.4e38;
  var bi = 0u;
  for (var i = tid; i < p.N; i = i + WG) {
    let v = logits[i];
    if (v > bv) { bv = v; bi = i; }      // strict > keeps the lowest index within this thread's stride
  }
  sval[tid] = bv; sidx[tid] = bi;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      let ov = sval[tid + s]; let oi = sidx[tid + s];
      if (ov > sval[tid] || (ov == sval[tid] && oi < sidx[tid])) { sval[tid] = ov; sidx[tid] = oi; }
    }
    workgroupBarrier();
  }
  if (tid == 0u) { outTok[p.outIdx] = sidx[0]; }
}
`,argmax_masked:`// Masked argmax: like argmax.wgsl but skips any id already chosen in a prior round, and writes BOTH
// the winning id and its logit value. Calling it K times (roundCount = 0..K-1, all in one compute
// pass so each round sees the prior rounds' writes) yields the exact top-K (id, logit) pairs in
// descending order = ONNX TopK over the (penalty-filtered) logits, which is what the transformers.js
// sampler consumes. Then only K pairs are read back (not the full vocab), and the CPU does
// temperature + softmax + multinomial. Single workgroup, no subgroup ops -> all devices. Tie-break =
// lowest index (strict >), matching argmax.wgsl / ORT TopK in practice.
override WG: u32 = 256u;
struct Params { N: u32, roundCount: u32, _0: u32, _1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> candIds: array<u32>;   // [K]; reads 0..roundCount-1, writes [roundCount]
@group(0) @binding(3) var<storage, read_write> candVals: array<f32>;  // [K]; writes [roundCount]

var<workgroup> sval: array<f32, 256>;
var<workgroup> sidx: array<u32, 256>;

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  var bv = -3.4e38;
  var bi = 0u;
  for (var i = tid; i < p.N; i = i + WG) {
    let v = logits[i];
    if (v > bv) {
      var skip = false;
      for (var r = 0u; r < p.roundCount; r = r + 1u) { if (candIds[r] == i) { skip = true; break; } }
      if (!skip) { bv = v; bi = i; }     // strict > keeps the lowest index within this thread's stride
    }
  }
  sval[tid] = bv; sidx[tid] = bi;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      let ov = sval[tid + s]; let oi = sidx[tid + s];
      if (ov > sval[tid] || (ov == sval[tid] && oi < sidx[tid])) { sval[tid] = ov; sidx[tid] = oi; }
    }
    workgroupBarrier();
  }
  if (tid == 0u) { candIds[p.roundCount] = sidx[0]; candVals[p.roundCount] = sval[0]; }
}
`,attention_online:`// Causal GQA attention with online (flash) softmax, head-dim up to the workgroup size (256) - the
// Qwen3.5 full-attention layers use head_dim 256, past the <=128 the register-array kernels assume.
// One workgroup per query (s,h); thread d owns output dim d. Streams keys j<=s keeping running
// max/sum/acc, so no O(S) score storage. Output gate + RoPE + QK-norm are applied separately.
override WGD: u32 = 256u;                  // threads == head_dim D
struct Params { S: u32, H: u32, KV: u32, D: u32, scale: f32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;    // [S, H, D]
@group(0) @binding(2) var<storage, read> k: array<f32>;    // [S, KV, D]
@group(0) @binding(3) var<storage, read> v: array<f32>;    // [S, KV, D]
@group(0) @binding(4) var<storage, read_write> outp: array<f32>; // [S, H, D]
var<workgroup> qsh: array<f32, 256>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(WGD)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let qi = wg.x;                 // query flat index = s*H + h
  let s = qi / p.H;
  let h = qi % p.H;
  let hkv = h / (p.H / p.KV);    // GQA: which kv head
  let d = lid.x;
  let D = p.D;
  if (d < D) { qsh[d] = q[qi * D + d]; }
  workgroupBarrier();

  var m = -1e30;
  var l = 0.0;
  var acc = 0.0;
  for (var j = 0u; j <= s; j = j + 1u) {
    red[d] = select(0.0, qsh[d] * k[(j * p.KV + hkv) * D + d], d < D);
    workgroupBarrier();
    for (var st = WGD / 2u; st > 0u; st = st >> 1u) {
      if (d < st) { red[d] = red[d] + red[d + st]; }
      workgroupBarrier();
    }
    let score = red[0] * p.scale;
    let mn = max(m, score);
    let corr = exp(m - mn);
    let pj = exp(score - mn);
    l = l * corr + pj;
    if (d < D) { acc = acc * corr + pj * v[(j * p.KV + hkv) * D + d]; }
    m = mn;
    workgroupBarrier();               // before next j overwrites red
  }
  if (d < D) { outp[qi * D + d] = acc / l; }
}
`,attention_online_cache:`// Causal GQA attention (online/flash softmax, head_dim up to 256) reading K/V from the persistent
// f32 cache (Kc/Vc, layout [pos*KV + kv_head, D]) - the Qwen3.5 full-attention path for both prefill
// and decode. One workgroup per query (s,h); thread d owns output dim d. The query at absolute
// position posBase+s attends to cache positions 0 .. posBase+s (causal). Keys are cached already
// RoPE'd, so no read-time rotation.
override WGD: u32 = 256u;                  // threads == head_dim D
struct Params { S: u32, H: u32, KV: u32, D: u32, scale: f32, posBase: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;    // [S, H, D]
@group(0) @binding(2) var<storage, read> kc: array<f32>;   // cache [cap*KV, D]
@group(0) @binding(3) var<storage, read> vc: array<f32>;   // cache [cap*KV, D]
@group(0) @binding(4) var<storage, read_write> outp: array<f32>; // [S, H, D]
var<workgroup> qsh: array<f32, 256>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(WGD)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let qi = wg.x;                 // query flat index = s*H + h
  let s = qi / p.H;
  let h = qi % p.H;
  let hkv = h / (p.H / p.KV);
  let d = lid.x;
  let D = p.D;
  if (d < D) { qsh[d] = q[qi * D + d]; }
  workgroupBarrier();

  var m = -1e30;
  var l = 0.0;
  var acc = 0.0;
  let last = p.posBase + s;      // inclusive: attend cache positions 0..last
  for (var j = 0u; j <= last; j = j + 1u) {
    red[d] = select(0.0, qsh[d] * kc[(j * p.KV + hkv) * D + d], d < D);
    workgroupBarrier();
    for (var st = WGD / 2u; st > 0u; st = st >> 1u) {
      if (d < st) { red[d] = red[d] + red[d + st]; }
      workgroupBarrier();
    }
    let score = red[0] * p.scale;
    let mn = max(m, score);
    let corr = exp(m - mn);
    let pj = exp(score - mn);
    l = l * corr + pj;
    if (d < D) { acc = acc * corr + pj * vc[(j * p.KV + hkv) * D + d]; }
    m = mn;
    workgroupBarrier();
  }
  if (d < D) { outp[qi * D + d] = acc / l; }
}
`,attention_online_cache_kv8:`// q8 variant of attention_online_cache: the Qwen3.5 full-attention path reading K/V from the packed
// snorm8 cache (kcQ/vcQ = 4 x snorm8 per u32 word, kcS/vcS = one f32 scale per 32-element block,
// llama.cpp q8_0-style, written by copy_kv8). Each element is dequantized with one unpack4x8snorm +
// block-scale multiply at read time; all online-softmax arithmetic stays f32, so this matches the f32
// attention_online_cache exactly except for the single snorm8 rounding of K/V at write (nothing
// compounds). Same structure: one workgroup per query (s,h), thread d owns output dim d. head_dim <=256.
override WGD: u32 = 256u;                  // threads == head_dim D
struct Params { S: u32, H: u32, KV: u32, D: u32, scale: f32, posBase: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;    // [S, H, D]
@group(0) @binding(2) var<storage, read> kcQ: array<u32>;  // cache [cap*KV, D/4] packed snorm8
@group(0) @binding(3) var<storage, read> kcS: array<f32>;  // cache [cap*KV, D/32] block scales
@group(0) @binding(4) var<storage, read> vcQ: array<u32>;  // cache [cap*KV, D/4]
@group(0) @binding(5) var<storage, read> vcS: array<f32>;  // cache [cap*KV, D/32]
@group(0) @binding(6) var<storage, read_write> outp: array<f32>; // [S, H, D]
var<workgroup> qsh: array<f32, 256>;
var<workgroup> red: array<f32, 256>;

@compute @workgroup_size(WGD)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let qi = wg.x;                 // query flat index = s*H + h
  let s = qi / p.H;
  let h = qi % p.H;
  let hkv = h / (p.H / p.KV);
  let d = lid.x;
  let D = p.D;
  let W4 = D / 4u;
  let NB = D / 32u;
  if (d < D) { qsh[d] = q[qi * D + d]; }
  workgroupBarrier();

  var m = -1e30;
  var l = 0.0;
  var acc = 0.0;
  let last = p.posBase + s;      // inclusive: attend cache positions 0..last
  for (var j = 0u; j <= last; j = j + 1u) {
    let row = j * p.KV + hkv;
    var kval = 0.0;
    if (d < D) { kval = unpack4x8snorm(kcQ[row * W4 + (d >> 2u)])[d & 3u] * kcS[row * NB + (d >> 5u)]; }
    red[d] = select(0.0, qsh[d] * kval, d < D);
    workgroupBarrier();
    for (var st = WGD / 2u; st > 0u; st = st >> 1u) {
      if (d < st) { red[d] = red[d] + red[d + st]; }
      workgroupBarrier();
    }
    let score = red[0] * p.scale;
    let mn = max(m, score);
    let corr = exp(m - mn);
    let pj = exp(score - mn);
    l = l * corr + pj;
    if (d < D) {
      let vval = unpack4x8snorm(vcQ[row * W4 + (d >> 2u)])[d & 3u] * vcS[row * NB + (d >> 5u)];
      acc = acc * corr + pj * vval;
    }
    m = mn;
    workgroupBarrier();
  }
  if (d < D) { outp[qi * D + d] = acc / l; }
}
`,attention_sg:`// Causal GQA attention, subgroup-parallel: one subgroup (= one workgroup) per (query, head).
// Lanes split head_dim; flash-style online softmax over the cached positions; the per-position
// score (q.k) is reduced with subgroupAdd. Fixes the decode bottleneck where attention ran only
// H threads. SG = device subgroup size (16/32/64, so head_dim/SG <= 8). Reads K/V from the cache.
enable subgroups;
override SG: u32 = 32u;
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [S, H, D]
@group(0) @binding(2) var<storage, read> Kc: array<f32>;       // [Ltot, KV, D]
@group(0) @binding(3) var<storage, read> Vc: array<f32>;       // [Ltot, KV, D]
@group(0) @binding(4) var<storage, read_write> out: array<f32>; // [S, H, D]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let idx = wg.x;
  if (idx >= p.S * p.H) { return; }
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));
  let dper = p.D / SG;                         // <= 8 for SG>=16, D=128

  var acc: array<f32, 8>;
  for (var t = 0u; t < dper; t = t + 1u) { acc[t] = 0.0; }
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let kb = (j * p.KV + kvh) * p.D;
    var part = 0.0;
    for (var t = 0u; t < dper; t = t + 1u) { let d = lane + t * SG; part = part + q[qb + d] * Kc[kb + d]; }
    let score = subgroupAdd(part) * inv;       // full q.k dot, broadcast to all lanes
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let w = exp(score - mnew);
    l = l * corr + w;
    for (var t = 0u; t < dper; t = t + 1u) { let d = lane + t * SG; acc[t] = acc[t] * corr + w * Vc[kb + d]; }
    m = mnew;
  }
  let ob = (qi * p.H + h) * p.D;
  for (var t = 0u; t < dper; t = t + 1u) { let d = lane + t * SG; out[ob + d] = acc[t] / l; }
}
`,attention_sg_kv16:`// attention_sg with an f16-STORAGE KV cache (kvCache: 'f16'). Keep in lockstep with
// attention_sg.wgsl: the ONLY difference is Kc/Vc are array<f16> and each cached value is
// widened to f32 at the read. All arithmetic (dot, softmax, accumulation) stays f32, so the
// precision loss is exactly one rounding of K/V at cache-write time, nothing compounding.
enable subgroups;
enable f16;
override SG: u32 = 32u;
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [S, H, D]
@group(0) @binding(2) var<storage, read> Kc: array<f16>;       // [Ltot, KV, D]
@group(0) @binding(3) var<storage, read> Vc: array<f16>;       // [Ltot, KV, D]
@group(0) @binding(4) var<storage, read_write> out: array<f32>; // [S, H, D]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let idx = wg.x;
  if (idx >= p.S * p.H) { return; }
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));
  let dper = p.D / SG;

  var acc: array<f32, 8>;
  for (var t = 0u; t < dper; t = t + 1u) { acc[t] = 0.0; }
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let kb = (j * p.KV + kvh) * p.D;
    var part = 0.0;
    for (var t = 0u; t < dper; t = t + 1u) { let d = lane + t * SG; part = part + q[qb + d] * f32(Kc[kb + d]); }
    let score = subgroupAdd(part) * inv;
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let w = exp(score - mnew);
    l = l * corr + w;
    for (var t = 0u; t < dper; t = t + 1u) { let d = lane + t * SG; acc[t] = acc[t] * corr + w * f32(Vc[kb + d]); }
    m = mnew;
  }
  let ob = (qi * p.H + h) * p.D;
  for (var t = 0u; t < dper; t = t + 1u) { let d = lane + t * SG; out[ob + d] = acc[t] / l; }
}
`,attention_sg_kv16_roll:`// attention_sg_kv16 for the rolling-window / attention-sinks mode (see attention_sg_roll.wgsl
// for the rope-at-read scheme). Keep in lockstep with attention_sg_kv16.wgsl: the ONLY
// difference is the K rotation in the score loop; each cached f16 value is widened to f32 at
// the read and rotated with the same \`k*cos + rot*sin\` operand order as rmsnorm_rope_sg.
// The engine only selects this kernel when SG <= D/2 (partner dim stays in-lane).
enable subgroups;
enable f16;
override SG: u32 = 32u;
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [S, H, D] (roped, cache-relative)
@group(0) @binding(2) var<storage, read> Kc: array<f16>;       // [Ltot, KV, D] UNROPED
@group(0) @binding(3) var<storage, read> Vc: array<f16>;       // [Ltot, KV, D]
@group(0) @binding(4) var<storage, read> cosT: array<f32>;     // [positions, D/2]
@group(0) @binding(5) var<storage, read> sinT: array<f32>;     // [positions, D/2]
@group(0) @binding(6) var<storage, read_write> out: array<f32>; // [S, H, D]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let idx = wg.x;
  if (idx >= p.S * p.H) { return; }
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));
  let dper = p.D / SG;
  let half = p.D / 2u;
  let hs = half / SG;                          // strides from a dim to its rotate partner

  var acc: array<f32, 8>;
  for (var t = 0u; t < dper; t = t + 1u) { acc[t] = 0.0; }
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let kb = (j * p.KV + kvh) * p.D;
    var kd: array<f32, 8>;
    for (var t = 0u; t < dper; t = t + 1u) { kd[t] = f32(Kc[kb + lane + t * SG]); }
    var part = 0.0;
    for (var t = 0u; t < dper; t = t + 1u) {
      let d = lane + t * SG;
      var rot: f32;
      if (d < half) { rot = -kd[t + hs]; } else { rot = kd[t - hs]; }
      let rb = j * half + (d % half);
      part = part + q[qb + d] * (kd[t] * cosT[rb] + rot * sinT[rb]);
    }
    let score = subgroupAdd(part) * inv;
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let w = exp(score - mnew);
    l = l * corr + w;
    for (var t = 0u; t < dper; t = t + 1u) { let d = lane + t * SG; acc[t] = acc[t] * corr + w * f32(Vc[kb + d]); }
    m = mnew;
  }
  let ob = (qi * p.H + h) * p.D;
  for (var t = 0u; t < dper; t = t + 1u) { let d = lane + t * SG; out[ob + d] = acc[t] / l; }
}
`,attention_sg_kv8:`// attention_sg with a q8 KV cache (kvCache: 'q8'). Keep in lockstep with attention_sg.wgsl: the
// ONLY difference is Kc/Vc are packed snorm8 words dequantized at the read with their per-block
// f32 scales (32-element blocks, q8_0-style; see copy_kv8.wgsl). All arithmetic (dot, softmax,
// accumulation) stays f32. Each lane owns whole packed words, so q is read in matching groups
// of 4.
enable subgroups;
override SG: u32 = 32u;
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;         // [S, H, D]
@group(0) @binding(2) var<storage, read> Kq: array<u32>;        // [Ltot, KV, D/4] packed snorm8
@group(0) @binding(3) var<storage, read> Vq: array<u32>;        // [Ltot, KV, D/4] packed snorm8
@group(0) @binding(4) var<storage, read> Ks: array<f32>;        // [Ltot, KV, D/32] block scales
@group(0) @binding(5) var<storage, read> Vs: array<f32>;        // [Ltot, KV, D/32] block scales
@group(0) @binding(6) var<storage, read_write> out: array<f32>; // [S, H, D]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let idx = wg.x;
  if (idx >= p.S * p.H) { return; }
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));
  let W4 = p.D / 4u;
  let B32 = p.D / 32u;

  var acc: array<vec4<f32>, 8>;      // words per lane: W4/SG <= 8 for SG >= 4
  for (var t = 0u; t < 8u; t = t + 1u) { acc[t] = vec4<f32>(0.0); }
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let rowQ = (j * p.KV + kvh) * W4;
    let rowS = (j * p.KV + kvh) * B32;
    var part = 0.0;
    for (var w = lane; w < W4; w = w + SG) {
      let kw = unpack4x8snorm(Kq[rowQ + w]) * Ks[rowS + (w >> 3u)];
      let qv = vec4<f32>(q[qb + w * 4u], q[qb + w * 4u + 1u], q[qb + w * 4u + 2u], q[qb + w * 4u + 3u]);
      part = part + dot(qv, kw);
    }
    let score = subgroupAdd(part) * inv;
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let wgt = exp(score - mnew);
    l = l * corr + wgt;
    var wi = 0u;
    for (var w = lane; w < W4; w = w + SG) {
      let vw = unpack4x8snorm(Vq[rowQ + w]) * Vs[rowS + (w >> 3u)];
      acc[wi] = acc[wi] * corr + wgt * vw;
      wi = wi + 1u;
    }
    m = mnew;
  }
  let ob = (qi * p.H + h) * p.D;
  var wi = 0u;
  for (var w = lane; w < W4; w = w + SG) {
    let o = acc[wi] / l;
    out[ob + w * 4u] = o.x;
    out[ob + w * 4u + 1u] = o.y;
    out[ob + w * 4u + 2u] = o.z;
    out[ob + w * 4u + 3u] = o.w;
    wi = wi + 1u;
  }
}
`,attention_sg_kv8_roll:`// attention_sg_kv8 for the rolling-window / attention-sinks mode (see attention_sg_roll.wgsl
// for the rope-at-read scheme). Keep in lockstep with attention_sg_kv8.wgsl: the ONLY
// difference is the K rotation in the score loop. The cache holds UNROPED quantized keys -
// the whole point: the packed bytes are immutable, so eviction never requantizes (llama.cpp's
// K-shift cannot do this on a quantized cache at all). A word's rotate partner is the word
// D/8 away (all 4 dims of a word share one half), dequantized from global with its own scale.
enable subgroups;
override SG: u32 = 32u;
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [S, H, D] (roped, cache-relative)
@group(0) @binding(2) var<storage, read> Kq: array<u32>;       // [Ltot, KV, D/4] packed snorm8, UNROPED
@group(0) @binding(3) var<storage, read> Vq: array<u32>;       // [Ltot, KV, D/4] packed snorm8
@group(0) @binding(4) var<storage, read> Ks: array<f32>;       // [Ltot, KV, D/32] block scales
@group(0) @binding(5) var<storage, read> Vs: array<f32>;       // [Ltot, KV, D/32] block scales
@group(0) @binding(6) var<storage, read> cosT: array<f32>;     // [positions, D/2]
@group(0) @binding(7) var<storage, read> sinT: array<f32>;     // [positions, D/2]
@group(0) @binding(8) var<storage, read_write> out: array<f32>; // [S, H, D]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let idx = wg.x;
  if (idx >= p.S * p.H) { return; }
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));
  let W4 = p.D / 4u;
  let B32 = p.D / 32u;
  let half = p.D / 2u;
  let hw = half / 4u;                          // words from a word to its rotate partner

  var acc: array<vec4<f32>, 8>;      // words per lane: W4/SG <= 8 for SG >= 4
  for (var t = 0u; t < 8u; t = t + 1u) { acc[t] = vec4<f32>(0.0); }
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let rowQ = (j * p.KV + kvh) * W4;
    let rowS = (j * p.KV + kvh) * B32;
    var part = 0.0;
    for (var w = lane; w < W4; w = w + SG) {
      let kw = unpack4x8snorm(Kq[rowQ + w]) * Ks[rowS + (w >> 3u)];
      let wp = select(w - hw, w + hw, w < hw);
      let kp = unpack4x8snorm(Kq[rowQ + wp]) * Ks[rowS + (wp >> 3u)];
      let rot = select(kp, -kp, w < hw);
      let cb = j * half + select(w - hw, w, w < hw) * 4u;
      let cs = vec4<f32>(cosT[cb], cosT[cb + 1u], cosT[cb + 2u], cosT[cb + 3u]);
      let sn = vec4<f32>(sinT[cb], sinT[cb + 1u], sinT[cb + 2u], sinT[cb + 3u]);
      let qv = vec4<f32>(q[qb + w * 4u], q[qb + w * 4u + 1u], q[qb + w * 4u + 2u], q[qb + w * 4u + 3u]);
      part = part + dot(qv, kw * cs + rot * sn);
    }
    let score = subgroupAdd(part) * inv;
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let wgt = exp(score - mnew);
    l = l * corr + wgt;
    var wi = 0u;
    for (var w = lane; w < W4; w = w + SG) {
      let vw = unpack4x8snorm(Vq[rowQ + w]) * Vs[rowS + (w >> 3u)];
      acc[wi] = acc[wi] * corr + wgt * vw;
      wi = wi + 1u;
    }
    m = mnew;
  }
  let ob = (qi * p.H + h) * p.D;
  var wi = 0u;
  for (var w = lane; w < W4; w = w + SG) {
    let o = acc[wi] / l;
    out[ob + w * 4u] = o.x;
    out[ob + w * 4u + 1u] = o.y;
    out[ob + w * 4u + 2u] = o.z;
    out[ob + w * 4u + 3u] = o.w;
    wi = wi + 1u;
  }
}
`,attention_sg_roll:`// attention_sg for the rolling-window / attention-sinks mode (overflow: 'sinks'): the cache
// holds UNROPED keys, and each cached row j is rotated AT READ by its cache-relative position
// (StreamingLLM-style; the cache bytes are immutable, so eviction compaction never re-rotates
// or requantizes anything). Keep in lockstep with attention_sg.wgsl: the ONLY difference is
// the K rotation in the score loop, written as \`k*cos + rot*sin\` with the same operand order
// as rmsnorm_rope_sg so the f32 path stays bit-identical to the roped-at-write kernels until
// the first eviction. cosT/sinT are the aux rope tables, [positions, D/2].
// Lane math: d = lane + t*SG, partner d±D/2 is (D/2)/SG strides away IN THE SAME LANE (the
// engine only selects this kernel when SG <= D/2).
enable subgroups;
override SG: u32 = 32u;
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [S, H, D] (roped, cache-relative)
@group(0) @binding(2) var<storage, read> Kc: array<f32>;       // [Ltot, KV, D] UNROPED
@group(0) @binding(3) var<storage, read> Vc: array<f32>;       // [Ltot, KV, D]
@group(0) @binding(4) var<storage, read> cosT: array<f32>;     // [positions, D/2]
@group(0) @binding(5) var<storage, read> sinT: array<f32>;     // [positions, D/2]
@group(0) @binding(6) var<storage, read_write> out: array<f32>; // [S, H, D]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let idx = wg.x;
  if (idx >= p.S * p.H) { return; }
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));
  let dper = p.D / SG;
  let half = p.D / 2u;
  let hs = half / SG;                          // strides from a dim to its rotate partner

  var acc: array<f32, 8>;
  for (var t = 0u; t < dper; t = t + 1u) { acc[t] = 0.0; }
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let kb = (j * p.KV + kvh) * p.D;
    var kd: array<f32, 8>;
    for (var t = 0u; t < dper; t = t + 1u) { kd[t] = Kc[kb + lane + t * SG]; }
    var part = 0.0;
    for (var t = 0u; t < dper; t = t + 1u) {
      let d = lane + t * SG;
      var rot: f32;
      if (d < half) { rot = -kd[t + hs]; } else { rot = kd[t - hs]; }
      let rb = j * half + (d % half);
      part = part + q[qb + d] * (kd[t] * cosT[rb] + rot * sinT[rb]);
    }
    let score = subgroupAdd(part) * inv;
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let w = exp(score - mnew);
    l = l * corr + w;
    for (var t = 0u; t < dper; t = t + 1u) { let d = lane + t * SG; acc[t] = acc[t] * corr + w * Vc[kb + d]; }
    m = mnew;
  }
  let ob = (qi * p.H + h) * p.D;
  for (var t = 0u; t < dper; t = t + 1u) { let d = lane + t * SG; out[ob + d] = acc[t] / l; }
}
`,attention_wg:`// Causal GQA attention, no-subgroup fallback: one workgroup per (query, head); threads split
// head_dim; flash-style online softmax over the cached positions; the per-position q.k score
// is tree-reduced via shared memory. Replaces attention_cache on this path: its single thread
// per (query, head) walked the WHOLE context serially, so fallback decode degraded linearly
// with conversation length and prefill attention was latency-bound. Mirrors attention_sg with
// subgroupAdd swapped for the shared-memory reduction. Fixed workgroup of 64: the per-thread
// accumulator covers head_dim <= 128 (enforced at manifest validation) in 2 strides.
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [S, H, D]
@group(0) @binding(2) var<storage, read> Kc: array<f32>;       // [Ltot, KV, D]
@group(0) @binding(3) var<storage, read> Vc: array<f32>;       // [Ltot, KV, D]
@group(0) @binding(4) var<storage, read_write> out: array<f32>; // [S, H, D]
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let idx = wg.x;                        // uniform across the workgroup -> early return is barrier-safe
  if (idx >= p.S * p.H) { return; }
  let tid = lid.x;
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));

  var acc: array<f32, 2>;
  acc[0] = 0.0;
  acc[1] = 0.0;
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let kb = (j * p.KV + kvh) * p.D;
    var part = 0.0;
    for (var t = 0u; t < 2u; t = t + 1u) {
      let d = tid + t * 64u;
      if (d < p.D) { part = part + q[qb + d] * Kc[kb + d]; }
    }
    red[tid] = part;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
      if (tid < s) { red[tid] = red[tid] + red[tid + s]; }
      workgroupBarrier();
    }
    let score = red[0] * inv;            // full q.k dot, visible to all threads
    workgroupBarrier();                  // red[0] consumed before the next position overwrites it
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let w = exp(score - mnew);
    l = l * corr + w;
    for (var t = 0u; t < 2u; t = t + 1u) {
      let d = tid + t * 64u;
      if (d < p.D) { acc[t] = acc[t] * corr + w * Vc[kb + d]; }
    }
    m = mnew;
  }
  let ob = (qi * p.H + h) * p.D;
  for (var t = 0u; t < 2u; t = t + 1u) {
    let d = tid + t * 64u;
    if (d < p.D) { out[ob + d] = acc[t] / l; }
  }
}
`,attention_wg_kv16:`// attention_wg with an f16-STORAGE KV cache (kvCache: 'f16'). Keep in lockstep with
// attention_wg.wgsl: the ONLY difference is Kc/Vc are array<f16>, widened to f32 at the read.
enable f16;
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [S, H, D]
@group(0) @binding(2) var<storage, read> Kc: array<f16>;       // [Ltot, KV, D]
@group(0) @binding(3) var<storage, read> Vc: array<f16>;       // [Ltot, KV, D]
@group(0) @binding(4) var<storage, read_write> out: array<f32>; // [S, H, D]
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let idx = wg.x;                        // uniform across the workgroup -> early return is barrier-safe
  if (idx >= p.S * p.H) { return; }
  let tid = lid.x;
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));

  var acc: array<f32, 2>;
  acc[0] = 0.0;
  acc[1] = 0.0;
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let kb = (j * p.KV + kvh) * p.D;
    var part = 0.0;
    for (var t = 0u; t < 2u; t = t + 1u) {
      let d = tid + t * 64u;
      if (d < p.D) { part = part + q[qb + d] * f32(Kc[kb + d]); }
    }
    red[tid] = part;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
      if (tid < s) { red[tid] = red[tid] + red[tid + s]; }
      workgroupBarrier();
    }
    let score = red[0] * inv;
    workgroupBarrier();
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let w = exp(score - mnew);
    l = l * corr + w;
    for (var t = 0u; t < 2u; t = t + 1u) {
      let d = tid + t * 64u;
      if (d < p.D) { acc[t] = acc[t] * corr + w * f32(Vc[kb + d]); }
    }
    m = mnew;
  }
  let ob = (qi * p.H + h) * p.D;
  for (var t = 0u; t < 2u; t = t + 1u) {
    let d = tid + t * 64u;
    if (d < p.D) { out[ob + d] = acc[t] / l; }
  }
}
`,attention_wg_kv16_roll:`// attention_wg_kv16 for the rolling-window / attention-sinks mode (see attention_sg_roll.wgsl
// for the rope-at-read scheme). Keep in lockstep with attention_wg_kv16.wgsl: the ONLY
// differences are the shared-memory K stage (kk, widened to f32) - the rotate partner d±D/2
// may live in another thread's stride - and the rotation in the score loop, written as
// \`k*cos + rot*sin\` with the same operand order as rmsnorm_rope_sg.
enable f16;
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [S, H, D] (roped, cache-relative)
@group(0) @binding(2) var<storage, read> Kc: array<f16>;       // [Ltot, KV, D] UNROPED
@group(0) @binding(3) var<storage, read> Vc: array<f16>;       // [Ltot, KV, D]
@group(0) @binding(4) var<storage, read> cosT: array<f32>;     // [positions, D/2]
@group(0) @binding(5) var<storage, read> sinT: array<f32>;     // [positions, D/2]
@group(0) @binding(6) var<storage, read_write> out: array<f32>; // [S, H, D]
var<workgroup> red: array<f32, 64>;
var<workgroup> kk: array<f32, 128>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let idx = wg.x;                        // uniform across the workgroup -> early return is barrier-safe
  if (idx >= p.S * p.H) { return; }
  let tid = lid.x;
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));
  let half = p.D / 2u;

  var acc: array<f32, 2>;
  acc[0] = 0.0;
  acc[1] = 0.0;
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let kb = (j * p.KV + kvh) * p.D;
    for (var t = 0u; t < 2u; t = t + 1u) {
      let d = tid + t * 64u;
      if (d < p.D) { kk[d] = f32(Kc[kb + d]); }
    }
    workgroupBarrier();
    var part = 0.0;
    for (var t = 0u; t < 2u; t = t + 1u) {
      let d = tid + t * 64u;
      if (d < p.D) {
        var rot: f32;
        if (d < half) { rot = -kk[d + half]; } else { rot = kk[d - half]; }
        let rb = j * half + (d % half);
        part = part + q[qb + d] * (kk[d] * cosT[rb] + rot * sinT[rb]);
      }
    }
    red[tid] = part;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
      if (tid < s) { red[tid] = red[tid] + red[tid + s]; }
      workgroupBarrier();
    }
    let score = red[0] * inv;
    workgroupBarrier();                  // red[0] + kk consumed before the next position overwrites them
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let w = exp(score - mnew);
    l = l * corr + w;
    for (var t = 0u; t < 2u; t = t + 1u) {
      let d = tid + t * 64u;
      if (d < p.D) { acc[t] = acc[t] * corr + w * f32(Vc[kb + d]); }
    }
    m = mnew;
  }
  let ob = (qi * p.H + h) * p.D;
  for (var t = 0u; t < 2u; t = t + 1u) {
    let d = tid + t * 64u;
    if (d < p.D) { out[ob + d] = acc[t] / l; }
  }
}
`,attention_wg_kv8:`// attention_wg with a q8 KV cache (kvCache: 'q8'): the no-subgroup fallback reader for the
// packed-snorm8 cache (see copy_kv8.wgsl for the write side). Keep in lockstep with
// attention_wg.wgsl: same online softmax, all arithmetic f32; each thread owns one packed word
// (D <= 128 -> at most 32 words, so threads 32..63 only carry zeros through the reduction).
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;         // [S, H, D]
@group(0) @binding(2) var<storage, read> Kq: array<u32>;        // [Ltot, KV, D/4] packed snorm8
@group(0) @binding(3) var<storage, read> Vq: array<u32>;        // [Ltot, KV, D/4] packed snorm8
@group(0) @binding(4) var<storage, read> Ks: array<f32>;        // [Ltot, KV, D/32] block scales
@group(0) @binding(5) var<storage, read> Vs: array<f32>;        // [Ltot, KV, D/32] block scales
@group(0) @binding(6) var<storage, read_write> out: array<f32>; // [S, H, D]
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let idx = wg.x;                    // uniform across the workgroup -> early return is barrier-safe
  if (idx >= p.S * p.H) { return; }
  let t = lid.x;
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));
  let W4 = p.D / 4u;

  var qv = vec4<f32>(0.0);
  if (t < W4) {
    qv = vec4<f32>(q[qb + t * 4u], q[qb + t * 4u + 1u], q[qb + t * 4u + 2u], q[qb + t * 4u + 3u]);
  }
  var acc = vec4<f32>(0.0);
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let rowQ = (j * p.KV + kvh) * W4;
    let rowS = (j * p.KV + kvh) * (p.D / 32u);
    var part = 0.0;
    if (t < W4) {
      let kw = unpack4x8snorm(Kq[rowQ + t]) * Ks[rowS + (t >> 3u)];
      part = dot(qv, kw);
    }
    red[t] = part;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
      if (t < s) { red[t] = red[t] + red[t + s]; }
      workgroupBarrier();
    }
    let score = red[0] * inv;
    workgroupBarrier();
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let wgt = exp(score - mnew);
    l = l * corr + wgt;
    if (t < W4) {
      let vw = unpack4x8snorm(Vq[rowQ + t]) * Vs[rowS + (t >> 3u)];
      acc = acc * corr + wgt * vw;
    }
    m = mnew;
  }
  if (t < W4) {
    let ob = (qi * p.H + h) * p.D;
    let o = acc / l;
    out[ob + t * 4u] = o.x;
    out[ob + t * 4u + 1u] = o.y;
    out[ob + t * 4u + 2u] = o.z;
    out[ob + t * 4u + 3u] = o.w;
  }
}
`,attention_wg_kv8_roll:`// attention_wg_kv8 for the rolling-window / attention-sinks mode: the no-subgroup fallback of
// attention_sg_kv8_roll (see there and attention_sg_roll.wgsl for the rope-at-read scheme).
// Keep in lockstep with attention_wg_kv8.wgsl: the ONLY difference is the K rotation in the
// score loop; each thread's word rotates against its partner word D/8 away, dequantized from
// global with its own scale.
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [S, H, D] (roped, cache-relative)
@group(0) @binding(2) var<storage, read> Kq: array<u32>;       // [Ltot, KV, D/4] packed snorm8, UNROPED
@group(0) @binding(3) var<storage, read> Vq: array<u32>;       // [Ltot, KV, D/4] packed snorm8
@group(0) @binding(4) var<storage, read> Ks: array<f32>;       // [Ltot, KV, D/32] block scales
@group(0) @binding(5) var<storage, read> Vs: array<f32>;       // [Ltot, KV, D/32] block scales
@group(0) @binding(6) var<storage, read> cosT: array<f32>;     // [positions, D/2]
@group(0) @binding(7) var<storage, read> sinT: array<f32>;     // [positions, D/2]
@group(0) @binding(8) var<storage, read_write> out: array<f32>; // [S, H, D]
var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let idx = wg.x;                    // uniform across the workgroup -> early return is barrier-safe
  if (idx >= p.S * p.H) { return; }
  let t = lid.x;
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));
  let W4 = p.D / 4u;
  let half = p.D / 2u;
  let hw = half / 4u;                          // words from a word to its rotate partner

  var qv = vec4<f32>(0.0);
  if (t < W4) {
    qv = vec4<f32>(q[qb + t * 4u], q[qb + t * 4u + 1u], q[qb + t * 4u + 2u], q[qb + t * 4u + 3u]);
  }
  var acc = vec4<f32>(0.0);
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let rowQ = (j * p.KV + kvh) * W4;
    let rowS = (j * p.KV + kvh) * (p.D / 32u);
    var part = 0.0;
    if (t < W4) {
      let kw = unpack4x8snorm(Kq[rowQ + t]) * Ks[rowS + (t >> 3u)];
      let wp = select(t - hw, t + hw, t < hw);
      let kp = unpack4x8snorm(Kq[rowQ + wp]) * Ks[rowS + (wp >> 3u)];
      let rot = select(kp, -kp, t < hw);
      let cb = j * half + select(t - hw, t, t < hw) * 4u;
      let cs = vec4<f32>(cosT[cb], cosT[cb + 1u], cosT[cb + 2u], cosT[cb + 3u]);
      let sn = vec4<f32>(sinT[cb], sinT[cb + 1u], sinT[cb + 2u], sinT[cb + 3u]);
      part = dot(qv, kw * cs + rot * sn);
    }
    red[t] = part;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
      if (t < s) { red[t] = red[t] + red[t + s]; }
      workgroupBarrier();
    }
    let score = red[0] * inv;
    workgroupBarrier();
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let wgt = exp(score - mnew);
    l = l * corr + wgt;
    if (t < W4) {
      let vw = unpack4x8snorm(Vq[rowQ + t]) * Vs[rowS + (t >> 3u)];
      acc = acc * corr + wgt * vw;
    }
    m = mnew;
  }
  if (t < W4) {
    let ob = (qi * p.H + h) * p.D;
    let o = acc / l;
    out[ob + t * 4u] = o.x;
    out[ob + t * 4u + 1u] = o.y;
    out[ob + t * 4u + 2u] = o.z;
    out[ob + t * 4u + 3u] = o.w;
  }
}
`,attention_wg_roll:`// attention_wg for the rolling-window / attention-sinks mode: no-subgroup fallback of
// attention_sg_roll (see there for the rope-at-read scheme). Keep in lockstep with
// attention_wg.wgsl: the ONLY differences are the shared-memory K stage (kk) - the rotate
// partner d±D/2 may live in another thread's stride - and the rotation in the score loop,
// written as \`k*cos + rot*sin\` with the same operand order as rmsnorm_rope_sg.
struct Params { S: u32, H: u32, KV: u32, D: u32, posBase: u32, Ltot: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;        // [S, H, D] (roped, cache-relative)
@group(0) @binding(2) var<storage, read> Kc: array<f32>;       // [Ltot, KV, D] UNROPED
@group(0) @binding(3) var<storage, read> Vc: array<f32>;       // [Ltot, KV, D]
@group(0) @binding(4) var<storage, read> cosT: array<f32>;     // [positions, D/2]
@group(0) @binding(5) var<storage, read> sinT: array<f32>;     // [positions, D/2]
@group(0) @binding(6) var<storage, read_write> out: array<f32>; // [S, H, D]
var<workgroup> red: array<f32, 64>;
var<workgroup> kk: array<f32, 128>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let idx = wg.x;                        // uniform across the workgroup -> early return is barrier-safe
  if (idx >= p.S * p.H) { return; }
  let tid = lid.x;
  let h = idx % p.H;
  let qi = idx / p.H;
  let pos = p.posBase + qi;
  let kvh = h / (p.H / p.KV);
  let qb = (qi * p.H + h) * p.D;
  let inv = 1.0 / sqrt(f32(p.D));
  let half = p.D / 2u;

  var acc: array<f32, 2>;
  acc[0] = 0.0;
  acc[1] = 0.0;
  var m = -1e30;
  var l = 0.0;
  for (var j = 0u; j <= pos; j = j + 1u) {
    let kb = (j * p.KV + kvh) * p.D;
    for (var t = 0u; t < 2u; t = t + 1u) {
      let d = tid + t * 64u;
      if (d < p.D) { kk[d] = Kc[kb + d]; }
    }
    workgroupBarrier();
    var part = 0.0;
    for (var t = 0u; t < 2u; t = t + 1u) {
      let d = tid + t * 64u;
      if (d < p.D) {
        var rot: f32;
        if (d < half) { rot = -kk[d + half]; } else { rot = kk[d - half]; }
        let rb = j * half + (d % half);
        part = part + q[qb + d] * (kk[d] * cosT[rb] + rot * sinT[rb]);
      }
    }
    red[tid] = part;
    workgroupBarrier();
    for (var s = 32u; s > 0u; s = s >> 1u) {
      if (tid < s) { red[tid] = red[tid] + red[tid + s]; }
      workgroupBarrier();
    }
    let score = red[0] * inv;            // full q.k dot, visible to all threads
    workgroupBarrier();                  // red[0] + kk consumed before the next position overwrites them
    let mnew = max(m, score);
    let corr = exp(m - mnew);
    let w = exp(score - mnew);
    l = l * corr + w;
    for (var t = 0u; t < 2u; t = t + 1u) {
      let d = tid + t * 64u;
      if (d < p.D) { acc[t] = acc[t] * corr + w * Vc[kb + d]; }
    }
    m = mnew;
  }
  let ob = (qi * p.H + h) * p.D;
  for (var t = 0u; t < 2u; t = t + 1u) {
    let d = tid + t * 64u;
    if (d < p.D) { out[ob + d] = acc[t] / l; }
  }
}
`,conv1d_causal:`// Depthwise causal Conv1d (kernel width K) + SiLU, for the gated-DeltaNet q/k/v stream. x is
// [S, C] (C = conv_dim channels), weight is [C, K] (per-channel taps, the GGUF ssm_conv1d layout).
// Carries a persistent left-context so segmented prefill and token-by-token decode continue across
// calls: state_in / state_out hold the last K-1 inputs ([K-1, C]); loadState!=0 uses them (else the
// causal left pad is zero). Extended input ext = [state_in (K-1), x (S)]:
//   y[t,c] = silu( sum_{j<K} w[c,j] * ext[t+j, c] ),   state_out[i,c] = ext[S+i, c]  (i < K-1)
struct Params { S: u32, C: u32, K: u32, loadState: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;          // [S, C]
@group(0) @binding(2) var<storage, read> w: array<f32>;          // [C, K]
@group(0) @binding(3) var<storage, read> state_in: array<f32>;   // [K-1, C]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;    // [S, C]
@group(0) @binding(5) var<storage, read_write> state_out: array<f32>; // [K-1, C]

fn ext(m: u32, c: u32) -> f32 {
  if (m + 1u < p.K) { return select(0.0, state_in[m * p.C + c], p.loadState != 0u); }
  return x[(m - (p.K - 1u)) * p.C + c];
}

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  let outN = p.S * p.C;
  if (i < outN) {
    let c = i % p.C;
    let t = i / p.C;
    var acc = 0.0;
    for (var j = 0u; j < p.K; j = j + 1u) { acc = acc + w[c * p.K + j] * ext(t + j, c); }
    y[i] = acc / (1.0 + exp(-acc));  // SiLU
  } else if (i < outN + (p.K - 1u) * p.C) {
    let si = i - outN;
    let sc = si % p.C;
    let sk = si / p.C;                 // 0 .. K-2
    state_out[sk * p.C + sc] = ext(p.S + sk, sc);
  }
}
`,copy:`// Copy src[0..n) into dst[dstOff..dstOff+n). Used to append K/V into the persistent cache.
struct Params { n: u32, dstOff: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (i >= p.n) { return; }
  dst[p.dstOff + i] = src[i];
}
`,copy_kv16:`// copy with an f16-STORAGE destination (kvCache: 'f16'): appends f32 K/V rows into the f16
// cache (one f32 -> f16 rounding per value). Keep in lockstep with copy.wgsl.
enable f16;
struct Params { n: u32, dstOff: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;
@group(0) @binding(2) var<storage, read_write> dst: array<f16>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (i >= p.n) { return; }
  dst[p.dstOff + i] = f16(src[i]);
}
`,copy_kv8:`// q8 cache append (kvCache: 'q8'): quantize f32 K/V rows into the packed-snorm8 cache, one f32
// scale per 32-element block (llama.cpp q8_0-style). One 64-thread workgroup per row of D
// elements: thread t owns packed word t (4 consecutive values), the workgroup reduces per-block
// absolute maxima through shared memory, then packs with pack4x8snorm. Replaces copy/copy_kv16
// at every cache-append site under q8. All attention arithmetic stays f32; the precision loss is
// exactly one snorm8 rounding of K/V at write time, nothing compounding.
struct Params { rows: u32, D: u32, dstRow0: u32, _p: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;          // [rows, D]
@group(0) @binding(2) var<storage, read_write> dstQ: array<u32>;   // packed 4 x snorm8 per word
@group(0) @binding(3) var<storage, read_write> dstS: array<f32>;   // [.., D/32] block scales

var<workgroup> wabs: array<f32, 64>; // per-word abs max (D <= 256 -> at most 64 words)
var<workgroup> wblk: array<f32, 8>;  // per-block scale (D/32 <= 8 blocks)

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x;                    // uniform across the workgroup -> early return is barrier-safe
  if (row >= p.rows) { return; }
  let t = lid.x;
  let W4 = p.D / 4u;
  let base = row * p.D;
  var v = vec4<f32>(0.0);
  if (t < W4) {
    v = vec4<f32>(src[base + t * 4u], src[base + t * 4u + 1u], src[base + t * 4u + 2u], src[base + t * 4u + 3u]);
    wabs[t] = max(max(abs(v.x), abs(v.y)), max(abs(v.z), abs(v.w)));
  }
  workgroupBarrier();
  if (t < p.D / 32u) {
    var m = 0.0;
    for (var i = 0u; i < 8u; i = i + 1u) { m = max(m, wabs[t * 8u + i]); }
    let s = max(m, 1e-30);           // an all-zero block packs zeros, never NaN
    wblk[t] = s;
    dstS[(p.dstRow0 + row) * (p.D / 32u) + t] = s;
  }
  workgroupBarrier();
  if (t < W4) {
    dstQ[(p.dstRow0 + row) * W4 + t] = pack4x8snorm(v / wblk[t >> 3u]);
  }
}
`,deltanet_gbeta:`// DeltaNet gate/decay compute: from the a (decay input) and b (beta input) projections,
//   g[s,h]    = a_neg[h] * softplus(a[s,h] + dt_bias[h])     (<= 0, log-space decay)
//   beta[s,h] = sigmoid(b[s,h])
// per value head h. a_neg is -exp(A_log): the PrismML GGUF stores this pre-computed in the ssm_a
// tensor (verified against the transformers A_log), so no exp() here. One invocation per (s,h);
// output is [g (S*H) ; beta (S*H)] concatenated (engine binds two sub-ranges). Matches qwen35_numpy.
struct Params { S: u32, H: u32, _p0: u32, _p1: u32 };  // H = num_value_heads
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> a: array<f32>;        // [S, H]
@group(0) @binding(2) var<storage, read> b: array<f32>;        // [S, H]
@group(0) @binding(3) var<storage, read> a_neg: array<f32>;    // [H] = -exp(A_log)
@group(0) @binding(4) var<storage, read> dt_bias: array<f32>;  // [H]
@group(0) @binding(5) var<storage, read_write> out: array<f32>;// [2*S*H]: g then beta

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  let n = p.S * p.H;
  if (i >= n) { return; }
  let h = i % p.H;
  let x = a[i] + dt_bias[h];
  let sp = max(x, 0.0) + log(1.0 + exp(-abs(x)));   // softplus (stable)
  out[i] = a_neg[h] * sp;                            // g  (a_neg already = -exp(A_log))
  out[n + i] = 1.0 / (1.0 + exp(-b[i]));             // beta
}
`,deltanet_norm_gate:`// Gated RMSNorm for the DeltaNet output: y = gamma * rmsnorm(core) * silu(z), normalized over the
// value head dim (one workgroup per head-vector row). Unlike the model's plain RMSNorm this uses
// the weight directly (not 1+weight), matching tools/qwen35_numpy (Qwen3NextRMSNormGated).
override WG: u32 = 128u;
struct Params { rows: u32, DV: u32, eps: f32, _pad: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> core: array<f32>;   // [rows, DV]
@group(0) @binding(2) var<storage, read> z: array<f32>;      // [rows, DV] gate
@group(0) @binding(3) var<storage, read> gamma: array<f32>;  // [DV]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;// [rows, DV]
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x;
  if (row >= p.rows) { return; }
  let tid = lid.x;
  let base = row * p.DV;
  var s = 0.0;
  for (var i = tid; i < p.DV; i = i + WG) { let c = core[base + i]; s = s + c * c; }
  sdata[tid] = s;
  workgroupBarrier();
  for (var st = WG / 2u; st > 0u; st = st >> 1u) {
    if (tid < st) { sdata[tid] = sdata[tid] + sdata[tid + st]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(sdata[0] / f32(p.DV) + p.eps);
  for (var i = tid; i < p.DV; i = i + WG) {
    let zz = z[base + i];
    y[base + i] = gamma[i] * (core[base + i] * inv) * (zz / (1.0 + exp(-zz)));  // * silu(z)
  }
}
`,deltanet_recur:`// Gated DeltaNet recurrent scan (the sequential O(1)/token gated delta rule; bitgpu's decode path
// and a correctness reference for prefill). One workgroup per value head; thread \`dv\` owns value
// column dv of the per-head state S[dk,dv], held in registers across the token loop. Per token:
//   S *= exp(g);  kv = Kn·S;  delta = (v - kv)·beta;  S += Kn⊗delta;  out = Qn·S
// with Kn = l2norm(k), Qn = l2norm(q)/sqrt(dk) (matches tools/qwen35_numpy._delta_recurrent).
// GQA: value head h reads q/k from key head h%HK. GGUF/bitgpu store value heads grouped
// [rep, n_key_heads] (transposed from HF's [n_key_heads, rep]), so the shared key/query head is
// h%HK (a "tile"), NOT h/(H/HK) (a "repeat-interleave"). loadState!=0 continues from state_in
// (persistent decode/cross-segment state); state_out always carries the final state out.
override WGV: u32 = 128u;                 // threads per workgroup == head_v_dim (dv)
struct Params { S: u32, H: u32, DK: u32, DV: u32, HK: u32, betaOff: u32, loadState: u32, tOff: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> q: array<f32>;      // [S, HK, DK]
@group(0) @binding(2) var<storage, read> k: array<f32>;      // [S, HK, DK]
@group(0) @binding(3) var<storage, read> v: array<f32>;      // [S, H, DV]
@group(0) @binding(4) var<storage, read> g: array<f32>;      // [S, H]
@group(0) @binding(5) var<storage, read> beta: array<f32>;   // [S, H]
@group(0) @binding(6) var<storage, read> state_in: array<f32>;    // [H, DK, DV]
@group(0) @binding(7) var<storage, read_write> core: array<f32>;  // [S, H, DV]
@group(0) @binding(8) var<storage, read_write> state_out: array<f32>; // [H, DK, DV]
var<workgroup> ksh: array<f32, 128>;      // current token's raw k (>= DK)
var<workgroup> qsh: array<f32, 128>;      // current token's raw q

@compute @workgroup_size(WGV)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let h = wg.x;                           // value head
  let hk = h % p.HK;                      // GQA: shared key/query head (GGUF [rep,n_key] tile order)
  let dv = lid.x;
  let DK = p.DK;
  let sbase = h * DK * p.DV + dv;         // state column S[:, dv] of head h, stride DV
  let scale = inverseSqrt(f32(DK));
  var s: array<f32, 128>;                 // state column S[:, dv], length DK
  for (var dk = 0u; dk < DK; dk = dk + 1u) { s[dk] = select(0.0, state_in[sbase + dk * p.DV], p.loadState != 0u); }

  for (var t = 0u; t < p.S; t = t + 1u) {
    let base = (t + p.tOff) * p.H + h;    // value-head row (v, g, beta, out); tOff = this chunk's
    let basek = (t + p.tOff) * p.HK + hk; // token offset when a long segment's scan is sub-chunked
    for (var i = lid.x; i < DK; i = i + WGV) { ksh[i] = k[basek * DK + i]; qsh[i] = q[basek * DK + i]; }
    workgroupBarrier();
    var sk = 0.0;
    var sq = 0.0;
    for (var dk = 0u; dk < DK; dk = dk + 1u) { sk = sk + ksh[dk] * ksh[dk]; sq = sq + qsh[dk] * qsh[dk]; }
    let ik = inverseSqrt(sk + 1e-6);              // l2norm(k)
    let iq = inverseSqrt(sq + 1e-6) * scale;      // l2norm(q) / sqrt(dk)
    if (dv < p.DV) {
      let gt = exp(g[base]);
      let bt = beta[p.betaOff + base];   // beta may share g's buffer (engine: gbeta = [g; beta])
      for (var dk = 0u; dk < DK; dk = dk + 1u) { s[dk] = s[dk] * gt; }   // decay
      var kv = 0.0;
      for (var dk = 0u; dk < DK; dk = dk + 1u) { kv = kv + s[dk] * ksh[dk] * ik; }
      let delta = (v[base * p.DV + dv] - kv) * bt;
      var o = 0.0;
      for (var dk = 0u; dk < DK; dk = dk + 1u) {
        s[dk] = s[dk] + ksh[dk] * ik * delta;      // S += Kn ⊗ delta
        o = o + s[dk] * qsh[dk] * iq;              // out = Qn · S (updated)
      }
      core[base * p.DV + dv] = o;
    }
    workgroupBarrier();
  }
  if (dv < p.DV) { for (var dk = 0u; dk < DK; dk = dk + 1u) { state_out[sbase + dk * p.DV] = s[dk]; } }
}
`,embed_gather:`// GPU embedding gather + 4-bit dequant: reads a token id from a GPU buffer and writes that token's
// embedding (H f32) directly into a GPU buffer, so the decode loop never round-trips the token id to
// the CPU. Faithful port of the CPU embedDequant (4-bit codes via the tgt4 LUT, per-128 zero-point,
// per-block scale). uint8 source arrays are read as u32 and byte-extracted (little-endian).
override WG: u32 = 256u;
struct Params { H: u32, srcIdx: u32, _0: u32, _1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> tokenId: array<u32>;   // tokenId[p.srcIdx] = the token to embed
@group(0) @binding(2) var<storage, read> embWq: array<u32>;     // uint8 [vocab * H/8] packed
@group(0) @binding(3) var<storage, read> tgt4: array<u32>;      // uint8 [256*4] packed (1 src byte -> 4)
@group(0) @binding(4) var<storage, read> embScales: array<f32>;// [vocab * H/128]
@group(0) @binding(5) var<storage, read> embZp: array<u32>;    // uint8 [vocab * ceil(H/256)] packed
@group(0) @binding(6) var<storage, read_write> out: array<f32>;// [H]

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let id = tokenId[p.srcIdx];
  // Per-row strides derived from H: rowBytes source bytes, scaleRow f32 scales,
  // zpRow packed zero-point bytes (H=2048 -> 256/16/8, H=2560 -> 320/20/10).
  let rowBytes = p.H >> 3u;
  let scaleRow = p.H >> 7u;
  let zpRow = (scaleRow + 1u) >> 1u;
  for (var k = lid.x; k < p.H; k = k + WG) {
    let i = k >> 3u;
    let qd = (k >> 1u) & 3u;
    let c = k & 1u;
    let wqIdx = id * rowBytes + i;
    let e = (embWq[wqIdx >> 2u] >> (8u * (wqIdx & 3u))) & 0xffu;   // source byte 0..255
    let tIdx = 4u * e + qd;
    let t = (tgt4[tIdx >> 2u] >> (8u * (tIdx & 3u))) & 0xffu;       // expanded byte (2 codes)
    let code = (t >> (4u * c)) & 0xfu;
    let blk = k >> 7u;
    let zpIdx = id * zpRow + (blk >> 1u);
    let zpByte = (embZp[zpIdx >> 2u] >> (8u * (zpIdx & 3u))) & 0xffu;
    let zp = (zpByte >> (4u * (blk & 1u))) & 0xfu;
    out[k] = (f32(code) - f32(zp)) * embScales[id * scaleRow + blk];
  }
}
`,embed_gather_batch:`// Batched GPU embedding gather + 4-bit dequant for PROMPT tokens: one invocation per output
// element writes out[s*H + k] for tokenIds[s]. A prefill segment uploads S u32 token ids
// instead of S*H dequantized floats, so the CPU-side embedding tables are not needed at all
// (~50-100 MB RAM per model). Same per-row stride math and dequant as embed_gather.wgsl
// (H=2048 -> 256/16/8 strides); uint8 sources read as u32 and byte-extracted (little-endian).
struct Params { S: u32, H: u32, _0: u32, _1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> tokenIds: array<u32>;   // [S]
@group(0) @binding(2) var<storage, read> embWq: array<u32>;      // uint8 [vocab * H/8] packed
@group(0) @binding(3) var<storage, read> tgt4: array<u32>;       // uint8 [256*4] packed (1 src byte -> 4)
@group(0) @binding(4) var<storage, read> embScales: array<f32>; // [vocab * H/128]
@group(0) @binding(5) var<storage, read> embZp: array<u32>;     // uint8 [vocab * ceil(H/256)] packed
@group(0) @binding(6) var<storage, read_write> out: array<f32>; // [S * H]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let gi = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (gi >= p.S * p.H) { return; }
  let k = gi % p.H;
  let id = tokenIds[gi / p.H];
  let rowBytes = p.H >> 3u;
  let scaleRow = p.H >> 7u;
  let zpRow = (scaleRow + 1u) >> 1u;
  let i = k >> 3u;
  let qd = (k >> 1u) & 3u;
  let c = k & 1u;
  let wqIdx = id * rowBytes + i;
  let e = (embWq[wqIdx >> 2u] >> (8u * (wqIdx & 3u))) & 0xffu;   // source byte 0..255
  let tIdx = 4u * e + qd;
  let t = (tgt4[tIdx >> 2u] >> (8u * (tIdx & 3u))) & 0xffu;       // expanded byte (2 codes)
  let code = (t >> (4u * c)) & 0xfu;
  let blk = k >> 7u;
  let zpIdx = id * zpRow + (blk >> 1u);
  let zpByte = (embZp[zpIdx >> 2u] >> (8u * (zpIdx & 3u))) & 0xffu;
  let zp = (zpByte >> (4u * (blk & 1u))) & 0xfu;
  out[gi] = (f32(code) - f32(zp)) * embScales[id * scaleRow + blk];
}
`,gate_sigmoid:`// Output gate for Qwen3.5 gated attention: y = x * sigmoid(gate), elementwise. Applied to the
// attention output before o_proj (the gate is the second half of the doubled q_proj).
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gate: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (i >= p.n) { return; }
  y[i] = x[i] / (1.0 + exp(-gate[i]));
}
`,logsumexp:`// log-sum-exp over the (penalty-filtered) logits, the softmax normalizer that turns a raw logit
// into a true logprob on the CPU: logprob(id) = logit[id] - lse. Runs AFTER sampler_penalty and
// BEFORE the argmax_masked rounds (those mask their winners in place, which would corrupt the
// sum). Two-phase single-workgroup reduction: strided max, then strided sum of exp(x - max);
// entries at the -inf sentinel (banned ids) contribute nothing. Only one f32 is read back.
override WG: u32 = 256u;
struct Params { N: u32, _0: u32, _1: u32, _2: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> outLse: array<f32>;   // outLse[0] = max + log(sum)

const NEG_SENTINEL: f32 = -3.0e38;   // below any real logit; banned entries sit at f32 -inf

var<workgroup> sval: array<f32, 256>;

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  var m = -3.4e38;
  for (var i = tid; i < p.N; i = i + WG) {
    let v = logits[i];
    if (v > NEG_SENTINEL && v > m) { m = v; }
  }
  sval[tid] = m;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s && sval[tid + s] > sval[tid]) { sval[tid] = sval[tid + s]; }
    workgroupBarrier();
  }
  let gmax = sval[0];
  workgroupBarrier();
  var acc = 0.0;
  for (var i = tid; i < p.N; i = i + WG) {
    let v = logits[i];
    if (v > NEG_SENTINEL) { acc = acc + exp(v - gmax); }
  }
  sval[tid] = acc;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) { sval[tid] = sval[tid] + sval[tid + s]; }
    workgroupBarrier();
  }
  if (tid == 0u) { outLse[0] = gmax + log(sval[0]); }
}
`,matmul_binary_vec4:`// Binary matmul, vectorized: y[M,N] = x[M,K] @ W[N,K]^T, W = (+/-1) * per-block scale.
// One thread per output; the K loop processes a 32-bit sign word at a time and the
// activations as vec4 via dot() (4 weights per FMA instead of 1 scalar op). M-agnostic
// (works for prefill M=S and decode M=1). x is bound as vec4 (K must be a multiple of 4).
struct Params { M: u32, N: u32, K: u32, nb: u32, block: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [M, K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;   // [M, N]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (idx >= p.M * p.N) { return; }
  let m = idx / p.N;
  let n = idx % p.N;
  let xRow = m * (p.K / 4u);
  let wRow = n * (p.K / 32u);
  let sbase = n * p.nb;
  let wordsPerBlock = p.block / 32u;   // 4 for block=128

  var acc = 0.0;
  for (var b = 0u; b < p.nb; b = b + 1u) {
    var bsum = 0.0;
    for (var w = 0u; w < wordsPerBlock; w = w + 1u) {
      let word = signbits[wRow + b * wordsPerBlock + w];
      let xb = xRow + b * (p.block / 4u) + w * 8u;   // vec4 base for this word (32 weights = 8 vec4)
      for (var g = 0u; g < 8u; g = g + 1u) {
        let bits4 = (word >> (g * 4u)) & 0xfu;
        let sv = vec4<f32>(
          select(-1.0, 1.0, (bits4 & 1u) != 0u),
          select(-1.0, 1.0, (bits4 & 2u) != 0u),
          select(-1.0, 1.0, (bits4 & 4u) != 0u),
          select(-1.0, 1.0, (bits4 & 8u) != 0u),
        );
        bsum = bsum + dot(x[xb + g], sv);
      }
    }
    acc = acc + bsum * scales[sbase + b];
  }
  y[idx] = acc;
}
`,matmul_q2:`// 2-bit dequant matmul (lm_head): y[M,N] = x[M,K] @ W[N,K]^T, W[n,k] = (code - zp) * scale[n, k/block].
// codes are 2-bit, 4 per byte, packed into u32 words. Correctness-first (one thread per output, fp32).
struct Params { M: u32, N: u32, K: u32, nb: u32, block: u32, zp: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;       // [M, K]
@group(0) @binding(2) var<storage, read> codes: array<u32>;   // [N, K/4] bytes packed as u32
@group(0) @binding(3) var<storage, read> scales: array<f32>;  // [N, nb]
@group(0) @binding(4) var<storage, read_write> y: array<f32>; // [M, N]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (idx >= p.M * p.N) { return; }
  let m = idx / p.N;
  let n = idx % p.N;
  let xbase = m * p.K;
  let cbyteBase = n * (p.K / 4u);   // byte offset of row n in the codes stream
  let sbase = n * p.nb;
  let zpf = f32(p.zp);

  var acc = 0.0;
  for (var b = 0u; b < p.nb; b = b + 1u) {
    var bsum = 0.0;
    let k0 = b * p.block;
    for (var j = 0u; j < p.block; j = j + 1u) {
      let k = k0 + j;
      let byteIdx = cbyteBase + (k >> 2u);
      let word = codes[byteIdx >> 2u];
      let byte = (word >> (8u * (byteIdx & 3u))) & 0xffu;
      let code = (byte >> (2u * (k & 3u))) & 3u;
      bsum = bsum + (f32(code) - zpf) * x[xbase + k];
    }
    acc = acc + bsum * scales[sbase + b];
  }
  y[idx] = acc;
}
`,matmul_q2_sg:`// Subgroup split-K GEMV for the 2-bit lm_head (M=1 decode). One subgroup per output column,
// lanes split K (vec4), reduce with subgroupAdd. value = (code - zp) * per-block scale.
// 2D dispatch since N (vocab) > 65535.
enable subgroups;
override SG: u32 = 32u;
struct Params { N: u32, K: u32, nb: u32, zp: u32, gridX: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [K/4]
@group(0) @binding(2) var<storage, read> codes: array<u32>;     // [N, K/4] bytes packed as u32
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;   // [N]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let n = wg.y * p.gridX + wg.x;
  if (n >= p.N) { return; }
  let cbase = n * (p.K / 4u);     // byte offset of row n in the codes stream
  let sbase = n * p.nb;
  let zpf = f32(p.zp);
  let Kvec = p.K / 4u;

  var acc = 0.0;
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let byteIdx = cbase + gi;
    let word = codes[byteIdx >> 2u];
    let byte = (word >> (8u * (byteIdx & 3u))) & 0xffu;
    let cv = vec4<f32>(f32(byte & 3u) - zpf, f32((byte >> 2u) & 3u) - zpf,
                       f32((byte >> 4u) & 3u) - zpf, f32((byte >> 6u) & 3u) - zpf);
    acc = acc + dot(x[gi], cv) * scales[sbase + (gi >> 5u)];   // block = (gi*4)/128 = gi/32
  }
  let total = subgroupAdd(acc);
  if (lane == 0u) { y[n] = total; }
}
`,matmul_q2_sm:`// Small-batch (M = 2..9) subgroup split-K GEMV for the 2-bit lm_head: the speculative-decode
// verify pass needs logits for every drafted row, and the scalar M-row kernel re-reads the
// ~77 MB code stream per output thread. Here each code word is loaded once per (column,
// k-chunk) and dotted with all M rows. Per row the loop stride and accumulation expression
// match matmul_q2_sg, so each row is bit-identical to the M=1 decode path.
enable subgroups;
override SG: u32 = 32u;
struct Params { N: u32, K: u32, nb: u32, zp: u32, gridX: u32, M: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [M, K/4] row-major
@group(0) @binding(2) var<storage, read> codes: array<u32>;     // [N, K/4] bytes packed as u32
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;   // [M, N]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let n = wg.y * p.gridX + wg.x;
  if (n >= p.N) { return; }
  let cbase = n * (p.K / 4u);
  let sbase = n * p.nb;
  let zpf = f32(p.zp);
  let Kvec = p.K / 4u;

  var acc: array<f32, 9>; // M <= 9
  for (var m = 0u; m < p.M; m = m + 1u) { acc[m] = 0.0; }
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let byteIdx = cbase + gi;
    let word = codes[byteIdx >> 2u];
    let byte = (word >> (8u * (byteIdx & 3u))) & 0xffu;
    let cv = vec4<f32>(f32(byte & 3u) - zpf, f32((byte >> 2u) & 3u) - zpf,
                       f32((byte >> 4u) & 3u) - zpf, f32((byte >> 6u) & 3u) - zpf);
    let s = scales[sbase + (gi >> 5u)]; // block = (gi*4)/128 = gi/32
    for (var m = 0u; m < p.M; m = m + 1u) {
      acc[m] = acc[m] + dot(x[m * Kvec + gi], cv) * s;
    }
  }
  for (var m = 0u; m < p.M; m = m + 1u) {
    let total = subgroupAdd(acc[m]);
    if (lane == 0u) { y[m * p.N + n] = total; }
  }
}
`,matmul_q2_wg:`// No-subgroup fallback: 2-bit lm_head GEMV for decode (M=1), workgroup-shared-memory reduction.
// One workgroup per output column; WG threads split K and tree-reduce. value = (code - zp) * scale.
// 2D dispatch since N (vocab) > 65535. This is the v1 path's biggest cost (scalar was ~48ms/token).
override WG: u32 = 64u;
struct Params { N: u32, K: u32, nb: u32, zp: u32, gridX: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [K/4]
@group(0) @binding(2) var<storage, read> codes: array<u32>;     // [N, K/4] bytes packed as u32
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;   // [N]
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = wg.y * p.gridX + wg.x;
  if (n >= p.N) { return; }
  let tid = lid.x;
  let cbase = n * (p.K / 4u);
  let sbase = n * p.nb;
  let zpf = f32(p.zp);
  let Kvec = p.K / 4u;
  var acc = 0.0;
  for (var gi = tid; gi < Kvec; gi = gi + WG) {
    let byteIdx = cbase + gi;
    let word = codes[byteIdx >> 2u];
    let byte = (word >> (8u * (byteIdx & 3u))) & 0xffu;
    let cv = vec4<f32>(f32(byte & 3u) - zpf, f32((byte >> 2u) & 3u) - zpf,
                       f32((byte >> 4u) & 3u) - zpf, f32((byte >> 6u) & 3u) - zpf);
    acc = acc + dot(x[gi], cv) * scales[sbase + (gi >> 5u)];
  }
  sdata[tid] = acc;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) { sdata[tid] = sdata[tid] + sdata[tid + s]; }
    workgroupBarrier();
  }
  if (tid == 0u) { y[n] = sdata[0]; }
}
`,matmul_resid:`// Binary matmul with a fused residual add: y[M,N] = x[M,K] @ W[N,K]^T + resid[M,N].
// Folds the residual add into o_proj / down_proj so it's not a separate dispatch.
struct Params { M: u32, N: u32, K: u32, nb: u32, block: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [M, K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read> resid: array<f32>;     // [M, N]
@group(0) @binding(5) var<storage, read_write> y: array<f32>;   // [M, N]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (idx >= p.M * p.N) { return; }
  let n = idx % p.N;
  let xRow = (idx / p.N) * (p.K / 4u);
  let wRow = n * (p.K / 32u);
  let sbase = n * p.nb;

  var acc = 0.0;
  for (var b = 0u; b < p.nb; b = b + 1u) {
    var bsum = 0.0;
    for (var w = 0u; w < 4u; w = w + 1u) {
      let word = signbits[wRow + b * 4u + w];
      let xb = xRow + b * 32u + w * 8u;
      for (var g = 0u; g < 8u; g = g + 1u) {
        let bits4 = (word >> (g * 4u)) & 0xfu;
        let sv = vec4<f32>(select(-1.0, 1.0, (bits4 & 1u) != 0u), select(-1.0, 1.0, (bits4 & 2u) != 0u),
                           select(-1.0, 1.0, (bits4 & 4u) != 0u), select(-1.0, 1.0, (bits4 & 8u) != 0u));
        bsum = bsum + dot(x[xb + g], sv);
      }
    }
    acc = acc + bsum * scales[sbase + b];
  }
  y[idx] = acc + resid[idx];
}
`,matmul_resid_mr_sg:`// Multi-row subgroup GEMV for decode (M=1) with fused residual. Same as matmul_resid_sg but each
// workgroup computes ROWS output columns at once: per K-step it issues ROWS independent weight
// loads before the dots, giving the memory system more in-flight requests (memory-level
// parallelism) to hide latency on the bandwidth-bound decode GEMV. One subgroup per workgroup;
// lanes split K; ROWS accumulators reduced with subgroupAdd. value = sign * per-block scale.
enable subgroups;
override SG: u32 = 32u;
override ROWS: u32 = 4u;
struct Params { N: u32, K: u32, nb: u32, gridX: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read> resid: array<f32>;     // [N]
@group(0) @binding(5) var<storage, read_write> y: array<f32>;   // [N]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let rowBase = (wg.y * p.gridX + wg.x) * ROWS;
  let Kvec = p.K / 4u;
  let wStride = p.K / 32u;

  var acc: array<f32, 8>;                         // ROWS <= 8
  for (var r = 0u; r < ROWS; r = r + 1u) { acc[r] = 0.0; }
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let k = gi * 4u;
    let xv = x[gi];
    let widx = k >> 5u;
    let sh = k & 31u;
    let sc = k / 128u;
    for (var r = 0u; r < ROWS; r = r + 1u) {
      let n = rowBase + r;
      if (n < p.N) {
        let w = (signbits[n * wStride + widx] >> sh) & 0xfu;
        let sv = vec4<f32>(select(-1.0, 1.0, (w & 1u) != 0u), select(-1.0, 1.0, (w & 2u) != 0u),
                           select(-1.0, 1.0, (w & 4u) != 0u), select(-1.0, 1.0, (w & 8u) != 0u));
        acc[r] = acc[r] + dot(xv, sv) * scales[n * p.nb + sc];
      }
    }
  }
  for (var r = 0u; r < ROWS; r = r + 1u) {
    let n = rowBase + r;
    let total = subgroupAdd(acc[r]);             // collective: called for every r by all lanes
    if (lane == 0u && n < p.N) { y[n] = total + resid[n]; }
  }
}
`,matmul_resid_mr_sg_af16:`// f16-activation variant of matmul_resid_mr_sg (multi-row decode GEMV + fused residual, M=1),
// used for down_proj (its input is the f16 SwiGLU intermediate). Reads f16 x, dots in f16,
// accumulates in f32; the residual add and the output stay f32. Weights unchanged.
enable subgroups;
enable f16;
override SG: u32 = 32u;
override ROWS: u32 = 4u;
struct Params { N: u32, K: u32, nb: u32, gridX: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f16>>;   // [K/4] f16 activations
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read> resid: array<f32>;     // [N]
@group(0) @binding(5) var<storage, read_write> y: array<f32>;   // [N]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let rowBase = (wg.y * p.gridX + wg.x) * ROWS;
  let Kvec = p.K / 4u;
  let wStride = p.K / 32u;

  var acc: array<f32, 8>;                         // ROWS <= 8
  for (var r = 0u; r < ROWS; r = r + 1u) { acc[r] = 0.0; }
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let k = gi * 4u;
    let xv = x[gi];
    let widx = k >> 5u;
    let sh = k & 31u;
    let sc = k / 128u;
    for (var r = 0u; r < ROWS; r = r + 1u) {
      let n = rowBase + r;
      if (n < p.N) {
        let w = (signbits[n * wStride + widx] >> sh) & 0xfu;
        let sv = vec4<f16>(select(-1.0h, 1.0h, (w & 1u) != 0u), select(-1.0h, 1.0h, (w & 2u) != 0u),
                           select(-1.0h, 1.0h, (w & 4u) != 0u), select(-1.0h, 1.0h, (w & 8u) != 0u));
        acc[r] = acc[r] + f32(dot(xv, sv)) * scales[n * p.nb + sc];
      }
    }
  }
  for (var r = 0u; r < ROWS; r = r + 1u) {
    let n = rowBase + r;
    let total = subgroupAdd(acc[r]);
    if (lane == 0u && n < p.N) { y[n] = total + resid[n]; }
  }
}
`,matmul_resid_sm:`// Small-batch (M = 2..9) subgroup split-K GEMV with fused residual add (o_proj / down_proj in
// the speculative-decode verify pass). One workgroup per output column; each weight word is
// loaded once and dotted with all M activation rows. Per row the loop stride and accumulation
// expression match the validated M=1 kernels, so results are row-wise bit-identical to them.
enable subgroups;
override SG: u32 = 32u;
struct Params { N: u32, K: u32, nb: u32, gridX: u32, M: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [M, K/4] row-major
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read> resid: array<f32>;     // [M, N]
@group(0) @binding(5) var<storage, read_write> y: array<f32>;   // [M, N]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let n = wg.y * p.gridX + wg.x;
  if (n >= p.N) { return; }
  let wRow = n * (p.K / 32u);
  let sbase = n * p.nb;
  let Kvec = p.K / 4u;

  var acc: array<f32, 9>; // M <= 9
  for (var m = 0u; m < p.M; m = m + 1u) { acc[m] = 0.0; }
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let k = gi * 4u;
    let word = signbits[wRow + (k >> 5u)];
    let bits4 = (word >> (k & 31u)) & 0xfu;
    let sv = vec4<f32>(select(-1.0, 1.0, (bits4 & 1u) != 0u), select(-1.0, 1.0, (bits4 & 2u) != 0u),
                       select(-1.0, 1.0, (bits4 & 4u) != 0u), select(-1.0, 1.0, (bits4 & 8u) != 0u));
    let s = scales[sbase + (k / 128u)];
    for (var m = 0u; m < p.M; m = m + 1u) {
      acc[m] = acc[m] + dot(x[m * Kvec + gi], sv) * s;
    }
  }
  for (var m = 0u; m < p.M; m = m + 1u) {
    let total = subgroupAdd(acc[m]);
    if (lane == 0u) { y[m * p.N + n] = total + resid[m * p.N + n]; }
  }
}
`,matmul_resid_tiled:`// Tiled register-blocked binary GEMM with fused residual, for PREFILL (M>1), vec4 K-accumulation:
//   y[M,N] = x[M,K] @ W[N,K]^T + resid[M,N],  W binary {-1,+1} sign-packed, per-128-block fp32 scale.
// 64x64 output tile per workgroup, 16x16 threads each computing a 4x4 register tile, BK=16 K-step.
// Activation + decoded/scaled weight tiles are staged in shared memory as vec4 (4 K per element);
// each inner step is a dot() of vec4s, and one weight load decodes a whole nibble (4 signs) at once.
// No subgroup ops -> all devices. Near-bit-exact (f32 accum; tiled K-order differs in last ULPs).
const BM: u32 = 64u;
const BN: u32 = 64u;
const BKV: u32 = 4u;          // BK / 4  (BK = 16)
struct Params { M: u32, N: u32, K: u32, nb: u32, _0: u32, _1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;  // [M, K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>; // [N, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;   // [N, nb]
@group(0) @binding(4) var<storage, read> resid: array<f32>;    // [M, N]
@group(0) @binding(5) var<storage, read_write> y: array<f32>;  // [M, N]

var<workgroup> xs: array<vec4<f32>, 256>;   // BM*BKV
var<workgroup> ws: array<vec4<f32>, 256>;   // BN*BKV

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let tileM = wg.y * BM;
  let tileN = wg.x * BN;
  let tr = (tid / 16u) * 4u;
  let tc = (tid % 16u) * 4u;
  let Kv = p.K / 4u;
  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }

  let Ksteps = Kv / BKV;
  for (var ks = 0u; ks < Ksteps; ks = ks + 1u) {
    let k0v = ks * BKV;
    for (var e = tid; e < BM * BKV; e = e + 256u) {           // stage activation tile (vec4)
      let m = e / BKV; let kv = e % BKV; let gm = tileM + m;
      xs[e] = select(vec4<f32>(0.0), x[gm * Kv + (k0v + kv)], gm < p.M);
    }
    for (var e = tid; e < BN * BKV; e = e + 256u) {           // stage decoded+scaled weight tile (vec4)
      let n = e / BKV; let kv = e % BKV; let gn = tileN + n; let k = (k0v + kv) * 4u;
      var wv = vec4<f32>(0.0);
      if (gn < p.N) {
        let bits4 = (signbits[gn * (p.K / 32u) + (k >> 5u)] >> (k & 31u)) & 0xfu;
        let s = scales[gn * p.nb + (k / 128u)];
        wv = vec4<f32>(select(-s, s, (bits4 & 1u) != 0u), select(-s, s, (bits4 & 2u) != 0u),
                       select(-s, s, (bits4 & 4u) != 0u), select(-s, s, (bits4 & 8u) != 0u));
      }
      ws[e] = wv;
    }
    workgroupBarrier();
    for (var kv = 0u; kv < BKV; kv = kv + 1u) {
      var xr: array<vec4<f32>, 4>;
      for (var tm = 0u; tm < 4u; tm = tm + 1u) { xr[tm] = xs[(tr + tm) * BKV + kv]; }
      for (var tn = 0u; tn < 4u; tn = tn + 1u) {
        let w = ws[(tc + tn) * BKV + kv];
        for (var tm = 0u; tm < 4u; tm = tm + 1u) { acc[tm * 4u + tn] = acc[tm * 4u + tn] + dot(xr[tm], w); }
      }
    }
    workgroupBarrier();
  }

  for (var tm = 0u; tm < 4u; tm = tm + 1u) {
    let gm = tileM + tr + tm;
    if (gm < p.M) {
      for (var tn = 0u; tn < 4u; tn = tn + 1u) {
        let gn = tileN + tc + tn;
        if (gn < p.N) { let idx = gm * p.N + gn; y[idx] = acc[tm * 4u + tn] + resid[idx]; }
      }
    }
  }
}
`,matmul_resid_wg:`// No-subgroup fallback: split-K GEMV for decode (M=1) with fused residual, workgroup-shared-memory
// reduction. One workgroup per output column; WG threads split K and tree-reduce. Used for o_proj/down.
override WG: u32 = 64u;
struct Params { N: u32, K: u32, nb: u32, gridX: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N, nb]
@group(0) @binding(4) var<storage, read> resid: array<f32>;     // [N]
@group(0) @binding(5) var<storage, read_write> y: array<f32>;   // [N]
var<workgroup> sdata: array<f32, 256>;

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let n = wg.y * p.gridX + wg.x;
  if (n >= p.N) { return; }
  let tid = lid.x;
  let wRow = n * (p.K / 32u);
  let sbase = n * p.nb;
  let Kvec = p.K / 4u;
  var acc = 0.0;
  for (var gi = tid; gi < Kvec; gi = gi + WG) {
    let k = gi * 4u;
    let word = signbits[wRow + (k >> 5u)];
    let bits4 = (word >> (k & 31u)) & 0xfu;
    let sv = vec4<f32>(select(-1.0, 1.0, (bits4 & 1u) != 0u), select(-1.0, 1.0, (bits4 & 2u) != 0u),
                       select(-1.0, 1.0, (bits4 & 4u) != 0u), select(-1.0, 1.0, (bits4 & 8u) != 0u));
    acc = acc + dot(x[gi], sv) * scales[sbase + (k / 128u)];
  }
  sdata[tid] = acc;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) { sdata[tid] = sdata[tid] + sdata[tid + s]; }
    workgroupBarrier();
  }
  if (tid == 0u) { y[n] = sdata[0] + resid[n]; }
}
`,matmul_split:`// Fused binary matmul writing to up to 3 output buffers (qkv or gate/up in one dispatch).
// Weights for the outputs are concatenated along N (rows N0 | N1 | N2). One thread per
// output column n routes its result to out0/out1/out2 by range. Vectorized like matmul_binary_vec4.
struct Params { M: u32, K: u32, nb: u32, N0: u32, N1: u32, N2: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [M, K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N0+N1+N2, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N0+N1+N2, nb]
@group(0) @binding(4) var<storage, read_write> out0: array<f32>;
@group(0) @binding(5) var<storage, read_write> out1: array<f32>;
@group(0) @binding(6) var<storage, read_write> out2: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let Ntot = p.N0 + p.N1 + p.N2;
  let idx = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (idx >= p.M * Ntot) { return; }
  let row = idx / Ntot;
  let n = idx % Ntot;
  let xRow = row * (p.K / 4u);
  let wRow = n * (p.K / 32u);
  let sbase = n * p.nb;

  var acc = 0.0;
  for (var b = 0u; b < p.nb; b = b + 1u) {
    var bsum = 0.0;
    for (var w = 0u; w < 4u; w = w + 1u) {
      let word = signbits[wRow + b * 4u + w];
      let xb = xRow + b * 32u + w * 8u;
      for (var g = 0u; g < 8u; g = g + 1u) {
        let bits4 = (word >> (g * 4u)) & 0xfu;
        let sv = vec4<f32>(select(-1.0, 1.0, (bits4 & 1u) != 0u), select(-1.0, 1.0, (bits4 & 2u) != 0u),
                           select(-1.0, 1.0, (bits4 & 4u) != 0u), select(-1.0, 1.0, (bits4 & 8u) != 0u));
        bsum = bsum + dot(x[xb + g], sv);
      }
    }
    acc = acc + bsum * scales[sbase + b];
  }

  if (n < p.N0) { out0[row * p.N0 + n] = acc; }
  else if (n < p.N0 + p.N1) { out1[row * p.N1 + (n - p.N0)] = acc; }
  else { out2[row * p.N2 + (n - p.N0 - p.N1)] = acc; }
}
`,matmul_split_sg:`// Subgroup split-K GEMV for decode (M=1), fused: one subgroup (= one workgroup) per output
// column; lanes split the K dimension and reduce with subgroupAdd (register-only, no barriers).
// Cuts each matmul's latency ~SG-fold vs one-thread-per-output (the real decode bottleneck:
// kernels run at full latency in the dependent chain). Routes to out0/out1/out2 by range (qkv / gate-up).
enable subgroups;
override SG: u32 = 32u;
struct Params { K: u32, nb: u32, N0: u32, N1: u32, N2: u32, gridX: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N0+N1+N2, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N0+N1+N2, nb]
@group(0) @binding(4) var<storage, read_write> out0: array<f32>;
@group(0) @binding(5) var<storage, read_write> out1: array<f32>;
@group(0) @binding(6) var<storage, read_write> out2: array<f32>;

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let Ntot = p.N0 + p.N1 + p.N2;
  let n = wg.y * p.gridX + wg.x;
  if (n >= Ntot) { return; }
  let wRow = n * (p.K / 32u);
  let sbase = n * p.nb;
  let Kvec = p.K / 4u;

  var acc = 0.0;
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let k = gi * 4u;
    let word = signbits[wRow + (k >> 5u)];
    let bits4 = (word >> (k & 31u)) & 0xfu;
    let sv = vec4<f32>(select(-1.0, 1.0, (bits4 & 1u) != 0u), select(-1.0, 1.0, (bits4 & 2u) != 0u),
                       select(-1.0, 1.0, (bits4 & 4u) != 0u), select(-1.0, 1.0, (bits4 & 8u) != 0u));
    acc = acc + dot(x[gi], sv) * scales[sbase + (k / 128u)];
  }
  let total = subgroupAdd(acc);
  if (lane == 0u) {
    if (n < p.N0) { out0[n] = total; }
    else if (n < p.N0 + p.N1) { out1[n - p.N0] = total; }
    else { out2[n - p.N0 - p.N1] = total; }
  }
}
`,matmul_split_sg_af16:`// f16-activation variant of matmul_split_sg (fused QKV decode GEMV, M=1). The activation x is
// read as f16 and the per-group dot runs in f16 (2x ALU rate on Apple/AMD/recent NVIDIA); the
// per-block accumulation stays f32 (dot promoted before x scale, acc in f32) so accuracy tracks
// f32 to ~f16 rounding. Weights (sign bits + f32 block scales) are unchanged. Outputs f32.
enable subgroups;
enable f16;
override SG: u32 = 32u;
struct Params { K: u32, nb: u32, N0: u32, N1: u32, N2: u32, gridX: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f16>>;   // [K/4] f16 activations
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N0+N1+N2, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N0+N1+N2, nb]
@group(0) @binding(4) var<storage, read_write> out0: array<f32>;
@group(0) @binding(5) var<storage, read_write> out1: array<f32>;
@group(0) @binding(6) var<storage, read_write> out2: array<f32>;

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let Ntot = p.N0 + p.N1 + p.N2;
  let n = wg.y * p.gridX + wg.x;
  if (n >= Ntot) { return; }
  let wRow = n * (p.K / 32u);
  let sbase = n * p.nb;
  let Kvec = p.K / 4u;

  var acc = 0.0;
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let k = gi * 4u;
    let word = signbits[wRow + (k >> 5u)];
    let bits4 = (word >> (k & 31u)) & 0xfu;
    let sv = vec4<f16>(select(-1.0h, 1.0h, (bits4 & 1u) != 0u), select(-1.0h, 1.0h, (bits4 & 2u) != 0u),
                       select(-1.0h, 1.0h, (bits4 & 4u) != 0u), select(-1.0h, 1.0h, (bits4 & 8u) != 0u));
    acc = acc + f32(dot(x[gi], sv)) * scales[sbase + (k / 128u)];
  }
  let total = subgroupAdd(acc);
  if (lane == 0u) {
    if (n < p.N0) { out0[n] = total; }
    else if (n < p.N0 + p.N1) { out1[n - p.N0] = total; }
    else { out2[n - p.N0 - p.N1] = total; }
  }
}
`,matmul_split_sm:`// Small-batch (M = 2..9) subgroup split-K GEMV, fused qkv / gate-up. The speculative-decode
// verify pass computes M drafted rows in one forward; the scalar prefill kernels re-read the
// weights per output thread, so a k-row pass cost ~k GEMVs. Here each weight word is loaded
// ONCE per (column, k-chunk) and dotted with all M activation rows (activations are ~8 KB/row,
// cache-resident). Per row the loop stride and accumulation expression are IDENTICAL to
// matmul_split_sg, so each row's partials - and therefore the subgroupAdd result - match the
// M=1 decode path bit-for-bit.
enable subgroups;
override SG: u32 = 32u;
struct Params { K: u32, nb: u32, N0: u32, N1: u32, N2: u32, gridX: u32, M: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [M, K/4] row-major
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N0+N1+N2, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N0+N1+N2, nb]
@group(0) @binding(4) var<storage, read_write> out0: array<f32>; // [M, N0]
@group(0) @binding(5) var<storage, read_write> out1: array<f32>; // [M, N1]
@group(0) @binding(6) var<storage, read_write> out2: array<f32>; // [M, N2]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let Ntot = p.N0 + p.N1 + p.N2;
  let n = wg.y * p.gridX + wg.x;
  if (n >= Ntot) { return; } // uniform per workgroup: the whole subgroup exits together
  let wRow = n * (p.K / 32u);
  let sbase = n * p.nb;
  let Kvec = p.K / 4u;

  var acc: array<f32, 9>; // M <= 9
  for (var m = 0u; m < p.M; m = m + 1u) { acc[m] = 0.0; }
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let k = gi * 4u;
    let word = signbits[wRow + (k >> 5u)];
    let bits4 = (word >> (k & 31u)) & 0xfu;
    let sv = vec4<f32>(select(-1.0, 1.0, (bits4 & 1u) != 0u), select(-1.0, 1.0, (bits4 & 2u) != 0u),
                       select(-1.0, 1.0, (bits4 & 4u) != 0u), select(-1.0, 1.0, (bits4 & 8u) != 0u));
    let s = scales[sbase + (k / 128u)];
    for (var m = 0u; m < p.M; m = m + 1u) {
      acc[m] = acc[m] + dot(x[m * Kvec + gi], sv) * s;
    }
  }
  for (var m = 0u; m < p.M; m = m + 1u) { // p.M is uniform: collective calls stay uniform
    let total = subgroupAdd(acc[m]);
    if (lane == 0u) {
      if (n < p.N0) { out0[m * p.N0 + n] = total; }
      else if (n < p.N0 + p.N1) { out1[m * p.N1 + (n - p.N0)] = total; }
      else { out2[m * p.N2 + (n - p.N0 - p.N1)] = total; }
    }
  }
}
`,matmul_split_tiled:`// Tiled register-blocked binary GEMM to 3 outputs (qkv or gate/up), PREFILL (M>1), vec4 K-accum.
// Weights concatenated along N (N0|N1|N2); each output element routes individually to
// out0/out1/out2 by its global column, so N0/N1/N2 need no alignment. Same vec4 design as
// matmul_resid_tiled.
const BM: u32 = 64u;
const BN: u32 = 64u;
const BKV: u32 = 4u;          // BK / 4  (BK = 16)
struct Params { M: u32, K: u32, nb: u32, N0: u32, N1: u32, N2: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;  // [M, K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>; // [N0+N1+N2, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;   // [N0+N1+N2, nb]
@group(0) @binding(4) var<storage, read_write> out0: array<f32>;
@group(0) @binding(5) var<storage, read_write> out1: array<f32>;
@group(0) @binding(6) var<storage, read_write> out2: array<f32>;

var<workgroup> xs: array<vec4<f32>, 256>;
var<workgroup> ws: array<vec4<f32>, 256>;

@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let Ntot = p.N0 + p.N1 + p.N2;
  let tid = lid.x;
  let tileM = wg.y * BM;
  let tileN = wg.x * BN;
  let tr = (tid / 16u) * 4u;
  let tc = (tid % 16u) * 4u;
  let Kv = p.K / 4u;
  var acc: array<f32, 16>;
  for (var i = 0u; i < 16u; i = i + 1u) { acc[i] = 0.0; }

  let Ksteps = Kv / BKV;
  for (var ks = 0u; ks < Ksteps; ks = ks + 1u) {
    let k0v = ks * BKV;
    for (var e = tid; e < BM * BKV; e = e + 256u) {
      let m = e / BKV; let kv = e % BKV; let gm = tileM + m;
      xs[e] = select(vec4<f32>(0.0), x[gm * Kv + (k0v + kv)], gm < p.M);
    }
    for (var e = tid; e < BN * BKV; e = e + 256u) {
      let n = e / BKV; let kv = e % BKV; let gn = tileN + n; let k = (k0v + kv) * 4u;
      var wv = vec4<f32>(0.0);
      if (gn < Ntot) {
        let bits4 = (signbits[gn * (p.K / 32u) + (k >> 5u)] >> (k & 31u)) & 0xfu;
        let s = scales[gn * p.nb + (k / 128u)];
        wv = vec4<f32>(select(-s, s, (bits4 & 1u) != 0u), select(-s, s, (bits4 & 2u) != 0u),
                       select(-s, s, (bits4 & 4u) != 0u), select(-s, s, (bits4 & 8u) != 0u));
      }
      ws[e] = wv;
    }
    workgroupBarrier();
    for (var kv = 0u; kv < BKV; kv = kv + 1u) {
      var xr: array<vec4<f32>, 4>;
      for (var tm = 0u; tm < 4u; tm = tm + 1u) { xr[tm] = xs[(tr + tm) * BKV + kv]; }
      for (var tn = 0u; tn < 4u; tn = tn + 1u) {
        let w = ws[(tc + tn) * BKV + kv];
        for (var tm = 0u; tm < 4u; tm = tm + 1u) { acc[tm * 4u + tn] = acc[tm * 4u + tn] + dot(xr[tm], w); }
      }
    }
    workgroupBarrier();
  }

  for (var tm = 0u; tm < 4u; tm = tm + 1u) {
    let gm = tileM + tr + tm;
    if (gm >= p.M) { continue; }
    for (var tn = 0u; tn < 4u; tn = tn + 1u) {
      let gn = tileN + tc + tn;
      if (gn >= Ntot) { continue; }
      let v = acc[tm * 4u + tn];
      if (gn < p.N0) { out0[gm * p.N0 + gn] = v; }
      else if (gn < p.N0 + p.N1) { out1[gm * p.N1 + (gn - p.N0)] = v; }
      else { out2[gm * p.N2 + (gn - p.N0 - p.N1)] = v; }
    }
  }
}
`,matmul_split_wg:`// No-subgroup fallback: split-K GEMV for decode (M=1), workgroup-shared-memory reduction instead
// of subgroupAdd. One workgroup per output column; WG threads split K and tree-reduce via shared
// memory + barriers. ~WG-fold faster than one-thread-per-output (the v1 path). Routes qkv / gate-up.
override WG: u32 = 64u;
struct Params { K: u32, nb: u32, N0: u32, N1: u32, N2: u32, gridX: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [N0+N1+N2, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [N0+N1+N2, nb]
@group(0) @binding(4) var<storage, read_write> out0: array<f32>;
@group(0) @binding(5) var<storage, read_write> out1: array<f32>;
@group(0) @binding(6) var<storage, read_write> out2: array<f32>;
var<workgroup> sdata: array<f32, 256>;                          // >= max WG

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let Ntot = p.N0 + p.N1 + p.N2;
  let n = wg.y * p.gridX + wg.x;          // uniform across the workgroup -> early return is barrier-safe
  if (n >= Ntot) { return; }
  let tid = lid.x;
  let wRow = n * (p.K / 32u);
  let sbase = n * p.nb;
  let Kvec = p.K / 4u;
  var acc = 0.0;
  for (var gi = tid; gi < Kvec; gi = gi + WG) {
    let k = gi * 4u;
    let word = signbits[wRow + (k >> 5u)];
    let bits4 = (word >> (k & 31u)) & 0xfu;
    let sv = vec4<f32>(select(-1.0, 1.0, (bits4 & 1u) != 0u), select(-1.0, 1.0, (bits4 & 2u) != 0u),
                       select(-1.0, 1.0, (bits4 & 4u) != 0u), select(-1.0, 1.0, (bits4 & 8u) != 0u));
    acc = acc + dot(x[gi], sv) * scales[sbase + (k / 128u)];
  }
  sdata[tid] = acc;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) { sdata[tid] = sdata[tid] + sdata[tid + s]; }
    workgroupBarrier();
  }
  if (tid == 0u) {
    let total = sdata[0];
    if (n < p.N0) { out0[n] = total; }
    else if (n < p.N0 + p.N1) { out1[n - p.N0] = total; }
    else { out2[n - p.N0 - p.N1] = total; }
  }
}
`,matmul_swiglu_mr_sg:`// Multi-row fused gate/up GEMV + SwiGLU for decode (M=1). Each workgroup computes ROWS
// intermediate indices; per K-step it issues 2*ROWS independent weight loads (gate row n and up
// row F+n for each of the ROWS) before the dots, giving the bandwidth-bound decode GEMV more
// in-flight memory requests. One subgroup per workgroup; lanes split K; reduced with subgroupAdd.
enable subgroups;
override SG: u32 = 32u;
override ROWS: u32 = 4u;
struct Params { K: u32, nb: u32, F: u32, gridX: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;   // [K/4]
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [2F, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [2F, nb]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;   // [F]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let nBase = (wg.y * p.gridX + wg.x) * ROWS;
  let Kvec = p.K / 4u;
  let wStride = p.K / 32u;

  var g: array<f32, 8>;                            // ROWS <= 8
  var u: array<f32, 8>;
  for (var r = 0u; r < ROWS; r = r + 1u) { g[r] = 0.0; u[r] = 0.0; }
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let k = gi * 4u;
    let xv = x[gi];
    let widx = k >> 5u;
    let sh = k & 31u;
    let sc = k / 128u;
    for (var r = 0u; r < ROWS; r = r + 1u) {
      let n = nBase + r;
      if (n < p.F) {
        let gw = (signbits[n * wStride + widx] >> sh) & 0xfu;
        let gv = vec4<f32>(select(-1.0, 1.0, (gw & 1u) != 0u), select(-1.0, 1.0, (gw & 2u) != 0u),
                           select(-1.0, 1.0, (gw & 4u) != 0u), select(-1.0, 1.0, (gw & 8u) != 0u));
        g[r] = g[r] + dot(xv, gv) * scales[n * p.nb + sc];
        let uw = (signbits[(p.F + n) * wStride + widx] >> sh) & 0xfu;
        let uv = vec4<f32>(select(-1.0, 1.0, (uw & 1u) != 0u), select(-1.0, 1.0, (uw & 2u) != 0u),
                           select(-1.0, 1.0, (uw & 4u) != 0u), select(-1.0, 1.0, (uw & 8u) != 0u));
        u[r] = u[r] + dot(xv, uv) * scales[(p.F + n) * p.nb + sc];
      }
    }
  }
  for (var r = 0u; r < ROWS; r = r + 1u) {
    let n = nBase + r;
    let gt = subgroupAdd(g[r]);
    let ut = subgroupAdd(u[r]);
    if (lane == 0u && n < p.F) { y[n] = (gt / (1.0 + exp(-gt))) * ut; }
  }
}
`,matmul_swiglu_mr_sg_af16:`// f16-activation variant of matmul_swiglu_mr_sg (fused gate/up GEMV + SwiGLU, M=1). Reads the
// f16 activation x, dots in f16, accumulates each of gate/up in f32, applies SwiGLU in f32, and
// writes the intermediate as f16 (the input side of the f16 down_proj). Weights unchanged.
enable subgroups;
enable f16;
override SG: u32 = 32u;
override ROWS: u32 = 4u;
struct Params { K: u32, nb: u32, F: u32, gridX: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<vec4<f16>>;   // [K/4] f16 activations
@group(0) @binding(2) var<storage, read> signbits: array<u32>;  // [2F, K/32]
@group(0) @binding(3) var<storage, read> scales: array<f32>;    // [2F, nb]
@group(0) @binding(4) var<storage, read_write> y: array<f16>;   // [F] f16 intermediate

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let nBase = (wg.y * p.gridX + wg.x) * ROWS;
  let Kvec = p.K / 4u;
  let wStride = p.K / 32u;

  var g: array<f32, 8>;                            // ROWS <= 8
  var u: array<f32, 8>;
  for (var r = 0u; r < ROWS; r = r + 1u) { g[r] = 0.0; u[r] = 0.0; }
  for (var gi = lane; gi < Kvec; gi = gi + SG) {
    let k = gi * 4u;
    let xv = x[gi];
    let widx = k >> 5u;
    let sh = k & 31u;
    let sc = k / 128u;
    for (var r = 0u; r < ROWS; r = r + 1u) {
      let n = nBase + r;
      if (n < p.F) {
        let gw = (signbits[n * wStride + widx] >> sh) & 0xfu;
        let gv = vec4<f16>(select(-1.0h, 1.0h, (gw & 1u) != 0u), select(-1.0h, 1.0h, (gw & 2u) != 0u),
                           select(-1.0h, 1.0h, (gw & 4u) != 0u), select(-1.0h, 1.0h, (gw & 8u) != 0u));
        g[r] = g[r] + f32(dot(xv, gv)) * scales[n * p.nb + sc];
        let uw = (signbits[(p.F + n) * wStride + widx] >> sh) & 0xfu;
        let uv = vec4<f16>(select(-1.0h, 1.0h, (uw & 1u) != 0u), select(-1.0h, 1.0h, (uw & 2u) != 0u),
                           select(-1.0h, 1.0h, (uw & 4u) != 0u), select(-1.0h, 1.0h, (uw & 8u) != 0u));
        u[r] = u[r] + f32(dot(xv, uv)) * scales[(p.F + n) * p.nb + sc];
      }
    }
  }
  for (var r = 0u; r < ROWS; r = r + 1u) {
    let n = nBase + r;
    let gt = subgroupAdd(g[r]);
    let ut = subgroupAdd(u[r]);
    if (lane == 0u && n < p.F) { y[n] = f16((gt / (1.0 + exp(-gt))) * ut); }
  }
}
`,rmsnorm_rope_sg:`// Fused per-head RMSNorm + RoPE for decode (S=1). One subgroup (= one workgroup) per head row;
// lanes split head_dim, reduce sum-of-squares with subgroupAdd, then apply rope. rotate_half
// pairs (d, d+-D/2): with SG>=32 and D=128 a lane owns d in {lane, lane+32, lane+64, lane+96},
// so every (d, d+-64) pair is held by the same lane (no cross-lane reads for the rotate).
// outOff/outStride let the K result write straight into the KV cache at its position.
enable subgroups;
override SG: u32 = 32u;
struct Params { R: u32, D: u32, eps: f32, outOff: u32, outStride: u32, _p: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;        // [R, D]
@group(0) @binding(2) var<storage, read> gamma: array<f32>;    // [D]
@group(0) @binding(3) var<storage, read> cos: array<f32>;      // [D]
@group(0) @binding(4) var<storage, read> sin: array<f32>;      // [D]
@group(0) @binding(5) var<storage, read_write> y: array<f32>;  // [outOff + R*outStride]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let row = wg.x;
  if (row >= p.R) { return; }
  let base = row * p.D;
  var s = 0.0;
  for (var i = lane; i < p.D; i = i + SG) { let v = x[base + i]; s = s + v * v; }
  let inv = inverseSqrt(subgroupAdd(s) / f32(p.D) + p.eps);
  let half = p.D / 2u;
  let ob = p.outOff + row * p.outStride;
  for (var i = lane; i < p.D; i = i + SG) {
    let nd = x[base + i] * inv * gamma[i];
    var pd: u32; var sgn: f32;
    if (i < half) { pd = i + half; sgn = -1.0; } else { pd = i - half; sgn = 1.0; }
    let rot = sgn * (x[base + pd] * inv * gamma[pd]);
    y[ob + i] = nd * cos[i] + rot * sin[i];
  }
}
`,rmsnorm_rope_sg_kv16:`// rmsnorm_rope_sg writing into an f16-STORAGE KV cache (kvCache: 'f16'): used ONLY for the K
// projection on the fused decode path, where the normed+roped K is written straight into the
// cache. Keep in lockstep with rmsnorm_rope_sg.wgsl: the ONLY difference is y is array<f16>
// (one f32 -> f16 rounding at the write). The q call keeps the f32 kernel.
enable subgroups;
enable f16;
override SG: u32 = 32u;
struct Params { R: u32, D: u32, eps: f32, outOff: u32, outStride: u32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;        // [R, D]
@group(0) @binding(2) var<storage, read> gamma: array<f32>;    // [D]
@group(0) @binding(3) var<storage, read> cos: array<f32>;      // [D]
@group(0) @binding(4) var<storage, read> sin: array<f32>;      // [D]
@group(0) @binding(5) var<storage, read_write> y: array<f16>;  // [outOff + R*outStride]

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let row = wg.x;
  if (row >= p.R) { return; }
  let base = row * p.D;
  var s = 0.0;
  for (var i = lane; i < p.D; i = i + SG) { let v = x[base + i]; s = s + v * v; }
  let inv = inverseSqrt(subgroupAdd(s) / f32(p.D) + p.eps);
  let half = p.D / 2u;
  let ob = p.outOff + row * p.outStride;
  for (var i = lane; i < p.D; i = i + SG) {
    let nd = x[base + i] * inv * gamma[i];
    var pd: u32; var sgn: f32;
    if (i < half) { pd = i + half; sgn = -1.0; } else { pd = i - half; sgn = 1.0; }
    let rot = sgn * (x[base + pd] * inv * gamma[pd]);
    y[ob + i] = f16(nd * cos[i] + rot * sin[i]);
  }
}
`,rmsnorm_rope_sg_kv8:`// rmsnorm_rope_sg writing into the q8 cache (kvCache: 'q8'): used ONLY for the K projection on
// the fused decode path, where the normed+roped K quantizes straight into the cache. Keep the
// math in lockstep with rmsnorm_rope_sg.wgsl; the write side mirrors copy_kv8.wgsl (packed
// snorm8 words + one f32 scale per 32-element block). The q call keeps the f32 kernel.
enable subgroups;
override SG: u32 = 32u;
struct Params { R: u32, D: u32, eps: f32, outRow0: u32, _p0: u32, _p1: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;            // [R, D]
@group(0) @binding(2) var<storage, read> gamma: array<f32>;        // [D]
@group(0) @binding(3) var<storage, read> cos: array<f32>;          // [D]
@group(0) @binding(4) var<storage, read> sin: array<f32>;          // [D]
@group(0) @binding(5) var<storage, read_write> dstQ: array<u32>;   // packed 4 x snorm8 per word
@group(0) @binding(6) var<storage, read_write> dstS: array<f32>;   // [.., D/32] block scales

var<workgroup> wabs: array<f32, 32>; // per-word abs max (D <= 128 -> at most 32 words)
var<workgroup> wblk: array<f32, 4>;  // per-block scale (D/32 <= 4 blocks)

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let row = wg.x;                    // uniform: the barrier pattern below stays safe
  if (row >= p.R) { return; }
  let base = row * p.D;
  var s = 0.0;
  for (var i = lane; i < p.D; i = i + SG) { let v = x[base + i]; s = s + v * v; }
  let inv = inverseSqrt(subgroupAdd(s) / f32(p.D) + p.eps);
  let half = p.D / 2u;
  let W4 = p.D / 4u;

  var vals: array<vec4<f32>, 8>;     // words per lane: W4/SG <= 8 for SG >= 4
  var wi = 0u;
  for (var w = lane; w < W4; w = w + SG) {
    var vv = vec4<f32>(0.0);
    for (var e = 0u; e < 4u; e = e + 1u) {
      let i = w * 4u + e;
      let nd = x[base + i] * inv * gamma[i];
      var pd: u32; var sgn: f32;
      if (i < half) { pd = i + half; sgn = -1.0; } else { pd = i - half; sgn = 1.0; }
      let rot = sgn * (x[base + pd] * inv * gamma[pd]);
      vv[e] = nd * cos[i] + rot * sin[i];
    }
    vals[wi] = vv;
    wi = wi + 1u;
    wabs[w] = max(max(abs(vv.x), abs(vv.y)), max(abs(vv.z), abs(vv.w)));
  }
  workgroupBarrier();
  if (lane < p.D / 32u) {
    var m = 0.0;
    for (var i = 0u; i < 8u; i = i + 1u) { m = max(m, wabs[lane * 8u + i]); }
    let sc = max(m, 1e-30);
    wblk[lane] = sc;
    dstS[(p.outRow0 + row) * (p.D / 32u) + lane] = sc;
  }
  workgroupBarrier();
  wi = 0u;
  for (var w = lane; w < W4; w = w + SG) {
    dstQ[(p.outRow0 + row) * W4 + w] = pack4x8snorm(vals[wi] / wblk[w >> 3u]);
    wi = wi + 1u;
  }
}
`,rmsnorm_sg:`// RMSNorm, subgroup-parallel: one subgroup (= one workgroup) per row; lanes split D and
// reduce the sum-of-squares with subgroupAdd (register-only, no barriers/shared memory).
// Fixes the decode bottleneck where R=1 ran on a single thread. SG is set from the device's
// subgroup size at pipeline creation; requires workgroup_size == subgroup size.
enable subgroups;
override SG: u32 = 32u;
struct Params { R: u32, D: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let row = wg.x;
  if (row >= p.R) { return; }
  let base = row * p.D;
  var s = 0.0;
  for (var i = lane; i < p.D; i = i + SG) { let v = x[base + i]; s = s + v * v; }
  let total = subgroupAdd(s);                 // sum across the subgroup, broadcast to all lanes
  let inv = inverseSqrt(total / f32(p.D) + p.eps);
  for (var i = lane; i < p.D; i = i + SG) { y[base + i] = x[base + i] * inv * gamma[i]; }
}
`,rmsnorm_sg_af16:`// RMSNorm (subgroup) that writes the normalized activation as f16 - the input side of the
// f16-activation decode matmuls (activation: 'f16'). Reads the f32 residual stream; the
// sum-of-squares reduction stays f32 (accuracy); only the stored output is rounded to f16.
// Identical reduction to rmsnorm_sg, so it is bit-comparable up to the final f16 rounding.
enable subgroups;
enable f16;
override SG: u32 = 32u;
struct Params { R: u32, D: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> gamma: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f16>;

@compute @workgroup_size(SG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(subgroup_invocation_id) lane: u32) {
  let row = wg.x;
  if (row >= p.R) { return; }
  let base = row * p.D;
  var s = 0.0;
  for (var i = lane; i < p.D; i = i + SG) { let v = x[base + i]; s = s + v * v; }
  let total = subgroupAdd(s);
  let inv = inverseSqrt(total / f32(p.D) + p.eps);
  for (var i = lane; i < p.D; i = i + SG) { y[base + i] = f16(x[base + i] * inv * gamma[i]); }
}
`,rmsnorm_wg:`// RMSNorm, no-subgroup fallback: one workgroup per row; threads split D and tree-reduce the
// sum of squares via shared memory. Replaces the one-thread-per-row kernel on this path: at
// decode (R=1) that kernel walked 2xD elements serially on a single thread, latency-bound,
// and it ran twice per layer - the dominant cost of the whole fallback decode step.
// Mirrors rmsnorm_sg exactly, with subgroupAdd swapped for the shared-memory reduction.
override WG: u32 = 64u;
struct Params { R: u32, D: u32, eps: f32, _pad: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;       // [R, D]
@group(0) @binding(2) var<storage, read> gamma: array<f32>;   // [D]
@group(0) @binding(3) var<storage, read_write> y: array<f32>; // [R, D]
var<workgroup> sdata: array<f32, 256>;                        // >= max WG

@compute @workgroup_size(WG)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let row = wg.x;                        // uniform across the workgroup -> early return is barrier-safe
  if (row >= p.R) { return; }
  let tid = lid.x;
  let base = row * p.D;
  var s = 0.0;
  for (var i = tid; i < p.D; i = i + WG) { let v = x[base + i]; s = s + v * v; }
  sdata[tid] = s;
  workgroupBarrier();
  for (var st = WG / 2u; st > 0u; st = st >> 1u) {
    if (tid < st) { sdata[tid] = sdata[tid] + sdata[tid + st]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(sdata[0] / f32(p.D) + p.eps);
  for (var i = tid; i < p.D; i = i + WG) { y[base + i] = x[base + i] * inv * gamma[i]; }
}
`,rope:`// RoPE (rotate_half) with precomputed full cos/sin [S, D]. x is [S, H, D]. One invocation per element.
struct Params { S: u32, H: u32, D: u32, _pad: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;       // [S, H, D]
@group(0) @binding(2) var<storage, read> cos: array<f32>;     // [S, D]
@group(0) @binding(3) var<storage, read> sin: array<f32>;     // [S, D]
@group(0) @binding(4) var<storage, read_write> y: array<f32>; // [S, H, D]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (idx >= p.S * p.H * p.D) { return; }
  let d = idx % p.D;
  let sh = idx / p.D;
  let s = sh / p.H;
  let half = p.D / 2u;
  let row = sh * p.D;  // s*H*D + h*D
  var rot: f32;
  if (d < half) {
    rot = -x[row + d + half];
  } else {
    rot = x[row + d - half];
  }
  y[idx] = x[idx] * cos[s * p.D + d] + rot * sin[s * p.D + d];
}
`,rope_partial:`// Partial RoPE: rotate only the first ROT dims of each head (rotate_half within [0,ROT)); the
// remaining head_dim-ROT dims pass through unrotated. cos/sin are [S, ROT]. x/y are [S, H, D].
// Matches tools/qwen35_numpy._rope_partial (Qwen3.5 full-attention layers, partial_rotary_factor).
struct Params { S: u32, H: u32, D: u32, ROT: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> x: array<f32>;        // [S, H, D]
@group(0) @binding(2) var<storage, read> cosb: array<f32>;     // [S, ROT]
@group(0) @binding(3) var<storage, read> sinb: array<f32>;     // [S, ROT]
@group(0) @binding(4) var<storage, read_write> y: array<f32>;  // [S, H, D]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let idx = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (idx >= p.S * p.H * p.D) { return; }
  let d = idx % p.D;
  if (d >= p.ROT) { y[idx] = x[idx]; return; }   // passthrough tail
  let sh = idx / p.D;
  let s = sh / p.H;
  let half = p.ROT / 2u;
  var rot: f32;
  if (d < half) { rot = -x[idx + half]; } else { rot = x[idx - half]; }
  y[idx] = x[idx] * cosb[s * p.ROT + d] + rot * sinb[s * p.ROT + d];
}
`,sampler_penalty:`// GPU logits pre-filter for sampling: applies repetition_penalty + presence_penalty, then
// no_repeat_ngram bans, in place on the full vocab logit buffer, so only a tiny top-K candidate set
// has to be read back (not all ~151k logits). rep_penalty matches transformers.js over the DEDUPED
// prompt+generated id set (logit<0 ? *penalty : /penalty); presence_penalty then SUBTRACTS a flat
// amount from every seen token (the additive anti-repetition knob the Qwen3.5 family recommends,
// applied after the multiplicative rep_penalty like vLLM); then ngram-banned next-tokens go to
// -Infinity. Both id lists are computed on the CPU each step (exact, since at syncN=1 the full
// history is known) and uploaded. presence is 0 unless requested, so \`v*penalty - 0.0 == v*penalty\`
// keeps the rep-penalty-only path bit-identical. Temperature is NOT applied here: top-k is invariant
// under the monotonic divide, so temperature is applied on the CPU to just the K candidate values
// before softmax (bit-identical, one less pass). Single workgroup, no subgroup ops -> all devices.
// The storageBarrier guarantees every penalty write lands before any ban write, so a token that is
// both repeated and ngram-banned ends at -inf (ban wins, matching the reference order penalties -> ngram).
override WG: u32 = 256u;
// negInf carries the -Infinity bit pattern (0xff800000) from the host: bitcasting it at RUNTIME yields
// -inf, whereas bitcast<f32>(0xff800000u) is a const-expression evaluating to inf, which is a WGSL
// shader-creation error. (Runtime inf is fine; only const/override inf/nan is rejected.)
struct Params { affectedLen: u32, banLen: u32, penalty: f32, negInf: u32, presence: f32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> affectedIds: array<u32>;   // deduped prompt+generated ids
@group(0) @binding(2) var<storage, read> banIds: array<u32>;        // ngram-banned next-token ids
@group(0) @binding(3) var<storage, read_write> logits: array<f32>;  // [vocab], modified in place

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  for (var i = tid; i < p.affectedLen; i = i + WG) {
    let t = affectedIds[i];
    let v = logits[t];
    let rp = select(v / p.penalty, v * p.penalty, v < 0.0);   // repetition_penalty (multiplicative)
    logits[t] = rp - p.presence;                              // presence_penalty (subtractive; 0 = no-op)
  }
  storageBarrier();                                  // all penalty writes before any ban write
  for (var i = tid; i < p.banLen; i = i + WG) {
    logits[banIds[i]] = bitcast<f32>(p.negInf);      // -Infinity (runtime bitcast)
  }
}
`,sampler_sigma:`// Mean/variance statistics of the (penalty-filtered) logits for the top-n-sigma warper
// (arXiv 2411.07641): the CPU keeps candidates with logit >= max - n * sigma, where sigma is the
// standard deviation of the FULL logit vector (the paper's statistic - a top-K-only estimate is
// biased). Runs AFTER sampler_penalty and BEFORE the argmax_masked rounds (those mask winners in
// place, which would corrupt the moments). Banned entries (-inf sentinel) are excluded; numerical
// stability comes from centering on the global max before accumulating (logits are O(10), so
// sum-of-squares around the max stays well inside f32). Three f32s are read back:
// out = [sum(x - max), sum((x - max)^2), count] -> CPU: var = q/c - (s/c)^2.
override WG: u32 = 256u;
struct Params { N: u32, _0: u32, _1: u32, _2: u32 };

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> logits: array<f32>;
@group(0) @binding(2) var<storage, read_write> outStats: array<f32>; // [sum, sumsq, count] centered on max

const NEG_SENTINEL: f32 = -3.0e38; // below any real logit; banned entries sit at f32 -inf

var<workgroup> sa: array<f32, 256>;
var<workgroup> sb: array<f32, 256>;
var<workgroup> sc: array<f32, 256>;

@compute @workgroup_size(WG)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  var m = -3.4e38;
  for (var i = tid; i < p.N; i = i + WG) {
    let v = logits[i];
    if (v > NEG_SENTINEL && v > m) { m = v; }
  }
  sa[tid] = m;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s && sa[tid + s] > sa[tid]) { sa[tid] = sa[tid + s]; }
    workgroupBarrier();
  }
  let gmax = sa[0];
  workgroupBarrier();
  var acc = 0.0;
  var accq = 0.0;
  var cnt = 0.0;
  for (var i = tid; i < p.N; i = i + WG) {
    let v = logits[i];
    if (v > NEG_SENTINEL) {
      let d = v - gmax;
      acc = acc + d;
      accq = accq + d * d;
      cnt = cnt + 1.0;
    }
  }
  sa[tid] = acc;
  sb[tid] = accq;
  sc[tid] = cnt;
  workgroupBarrier();
  for (var s = WG / 2u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      sa[tid] = sa[tid] + sa[tid + s];
      sb[tid] = sb[tid] + sb[tid + s];
      sc[tid] = sc[tid] + sc[tid + s];
    }
    workgroupBarrier();
  }
  if (tid == 0u) {
    outStats[0] = sa[0];
    outStats[1] = sb[0];
    outStats[2] = sc[0];
  }
}
`,slice_cols:`// Extract a contiguous column range [off, off+w) from each row of a [rows, stride] buffer into a
// packed [rows, w] buffer. Splits the DeltaNet conv output (q|k|v concatenated per token) into the
// separate q/k/v activation buffers the scan reads.
struct Params { rows: u32, w: u32, stride: u32, off: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;      // [rows, stride]
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;// [rows, w]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (i >= p.rows * p.w) { return; }
  let r = i / p.w;
  let c = i % p.w;
  dst[i] = src[r * p.stride + p.off + c];
}
`,split_head:`// De-interleave a per-head doubled projection [S, H, 2*Dh] into [S, H, Dh], taking the half at
// \`off\` (0 = query, Dh = gate). The Qwen3.5 gated-attention q_proj packs query and output-gate
// interleaved per head; this pulls one out into a packed buffer.
struct Params { S: u32, H: u32, Dh: u32, off: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> src: array<f32>;      // [S, H, 2*Dh]
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;// [S, H, Dh]

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (i >= p.S * p.H * p.Dh) { return; }
  let d = i % p.Dh;
  let sh = i / p.Dh;          // s*H + h
  dst[i] = src[sh * (2u * p.Dh) + p.off + d];
}
`,swiglu:`// SwiGLU gate: y[i] = silu(gate[i]) * up[i], silu(g) = g * sigmoid(g). One invocation per element.
struct Params { n: u32, _p0: u32, _p1: u32, _p2: u32 };
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var<storage, read> gate: array<f32>;
@group(0) @binding(2) var<storage, read> up: array<f32>;
@group(0) @binding(3) var<storage, read_write> y: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>, @builtin(num_workgroups) nwg: vec3<u32>) {
  let i = (wid.y * nwg.x + wid.x) * 64u + lid.x;
  if (i >= p.n) { return; }
  let g = gate[i];
  y[i] = (g / (1.0 + exp(-g))) * up[i];
}
`};var Vt=class extends Error{name="WebGPUUnavailableError";constructor(k){super(k)}},Lr=class extends Error{name="GpuOutOfMemoryError";constructor(k){super(k)}};function ya(k,b,d){if(d<=0)return[];for(let x=Math.min(b,k.length-1);x>=2;x--){const K=k.length-x;e:for(let X=K-1;X>=0;X--){for(let V=0;V<x;V++)if(k[X+V]!==k[K+V])continue e;const N=X+x;return k.slice(N,Math.min(N+d,k.length))}}return[]}function _a(k,b,d){return b<=0?!1:k/b>=(d?1.5:2)}var Rt=class{mt=new Uint32Array(624);idx=625;constructor(k){this.seed(k)}seed(k){if(k==null){const N=new Uint32Array(1);crypto.getRandomValues(N),k=N[0]}const b=this.mt,d=(N,V)=>Math.imul(N,V)>>>0,x=[];for(let N=k||0;N>0;N=Math.floor(N/4294967296))x.push(N&4294967295);x.length||x.push(0),b[0]=19650218;for(let N=1;N<624;++N)b[N]=d(1812433253,b[N-1]^b[N-1]>>>30)+N>>>0;let K=1,X=0;for(let N=Math.max(624,x.length);N>0;--N,++K,++X)K>=624&&(b[0]=b[623],K=1),X>=x.length&&(X=0),b[K]=(b[K]^d(b[K-1]^b[K-1]>>>30,1664525))+x[X]+X>>>0;for(let N=623;N>0;--N,++K)K>=624&&(b[0]=b[623],K=1),b[K]=(b[K]^d(b[K-1]^b[K-1]>>>30,1566083941))-K>>>0;b[0]=2147483648,this.idx=624}int32(){const k=this.mt;if(this.idx>=624){for(let d=0;d<624;++d){const x=k[d]&2147483648|k[(d+1)%624]&2147483647;k[d]=(k[(d+397)%624]^x>>>1^(x&1?2567483615:0))>>>0}this.idx=0}let b=k[this.idx++];return b^=b>>>11,b^=b<<7&2636928640,b^=b<<15&4022730752,b^=b>>>18,b>>>0}random(){return((this.int32()>>>5)*67108864+(this.int32()>>>6))/9007199254740992}};function it(k){return Uint32Array.from(new Set(k))}function st(k,b){if(k.length+1<b)return[];const d=new Map;for(let K=0;K<k.length+1-b;++K){const X=[];for(let he=0;he<b;++he)X.push(k[K+he]);const N=JSON.stringify(X.slice(0,b-1)),V=d.get(N)??[];V.push(X[b-1]),d.set(N,V)}const x=k.slice(k.length+1-b,k.length);return d.get(JSON.stringify(x))??[]}function ut(k,b,d,x){const K=d.length,X=x.range>0?Math.max(0,K-x.range):0,N=x.allowedLength+32,V=Array.from(b),he=Array.from(k);if(K>X&&x.multiplier>0)for(let c=0;c<he.length;c++){const S=he[c];if(x.breakers.has(S))continue;let T=0;for(let Y=X;Y<K;Y++){if(d[Y]!==S)continue;let te=0;for(;te<N&&Y-1-te>=X&&!x.breakers.has(d[Y-1-te])&&!x.breakers.has(d[K-1-te])&&d[Y-1-te]===d[K-1-te];)te++;if(te>T&&(T=te),T>=N)break}T>=x.allowedLength&&(V[c]-=x.multiplier*Math.pow(x.base,T-x.allowedLength))}const se=Array.from(he.keys()).sort((c,S)=>V[S]-V[c]||c-S);return{ids:se.map(c=>he[c]),vals:se.map(c=>V[c])}}function xa(k){let b=k[0];for(let K=1;K<k.length;++K)k[K]>b&&(b=k[K]);const d=Array.from(k,K=>Math.exp(K-b));let x=0;for(const K of d)x+=K;return d.map(K=>K/x)}function jr(k,b,d,x,K=1,X=0){const N=b.length,V=new Float32Array(N);for(let T=0;T<N;++T)V[T]=b[T]/d;const he=xa(V);let se=N;if(X>0){const T=X*he[0];let Y=1;for(;Y<N&&he[Y]>=T;)Y++;Y<se&&(se=Y)}if(K<1){let T=0,Y=0;for(;Y<N&&(T+=he[Y],Y++,!(T>=K)););Y<se&&(se=Y)}se<1&&(se=1);let c=0;for(let T=0;T<se;++T)c+=he[T];let S=x.random()*c;for(let T=0;T<se;++T)if(S-=he[T],S<0)return k[T];return k[se-1]}const Ka={FLOAT:Float32Array,UINT8:Uint8Array,FLOAT16:Uint16Array},Na=["matmul_split","matmul_resid","matmul_q2","rope","swiglu","copy"],qa=2048,lt=256,zn=16,At=new ArrayBuffer(64),Ht=new DataView(At),Da=new Uint8Array(At);function Et(k){for(let b=0;b<k.length;b++){const d=k[b];d[0]==="f"?Ht.setFloat32(b*4,d[1],!0):Ht.setUint32(b*4,d[1]>>>0,!0)}return Da.subarray(0,Math.ceil(k.length/4)*16)}const Ma=(k,b)=>{for(let d=0;d<b.length;d++)if(k[d]!==b[d])return!1;return!0};function Sa(k,b,d=k.head_dim){const x=d/2,K=k.rope,X=K.rope_theta,N=K.rope_type==="yarn"?K.factor??1:1,V=new Float64Array(x),he=K.original_max_position_embeddings??0,se=N===1?0:Math.max(0,Math.floor(d*Math.log(he/(64*Math.PI))/(2*Math.log(X)))),c=N===1?0:Math.min(x-1,Math.ceil(d*Math.log(he/(2*Math.PI))/(2*Math.log(X))));for(let te=0;te<x;te++){const an=X**(2*te/d);if(N===1){V[te]=1/an;continue}const Me=Math.min(1,Math.max(0,(te-se)/(c-se)));V[te]=1/(N*an)*Me+1/an*(1-Me)}const S=N===1?1:Math.fround(.1*Math.log(N)+1),T=new Float32Array(b*x),Y=new Float32Array(b*x);for(let te=0;te<b;te++)for(let an=0;an<x;an++){const Me=te*V[an];T[te*x+an]=Math.fround(Math.cos(Me)*S),Y[te*x+an]=Math.fround(Math.sin(Me)*S)}return[T,Y]}async function Ga(k){const b={};try{return await Ba(k,b)}catch(d){throw b.device?.destroy(),d}}async function Ba(k,b){const d=typeof k=="string"?{modelUrl:k}:k,x=d.modelUrl?d.modelUrl.replace(/\/$/,""):null;if(!x&&!d.manifestUrl&&!d.manifest)throw new Error("createEngine: provide modelUrl, manifestUrl, or an in-memory manifest");if(d.manifest&&!x&&!d.dataUrl)throw new Error("createEngine: an in-memory manifest needs dataUrl (or modelUrl) for the weights file");const K=d.powerPreference??"high-performance",X=d.fetchJson??(async e=>{const n=await fetch(e);if(!n.ok)throw new Error(`bitgpu: fetch ${e} failed: HTTP ${n.status}`);if((n.headers.get("content-type")??"").includes("text/html"))throw new Error(`bitgpu: ${e} returned HTML, not JSON (a SPA fallback is probably serving index.html for missing model files)`);return n.json()}),N=d.fetchArrayBuffer??(async e=>{const n=await fetch(e);if(!n.ok)throw new Error(`bitgpu: fetch ${e} failed: HTTP ${n.status}`);const t=Number(n.headers.get("content-length")??0);if(!n.body||!t)return n.arrayBuffer();const r=n.body.getReader(),a=[];let o=0;for(;;){const{done:l,value:f}=await r.read();if(l)break;a.push(f),o+=f.byteLength,d.onProgress?.({phase:"weights",loaded:o,total:t})}const u=new Uint8Array(o);let i=0;for(const l of a)u.set(l,i),i+=l.byteLength;return u.buffer});if(typeof navigator>"u"||!navigator.gpu)throw new Vt("WebGPU is not available (no navigator.gpu). Use a WebGPU-capable browser over a secure context.");d.onProgress?.({phase:"manifest"});const V=d.manifest??await X(d.manifestUrl??`${x}/manifest.json`);d.onProgress?.({phase:"weights"});const he=d.dataUrl??`${x}/${V.data_file}`;let se;d.aux?se=d.aux instanceof Uint8Array?new Uint8Array(d.aux).buffer:d.aux:se=await N(d.auxUrl??`${x}/${V.aux_file}`);const c=V.arch,S=V.tensors,T=`layers.${c.layers}.final_norm_layernorm`;if(c.act!=="silu")throw new Error(`bitgpu: unsupported activation '${c.act}' (kernels implement silu/SwiGLU)`);if(c.head_dim>(c.hybrid?256:128))throw new Error(`bitgpu: unsupported head_dim ${c.head_dim}`);const Y=c.hybrid?.rotary_dim??c.head_dim;if(c.heads%c.kv_heads!==0)throw new Error(`bitgpu: heads ${c.heads} not divisible by kv_heads ${c.kv_heads} (GQA kernels assume an integer group size)`);if(!S[T])throw new Error(`bitgpu: manifest is missing the final norm tensor '${T}'`);if(V.version!==void 0&&V.version!==1&&V.version!==2)throw new Error(`bitgpu: unsupported manifest version ${V.version} (this engine reads versions 1 and 2)`);if(!S.cos_cache!=!S.sin_cache)throw new Error("bitgpu: manifest has only one of cos_cache/sin_cache");if(!S.cos_cache&&!(c.rope&&c.rope.rope_theta))throw new Error("bitgpu: manifest has neither baked cos_cache/sin_cache RoPE tensors nor arch.rope parameters");for(const[e,n]of Object.entries(S)){if(n.block!==void 0&&n.block!==128)throw new Error(`bitgpu: tensor ${e} has block ${n.block} (kernels assume 128)`);if(n.container===void 0)continue;if(n.container!=="q1_0")throw new Error(`bitgpu: tensor ${e} has unknown container '${n.container}'`);const t=n.N??n.rows,r=n.K??n.cols;if(!t||!r||r%128!==0)throw new Error(`bitgpu: tensor ${e}: q1_0 container needs N/K (or rows/cols) with K a multiple of 128`);const a=n.weight;if(!a||a.src!=="data"||a.len!==t*(r/128)*18)throw new Error(`bitgpu: tensor ${e}: q1_0 region is ${a?.len} bytes in '${a?.src}', expected ${t*(r/128)*18} in the data file`);n.q1_0=a,n.weight={dtype:"UINT8",src:a.src,off:a.off,len:t*(r/8)},n.scales={dtype:"FLOAT",src:a.src,off:a.off,len:t*(r/128)*4},n.zp=void 0}const te=e=>{if(e.src!=="aux")throw new Error("bitgpu: internal - readRef reads aux-file refs; data-file tensors stream through routes");if(e.off+e.len>se.byteLength)throw new Error(`bitgpu: tensor range ${e.off}+${e.len} exceeds the aux file (${se.byteLength} bytes); the download is truncated or the manifest does not match it`);const n=Ka[e.dtype];if(n===Uint8Array)return new Uint8Array(se,e.off,e.len);const t=n.BYTES_PER_ELEMENT;return e.off%t===0?new n(se,e.off,e.len/t):new n(se.slice(e.off,e.off+e.len))},an=e=>te(e),Me=await navigator.gpu.requestAdapter({powerPreference:K});if(!Me)throw new Vt("No suitable WebGPU adapter was found.");const Wt=Me.features.has("subgroups"),Cn=Me.info??{},Ve=Cn.subgroupMaxSize??32,Lt=Cn.subgroupMinSize??Ve,jt=d.forceNoSubgroups??!1,Tt=Math.min(256,Math.max(32,1<<Math.round(Math.log2(d.noSubgroupWorkgroupSize??64)))),Ut=d.prefillTiling==="never",Ct=d.prefillTiling==="always",ct=e=>Ct||!Ut&&e>=64,br=Math.max(1,d.syncSteps??4),E=Math.max(1,d.maxSeqLen??qa),xe=Wt&&Lt===Ve&&(Ve===16||Ve===32||Ve===64)&&c.head_dim%Ve===0&&!jt,on=d.kvCache==="f16"&&!c.hybrid&&Me.features.has("shader-f16"),be=d.kvCache==="q8",ve=d.overflow==="sinks",Xe=ve?Math.max(1,Math.floor(d.sinkTokens??4)):0;if(ve&&c.hybrid)throw new Error("bitgpu: overflow 'sinks' is not yet supported for the qwen3_5 hybrid backbone (the full-attention read path has no sink/roll K-rotation, and windowing only the full layers while the linear layers keep full-history state is unvalidated)");if(ve&&E<Xe+64)throw new Error(`bitgpu: overflow 'sinks' needs maxSeqLen >= sinkTokens + 64 (got ${E} with ${Xe} sinks)`);const qn=on?2:be?1:4,er=d.activation==="f16"&&xe&&Me.features.has("shader-f16"),wr=[];xe&&wr.push("subgroups"),(on||er)&&wr.push("shader-f16");const vr=Me.features.has("timestamp-query");vr&&wr.push("timestamp-query");const $t=134217728,It=268435456;let rn=0,dt=0;const Vn=e=>{const n=e+3&-4;rn=Math.max(rn,n),dt+=n};for(const e of Object.values(S))e.kind==="q2"?(Vn(e.weight.len*2),Vn(e.scales.len)):e.kind==="f32"&&e.weight&&Vn(e.weight.len);const pt=(e,n)=>e.reduce((t,r)=>t+S[r][n].len,0);for(let e=0;e<c.layers;e++){let n;c.hybrid?n=[...c.hybrid.layer_types[e]==="full"?["attn.q_proj","attn.k_proj","attn.v_proj","attn.o_proj"]:["linear.in_qkv","linear.z","linear.a","linear.b","linear.out_proj"],"mlp.gate_proj","mlp.up_proj","mlp.down_proj"].map(t=>[`layers.${e}.${t}`]):n=[[`layers.${e}.attn.q_proj`,`layers.${e}.attn.k_proj`,`layers.${e}.attn.v_proj`],[`layers.${e}.mlp.gate_proj`,`layers.${e}.mlp.up_proj`],[`layers.${e}.attn.o_proj`],[`layers.${e}.mlp.down_proj`]];for(const t of n)Vn(pt(t,"weight")),Vn(pt(t,"scales"))}const Tr=S.embed_tokens.zp?.len??S.embed_tokens.rows*(S.embed_tokens.cols/128)/2;for(const e of[S.embed_tokens.weight,S.embed_tokens.scales,V.luts.tgt4])Vn(e.len);if(Vn(Tr),be&&c.head_dim%32!==0)throw new Error(`bitgpu: kvCache 'q8' needs head_dim divisible by 32 (got ${c.head_dim}); use 'f16' or 'f32' for this model`);const Ot=E*c.kv_heads*c.head_dim*qn;rn=Math.max(rn,Ot,lt*Math.max(c.heads*c.head_dim,c.intermediate)*4,32*c.vocab*4);const kr={};if(rn>$t){if(rn>Me.limits.maxStorageBufferBindingSize)throw new Lr(`this model needs a ${Math.ceil(rn/1048576)} MiB storage binding but the adapter's maxStorageBufferBindingSize is ${Math.floor(Me.limits.maxStorageBufferBindingSize/1048576)} MiB`);kr.maxStorageBufferBindingSize=rn}if(rn>It){if(rn>Me.limits.maxBufferSize)throw new Lr(`this model needs a ${Math.ceil(rn/1048576)} MiB buffer but the adapter's maxBufferSize is ${Math.floor(Me.limits.maxBufferSize/1048576)} MiB`);kr.maxBufferSize=rn}const s=await Me.requestDevice({requiredFeatures:wr,requiredLimits:Object.keys(kr).length?kr:void 0});b.device=s;const Ft=s.lost.then(e=>{const n={reason:String(e.reason??"unknown"),message:e.message};return n.reason!=="destroyed"&&d.onDeviceLost?.(n),n});s.addEventListener("uncapturederror",e=>{console.error(`[bitgpu] uncaptured WebGPU error: ${e.error.message}`)}),d.onProgress?.({phase:"pipelines"});const yr={},Qt=async(e,n)=>{const t=ka[e];if(t===void 0)throw new Error(`shader not found: ${e}`);const r=s.createShaderModule({code:t,label:e}),a=(await r.getCompilationInfo()).messages.find(o=>o.type==="error");if(a)throw new Error(`WGSL compile error in ${e} (L${a.lineNum}:${a.linePos}): ${a.message}`);yr[e]=await s.createComputePipelineAsync({layout:"auto",compute:{module:r,entryPoint:"main",constants:n}})},_r=4,ke=[...Na.map(e=>[e]),["matmul_split_tiled"],["matmul_resid_tiled"],["argmax"],["embed_gather"],["embed_gather_batch"],["sampler_penalty"],["argmax_masked"],["logsumexp"],["sampler_sigma"]];if(xe){for(const e of["rmsnorm_sg","attention_sg","matmul_split_sg","matmul_q2_sg","rmsnorm_rope_sg"])ke.push([e,{SG:Ve}]);for(const e of["matmul_split_sm","matmul_resid_sm","matmul_q2_sm"])ke.push([e,{SG:Ve}]);for(const e of["matmul_resid_mr_sg","matmul_swiglu_mr_sg"])ke.push([e,{SG:Ve,ROWS:_r}])}else{for(const e of["matmul_split_wg","matmul_resid_wg","matmul_q2_wg","rmsnorm_wg"])ke.push([e,{WG:Tt}]);ke.push(["attention_wg"])}if(on)if(ke.push(["copy_kv16"]),xe)for(const e of["attention_sg_kv16","rmsnorm_rope_sg_kv16"])ke.push([e,{SG:Ve}]);else ke.push(["attention_wg_kv16"]);if(be)if(ke.push(["copy_kv8"]),xe)for(const e of["attention_sg_kv8","rmsnorm_rope_sg_kv8"])ke.push([e,{SG:Ve}]);else ke.push(["attention_wg_kv8"]);if(er){for(const e of["rmsnorm_sg_af16","matmul_split_sg_af16"])ke.push([e,{SG:Ve}]);for(const e of["matmul_swiglu_mr_sg_af16","matmul_resid_mr_sg_af16"])ke.push([e,{SG:Ve,ROWS:_r}])}const xr=xe&&Ve<=c.head_dim/2;if(ve){const e=on?"attention_sg_kv16_roll":be?"attention_sg_kv8_roll":"attention_sg_roll",n=on?"attention_wg_kv16_roll":be?"attention_wg_kv8_roll":"attention_wg_roll";xr?ke.push([e,{SG:Ve}]):ke.push([n])}const ft=ve?on?xr?"attention_sg_kv16_roll":"attention_wg_kv16_roll":be?xr?"attention_sg_kv8_roll":"attention_wg_kv8_roll":xr?"attention_sg_roll":"attention_wg_roll":on?xe?"attention_sg_kv16":"attention_wg_kv16":be?xe?"attention_sg_kv8":"attention_wg_kv8":xe?"attention_sg":"attention_wg",Xt=on?"rmsnorm_rope_sg_kv16":"rmsnorm_rope_sg",Yt=on?"copy_kv16":"copy";if(c.hybrid){for(const e of["conv1d_causal","deltanet_gbeta","rope_partial","slice_cols","split_head","gate_sigmoid"])ke.push([e]);ke.push(["deltanet_recur",{WGV:c.hybrid.linear_head_dim}]),ke.push(["deltanet_norm_gate",{WG:64}]),ke.push(["attention_online",{WGD:c.head_dim}]),ke.push([be?"attention_online_cache_kv8":"attention_online_cache",{WGD:c.head_dim}])}await Promise.all(ke.map(([e,n])=>Qt(e,n)));const y=GPUBufferUsage.STORAGE,q=GPUBufferUsage.COPY_DST,P=GPUBufferUsage.COPY_SRC,gt=GPUBufferUsage.UNIFORM;let ne=null;const Ie=()=>{if(ne){for(const e of ne)e.destroy();ne=[],Dn=null}};let Dn=null,Kr=null;const mt=e=>{if(!Dn)return;const n=Dn.get(e.size);n?n.push(e):Dn.set(e.size,[e])},Nr=(e,n=y|q)=>{const t=s.createBuffer({size:e.byteLength,usage:n});return s.queue.writeBuffer(t,0,e),ne?.push(t),t},ht={};let sn=null,Rn=0,qr=0,Ur=0,bt=0;const Mn=(e,n=0,t=0)=>{sn=e?ht[e]??={buf:[],disp:[]}:null,Ur=n,bt=t,Rn=0,qr=0},nr=()=>{for(const e of Object.values(ht))for(const n of e.disp)n.bg=null,n.last=null},v=e=>{if(!sn){const r=Dn?.get(e*4)?.pop();if(r)return Kr?.push(r),r;const a=s.createBuffer({size:e*4,usage:y|P|q});return ne?.push(a),Kr?.push(a),a}const n=Ur>0?e/Ur*bt:e;let t=sn.buf[Rn];return(!t||t.size!==n*4)&&(t=s.createBuffer({size:n*4,usage:y|P|q}),sn.buf[Rn]=t),Rn++,t},rr=e=>{const n=e*2;if(!sn){const r=s.createBuffer({size:n,usage:y|P|q});return ne?.push(r),r}let t=sn.buf[Rn];return(!t||t.size!==n)&&(t=s.createBuffer({size:n,usage:y|P|q}),sn.buf[Rn]=t),Rn++,t},hn=s.createBuffer({size:16,usage:y}),_n=s.createBuffer({size:16,usage:y});let Ke=[],Ne=0;const Zt=()=>{Ke=[],Ne=0};if(V.luts.tgt2.src!=="aux")throw new Error("bitgpu: luts.tgt2 must live in the aux file (the streaming loader needs it before the weights arrive)");const Cr=an(V.luts.tgt2),wt=new Uint8Array(256);for(let e=0;e<256;e++){let n=0;for(let t=0;t<8;t++)n|=((Cr[2*e+(t>>2)]>>2*(t&3)&3)>>1&1)<<t;wt[e]=n}const Dr=[],xn=e=>s.createBuffer({size:e+3&-4,usage:y|q}),Kn=(e,n)=>{let t=0,r=new Uint8Array(0);return{push(a){let o=a;r.length&&(o=new Uint8Array(r.length+a.length),o.set(r),o.set(a,r.length));const u=o.length&-4;u&&s.queue.writeBuffer(e,n+t,o,0,u),r=o.subarray(u).slice(),t+=u},finish(){if(!r.length)return;const a=new Uint8Array(4);a.set(r),s.queue.writeBuffer(e,n+t,a),t+=4,r=new Uint8Array(0)}}},Mr=(e,n,t)=>{e.src==="aux"?(n(new Uint8Array(te(e).buffer,e.off,e.len)),t()):Dr.push({off:e.off,len:e.len,push:n,finish:t})},Sr=(e,n,t=0)=>{const r=Kn(n,t);Mr(e,r.push,r.finish)},vt=e=>n=>{const t=new Uint8Array(n.length);for(let r=0;r<n.length;r++)t[r]=wt[n[r]];e(t)},kt=e=>n=>{const t=new Uint8Array(n.length*2);for(let r=0;r<n.length;r++)t[2*r]=Cr[2*n[r]],t[2*r+1]=Cr[2*n[r]+1];e(t)},Jt=(e,n,t=0)=>{const r=Kn(n,t);Mr(e,vt(r.push),r.finish)},ea=(e,n)=>{const t=Kn(n,0);Mr(e,kt(t.push),t.finish)},na=e=>{const n=e&32768?-1:1,t=e>>10&31,r=e&1023;return t===0?n*r*2**-24:t===31?r?NaN:n*(1/0):n*(1024+r)*2**(t-25)},$r=(e,n,t,r,a)=>{let o=0,u=0;const i=l=>{const f=new Uint8Array(l.length),m=new Float32Array((l.length>>4)+2);let p=0,h=0;for(let D=0;D<l.length;D++)o===0?(u=l[D],o=1):o===1?(m[h++]=na(u|l[D]<<8),o=2):(f[p++]=l[D],o=o===17?0:o+1);p&&n(f.subarray(0,p)),h&&r(new Uint8Array(m.buffer,0,h*4))};Dr.push({off:e.off,len:e.len,push:i,finish:()=>{t(),a()}})},Ir=e=>{const n=new Uint8Array(e.len);let t=0;return Mr(e,r=>{n.set(r,t),t+=r.length},()=>{}),n};s.pushErrorScope("validation"),s.pushErrorScope("out-of-memory");const Q={},Or=[];for(const[e,n]of Object.entries(S))if(n.kind==="q2"){const t=xn(n.weight.len*2),r=xn(n.scales.len);if(n.q1_0){const o=Kn(t,0),u=Kn(r,0);$r(n.q1_0,kt(o.push),o.finish,u.push,u.finish)}else ea(n.weight,t),Sr(n.scales,r);const a={N:n.N,K:n.K,nb:n.K/128,zp:2,codes:t,scales:r};if(n.zp){const o=Ir(n.zp);Or.push(()=>{const u=o[0];for(let l=1;l<o.length;l++)if(o[l]!==u)throw new Error(`bitgpu: tensor ${e} has non-uniform 2-bit zero-points (the q2 kernels assume one zp for the whole tensor)`);const i=u&3;if(u!==i*85)throw new Error(`bitgpu: tensor ${e} has non-uniform 2-bit zero-points within a byte (the q2 kernels assume one zp for the whole tensor)`);a.zp=i})}Q[e]=a}else if(n.kind==="f32"&&n.weight){const t=xn(n.weight.len);Sr(n.weight,t),Q[e]={buf:t}}const Br=e=>{const n=xn(e.reduce((o,u)=>o+u.weight.len,0)),t=xn(e.reduce((o,u)=>o+u.scales.len,0));let r=0,a=0;for(const o of e){if(o.q1_0){const u=Kn(n,r),i=Kn(t,a);$r(o.q1_0,vt(u.push),u.finish,i.push,i.finish)}else Jt(o.weight,n,r),Sr(o.scales,t,a);r+=o.weight.len,a+=o.scales.len}return{sign:n,scales:t}};if(V.arch.hybrid){const e=n=>{const t=S[n];Q[n]={N:t.N,K:t.K,nb:t.K/128,N0:t.N,N1:0,N2:0,...Br([t])}};for(let n=0;n<c.layers;n++){for(const t of["mlp.gate_proj","mlp.up_proj","mlp.down_proj"])e(`layers.${n}.${t}`);if(V.arch.hybrid.layer_types[n]==="full")for(const t of["attn.q_proj","attn.k_proj","attn.v_proj","attn.o_proj"])e(`layers.${n}.${t}`);else for(const t of["linear.in_qkv","linear.z","linear.a","linear.b","linear.out_proj"])e(`layers.${n}.${t}`)}}else for(let e=0;e<c.layers;e++){const n=S[`layers.${e}.attn.q_proj`],t=S[`layers.${e}.attn.k_proj`],r=S[`layers.${e}.attn.v_proj`];Q[`layers.${e}.attn.qkv`]={K:n.K,nb:n.K/128,N0:n.N,N1:t.N,N2:r.N,...Br([n,t,r])};const a=S[`layers.${e}.mlp.gate_proj`],o=S[`layers.${e}.mlp.up_proj`];Q[`layers.${e}.mlp.gateup`]={K:a.K,nb:a.K/128,N0:a.N,N1:o.N,N2:0,...Br([a,o])};for(const u of[`layers.${e}.attn.o_proj`,`layers.${e}.mlp.down_proj`]){const i=S[u];Q[u]={N:i.N,K:i.K,nb:i.K/128,...Br([i])}}}const Gr=e=>{const n=xn(e.len);return Sr(e,n),n};let Hn,En,An;if(S.embed_tokens.q1_0){Hn=xn(S.embed_tokens.weight.len),En=xn(S.embed_tokens.scales.len);const e=Kn(Hn,0),n=Kn(En,0);$r(S.embed_tokens.q1_0,e.push,e.finish,n.push,n.finish),An=xn(Tr),s.queue.writeBuffer(An,0,new Uint8Array(Tr+3&-4).fill(136))}else Hn=Gr(S.embed_tokens.weight),En=Gr(S.embed_tokens.scales),An=Gr(S.embed_tokens.zp);const Pr=Gr(V.luts.tgt4);let $n,tr;if(S.cos_cache){const e=Ir(S.cos_cache),n=Ir(S.sin_cache);$n=new Float32Array(e.buffer),tr=new Float32Array(n.buffer);const t=$n.length/(Y/2);if(E>t)throw new Error(`bitgpu: maxSeqLen ${E} exceeds the model's baked RoPE cache (${t} positions); lower maxSeqLen or re-export with a longer cache`)}else{const e=c.max_positions??40960;if(E>e)throw new Error(`bitgpu: maxSeqLen ${E} exceeds the model's max_positions (${e})`);[$n,tr]=Sa(c,E,Y)}Dr.sort((e,n)=>e.off-n.off||e.len-n.len);const bn=[];for(const e of Dr){const n=bn[bn.length-1];if(n&&n.off===e.off&&n.len===e.len){const t=n.push,r=n.finish;n.push=a=>{t(a),e.push(a)},n.finish=()=>{r(),e.finish()}}else{if(n&&e.off<n.off+n.len)throw new Error("bitgpu: partially overlapping data-file tensor ranges (unsupported by the streaming loader)");bn.push(e)}}const zr=bn.length?bn[bn.length-1].off+bn[bn.length-1].len:0,ra=(d.fetchStream?await d.fetchStream(he):d.fetchArrayBuffer?new Response(await d.fetchArrayBuffer(he)).body:await(async()=>{const e=await fetch(he);if(!e.ok)throw new Error(`bitgpu: fetch ${he} failed: HTTP ${e.status}`);return e.body??new Response(await e.arrayBuffer()).body})()).getReader();let un=0,Vr=0;for(;;){const{done:e,value:n}=await ra.read();if(e)break;let t=0;for(;t<n.byteLength&&Vr<bn.length;){const r=bn[Vr];if(un>=r.off+r.len){Vr++;continue}if(un<r.off){const o=Math.min(r.off-un,n.byteLength-t);un+=o,t+=o;continue}const a=Math.min(r.off+r.len-un,n.byteLength-t);r.push(n.subarray(t,t+a)),un+=a,t+=a,un===r.off+r.len&&(r.finish(),Vr++)}un+=n.byteLength-t,d.onProgress?.({phase:"weights",loaded:Math.min(un,zr),total:zr})}if(un<zr)throw new Error(`bitgpu: the data file ended at ${un} bytes but tensors extend to ${zr}; the download is truncated or the manifest does not match it`);for(const e of Or)e();Or.length=0,se=null;function ar(e,n){const t=Nr(new Uint32Array(n)),r=v(n.length*w),a=e.beginComputePass();return ze(a,"embed_gather_batch",[["u",n.length],["u",w],["u",0],["u",0]],[t,Hn,Pr,En,An],r,n.length*w),a.end(),r}function ta(e,n){const t=Y,r=t/2,a=new Float32Array(n*t),o=new Float32Array(n*t);for(let l=0;l<n;l++)for(let f=0;f<t;f++)a[l*t+f]=$n[(e+l)*r+f%r],o[l*t+f]=tr[(e+l)*r+f%r];const u=v(n*t),i=v(n*t);return s.queue.writeBuffer(u,0,a),s.queue.writeBuffer(i,0,o),{cos:u,sin:i}}const _=c.kv_heads,g=c.head_dim,w=c.hidden,A=c.heads,We=c.intermediate,yt=c.hybrid?.linear_key_heads??0,Ye=c.hybrid?.linear_value_heads??0,Wn=c.hybrid?.linear_head_dim??0,Fr=c.hybrid?.conv_kernel??0,wn=yt*Wn,Sn=Ye*Wn,vn=wn*2+Sn,aa=1.25*(1<<30),oa=(()=>{if(!c.hybrid)return 0;const e=3*w+3*We,n=2*vn+2*wn+4*Sn+4*Ye+w,t=8*A*g+4*_*g+w;return 3*(e+Math.max(n,t))*4})(),_t=c.hybrid?Math.max(zn,Math.min(lt,Math.floor(aa/oa/zn)*zn)):lt;let ln=Math.min(E,512);const Pe=[],Le=[],Ze=[],tn=[],or=e=>e*_*(g/32)*4,cn=[];for(let e=0;e<c.layers;e++)(!c.hybrid||c.hybrid.layer_types[e]==="full")&&cn.push(e);const ir=[];for(let e=0;e<c.layers;e++)c.hybrid&&c.hybrid.layer_types[e]==="linear"&&ir.push(e);const In=c.hybrid?Ye*Wn*Wn*4:0,On=c.hybrid?(Fr-1)*vn*4:0;for(const e of cn)Pe[e]=s.createBuffer({size:ln*_*g*qn,usage:y|P|q}),Le[e]=s.createBuffer({size:ln*_*g*qn,usage:y|P|q}),be&&(Ze[e]=s.createBuffer({size:or(ln),usage:y|P|q}),tn[e]=s.createBuffer({size:or(ln),usage:y|P|q}));const sr=[],ur=[];let lr=0;if(c.hybrid)for(const e of ir)sr[e]=[s.createBuffer({size:In,usage:y|P|q}),s.createBuffer({size:In,usage:y|P|q})],ur[e]=[s.createBuffer({size:On,usage:y|P|q}),s.createBuffer({size:On,usage:y|P|q})];let Qr=null,Xr=null,Yr=null,Zr=null;if(ve){const e=g/2;Qr=s.createBuffer({size:E*e*4,usage:y|q}),Xr=s.createBuffer({size:E*e*4,usage:y|q}),s.queue.writeBuffer(Qr,0,$n.buffer,$n.byteOffset,E*e*4),s.queue.writeBuffer(Xr,0,tr.buffer,tr.byteOffset,E*e*4),Yr=s.createBuffer({size:g*4,usage:y|q}),Zr=s.createBuffer({size:g*4,usage:y|q}),s.queue.writeBuffer(Yr,0,new Float32Array(g).fill(1)),s.queue.writeBuffer(Zr,0,new Float32Array(g))}const xt=await s.popErrorScope(),Kt=await s.popErrorScope();if(xt)throw new Lr(`GPU allocation failed while loading weights (~${Math.round(dt/1048576)} MB VRAM needed): ${xt.message}`);if(Kt)throw new Error(`bitgpu: WebGPU validation error while loading weights: ${Kt.message}`);async function kn(e){if(e=Math.min(e,E),e<=ln)return;const n=Math.min(E,Math.max(e,ln*2)),t=ln*_*g*qn,r=or(ln);s.pushErrorScope("out-of-memory");const a=s.createCommandEncoder(),o=[];for(const i of cn){const l=s.createBuffer({size:n*_*g*qn,usage:y|P|q}),f=s.createBuffer({size:n*_*g*qn,usage:y|P|q});a.copyBufferToBuffer(Pe[i],0,l,0,t),a.copyBufferToBuffer(Le[i],0,f,0,t);const m=[Pe[i],Le[i]];if(Pe[i]=l,Le[i]=f,be){const p=s.createBuffer({size:or(n),usage:y|P|q}),h=s.createBuffer({size:or(n),usage:y|P|q});a.copyBufferToBuffer(Ze[i],0,p,0,r),a.copyBufferToBuffer(tn[i],0,h,0,r),m.push(Ze[i],tn[i]),Ze[i]=p,tn[i]=h}o[i]=m}s.queue.submit([a.finish()]),await s.queue.onSubmittedWorkDone();const u=await s.popErrorScope();if(u){for(const i of cn)Pe[i].destroy(),Le[i].destroy(),Pe[i]=o[i][0],Le[i]=o[i][1],be&&(Ze[i].destroy(),tn[i].destroy(),Ze[i]=o[i][2],tn[i]=o[i][3]);throw nr(),new Lr(`KV cache growth to ${n} positions failed: ${u.message}`)}for(const i of cn)for(const l of o[i])l.destroy();ln=n,nr()}let Fn=null;function Nt(e,n){const t=e-Xe,r=Math.min(t,Math.max(n,Math.ceil((E-Xe)/4))),a=t-r;if(a<=0)return Xe;const o=_*g*qn,u=be?_*(g/32)*4:0;(!Fn||Fn.size<a*o)&&(Fn?.destroy(),Fn=s.createBuffer({size:a*o,usage:P|q}));const i=s.createCommandEncoder(),l=[];for(const f of cn)l.push([Pe[f],o],[Le[f],o]),be&&l.push([Ze[f],u],[tn[f],u]);for(const[f,m]of l)i.copyBufferToBuffer(f,(Xe+r)*m,Fn,0,a*m),i.copyBufferToBuffer(Fn,0,f,Xe*m,a*m);return s.queue.submit([i.finish()]),Xe+a}const Jr=(e,n)=>ve&&e+n>E?Nt(e,e+n-E):e;async function Je(e,n){const t=s.createBuffer({size:n*4,usage:GPUBufferUsage.MAP_READ|q}),r=s.createCommandEncoder();r.copyBufferToBuffer(e,0,t,0,n*4),s.queue.submit([r.finish()]),await t.mapAsync(GPUMapMode.READ);const a=new Float32Array(t.getMappedRange().slice(0));return t.unmap(),t.destroy(),a}async function cr(e,n){const t=s.createBuffer({size:n*4,usage:GPUBufferUsage.MAP_READ|q}),r=s.createCommandEncoder();r.copyBufferToBuffer(e,0,t,0,n*4),s.queue.submit([r.finish()]),await t.mapAsync(GPUMapMode.READ);const a=new Uint32Array(t.getMappedRange().slice(0));return t.unmap(),t.destroy(),a}let Rr=null,et=!1,ia=null,sa=null,ua=null;const nt=()=>ia??=s.createQuerySet({type:"timestamp",count:2}),qt=()=>sa??=s.createBuffer({size:16,usage:GPUBufferUsage.QUERY_RESOLVE|P}),Hr=()=>ua??=s.createBuffer({size:16,usage:GPUBufferUsage.MAP_READ|q}),dr=e=>Rr===null||Rr.has(e)||e==="embed_gather_batch";let Qn=!1,Xn=0,Ln=null;const Yn=(e,n,t)=>{e===0&&Ln&&(Ln[n]=t)};function je(e,n,t,r,a){if(e.setPipeline(yr[n]),sn){let o=sn.disp[qr];o||(o={uni:s.createBuffer({size:64,usage:gt|q}),bg:null,last:null,bufs:null},sn.disp[qr]=o);const u=Et(t);if((!o.last||!Ma(o.last,u))&&(s.queue.writeBuffer(o.uni,0,u),o.last=u.slice()),o.bg&&o.bufs){const i=r.length+a.length;if(o.bufs.length!==i)o.bg=null;else{for(let l=0;l<r.length;l++)if(o.bufs[l]!==r[l]){o.bg=null;break}if(o.bg){for(let l=0;l<a.length;l++)if(o.bufs[r.length+l]!==a[l]){o.bg=null;break}}}}if(!o.bg){const i=[{binding:0,resource:{buffer:o.uni}}];r.forEach((l,f)=>i.push({binding:f+1,resource:{buffer:l}})),a.forEach((l,f)=>i.push({binding:1+r.length+f,resource:{buffer:l}})),o.bg=s.createBindGroup({layout:yr[n].getBindGroupLayout(0),entries:i}),o.bufs=[...r,...a]}e.setBindGroup(0,o.bg),qr++}else{const o=[{binding:0,resource:{buffer:Nr(Et(t),gt|q)}}];r.forEach((u,i)=>o.push({binding:i+1,resource:{buffer:u}})),a.forEach((u,i)=>o.push({binding:1+r.length+i,resource:{buffer:u}})),e.setBindGroup(0,s.createBindGroup({layout:yr[n].getBindGroupLayout(0),entries:o}))}}const Dt=e=>{const n=Math.ceil(e/65535);return[Math.ceil(e/n),n]};function rt(e,n,t,r,a,o){if(je(e,n,t,r,a),!dr(n))return void e.dispatchWorkgroups(1);const[u,i]=Dt(Math.ceil(o/64));e.dispatchWorkgroups(u,i,1)}const ze=(e,n,t,r,a,o)=>rt(e,n,t,r,[a],o);function Re(e,n,t,r,a,o){je(e,n,t,r,[a]),e.dispatchWorkgroups(dr(n)?o:1)}function en(e,n,t,r,a,o,u){je(e,n,t,r,a);const i=dr(n);e.dispatchWorkgroups(i?o:1,i?u:1,1)}const yn=(e,n,t,r,a,o,u=!1)=>u?Re(e,"rmsnorm_sg_af16",[["u",r],["u",a],["f",c.rms_eps],["u",0]],[n,Q[t].buf],o,r):xe?Re(e,"rmsnorm_sg",[["u",r],["u",a],["f",c.rms_eps],["u",0]],[n,Q[t].buf],o,r):Re(e,"rmsnorm_wg",[["u",r],["u",a],["f",c.rms_eps],["u",0]],[n,Q[t].buf],o,r);function dn(e,n,t,r,a,o=!1){const u=n.N0+n.N1+n.N2;if(xe&&r===1){const i=Math.min(u,65535);en(e,o?"matmul_split_sg_af16":"matmul_split_sg",[["u",n.K],["u",n.nb],["u",n.N0],["u",n.N1],["u",n.N2],["u",i]],[t,n.sign,n.scales],a,i,Math.ceil(u/i))}else if(r===1){const i=Math.min(u,65535);en(e,"matmul_split_wg",[["u",n.K],["u",n.nb],["u",n.N0],["u",n.N1],["u",n.N2],["u",i]],[t,n.sign,n.scales],a,i,Math.ceil(u/i))}else if(xe&&r===Xn){const i=Math.min(u,65535);en(e,"matmul_split_sm",[["u",n.K],["u",n.nb],["u",n.N0],["u",n.N1],["u",n.N2],["u",i],["u",r]],[t,n.sign,n.scales],a,i,Math.ceil(u/i))}else ct(r)?en(e,"matmul_split_tiled",[["u",r],["u",n.K],["u",n.nb],["u",n.N0],["u",n.N1],["u",n.N2]],[t,n.sign,n.scales],a,Math.ceil(u/64),Math.ceil(r/64)):rt(e,"matmul_split",[["u",r],["u",n.K],["u",n.nb],["u",n.N0],["u",n.N1],["u",n.N2]],[t,n.sign,n.scales],a,r*u)}function jn(e,n,t,r,a,o,u=!1){if(xe&&a===1){const i=Math.ceil(n.N/_r),l=Math.min(i,65535);en(e,u?"matmul_resid_mr_sg_af16":"matmul_resid_mr_sg",[["u",n.N],["u",n.K],["u",n.nb],["u",l],["u",0],["u",0]],[t,n.sign,n.scales,r],[o],l,Math.ceil(i/l))}else if(a===1){const i=Math.min(n.N,65535);en(e,"matmul_resid_wg",[["u",n.N],["u",n.K],["u",n.nb],["u",i],["u",0],["u",0]],[t,n.sign,n.scales,r],[o],i,Math.ceil(n.N/i))}else if(xe&&a===Xn){const i=Math.min(n.N,65535);en(e,"matmul_resid_sm",[["u",n.N],["u",n.K],["u",n.nb],["u",i],["u",a],["u",0]],[t,n.sign,n.scales,r],[o],i,Math.ceil(n.N/i))}else ct(a)?en(e,"matmul_resid_tiled",[["u",a],["u",n.N],["u",n.K],["u",n.nb],["u",0],["u",0]],[t,n.sign,n.scales,r],[o],Math.ceil(n.N/64),Math.ceil(a/64)):rt(e,"matmul_resid",[["u",a],["u",n.N],["u",n.K],["u",n.nb],["u",128],["u",0]],[t,n.sign,n.scales,r],[o],a*n.N)}function pr(e,n,t,r,a,o){const u=t===0?Pe[r]:Le[r];be?(je(e,"copy_kv8",[["u",a],["u",g],["u",o],["u",0]],[n],[u,t===0?Ze[r]:tn[r]]),e.dispatchWorkgroups(dr("copy_kv8")?a:1)):ze(e,Yt,[["u",a*g],["u",o*g],["u",0],["u",0]],[n],u,a*g)}const Mt=(e,n)=>{const t=be?[e,Pe[n],Le[n],Ze[n],tn[n]]:[e,Pe[n],Le[n]];return ve&&t.push(Qr,Xr),t};function la(e,n,t,r,a,o,u){const i=V.arch.hybrid,l=ue=>Q[`layers.${n}.${ue}`],f=a>0?1:0,m=lr,p=er&&r===1&&!Qn,h=p?rr(r*w):v(r*w);yn(e,t,`layers.${n}.input_layernorm`,r,w,h,p);let D;if(i.layer_types[n]==="linear"){const ue=v(r*vn);dn(e,l("linear.in_qkv"),h,r,[ue,hn,_n],p);const Z=v(r*vn),le=(Fr-1)*vn;je(e,"conv1d_causal",[["u",r],["u",vn],["u",Fr],["u",f]],[ue,l("linear.conv1d").buf,ur[n][m]],[Z,ur[n][m^1]]);{const[W,ge]=Dt(Math.ceil((r*vn+le)/64));e.dispatchWorkgroups(W,ge,1)}const $=v(r*wn),oe=v(r*wn),F=v(r*Sn);ze(e,"slice_cols",[["u",r],["u",wn],["u",vn],["u",0]],[Z],$,r*wn),ze(e,"slice_cols",[["u",r],["u",wn],["u",vn],["u",wn]],[Z],oe,r*wn),ze(e,"slice_cols",[["u",r],["u",Sn],["u",vn],["u",2*wn]],[Z],F,r*Sn);const we=v(r*Ye),qe=v(r*Ye);dn(e,l("linear.a"),h,r,[we,hn,_n],p),dn(e,l("linear.b"),h,r,[qe,hn,_n],p);const ce=v(2*r*Ye);ze(e,"deltanet_gbeta",[["u",r],["u",Ye],["u",0],["u",0]],[we,qe,l("linear.A_log").buf,l("linear.dt_bias").buf],ce,r*Ye);const ye=v(r*Sn);{const W=[];Math.ceil(r/zn)%2===0&&W.push(zn/2,zn/2);for(let I=r-W.reduce((He,J)=>He+J,0);I>0;I-=zn)W.push(Math.min(I,zn));let ge=0;for(let I=0;I<W.length;I++){const He=sr[n][m^I&1],J=sr[n][m^(I&1^1)];je(e,"deltanet_recur",[["u",W[I]],["u",Ye],["u",Wn],["u",Wn],["u",yt],["u",r*Ye],["u",I===0?f:1],["u",ge]],[$,oe,F,ce,ce,He],[ye,J]),e.dispatchWorkgroups(Ye),ge+=W[I]}}const ie=v(r*Sn);dn(e,l("linear.z"),h,r,[ie,hn,_n],p);const de=v(r*Sn);Re(e,"deltanet_norm_gate",[["u",r*Ye],["u",Wn],["f",c.rms_eps],["u",0]],[ye,ie,l("linear.norm").buf],de,r*Ye),D=v(r*w),jn(e,l("linear.out_proj"),de,t,r,D)}else{const ue=v(r*A*g*2);dn(e,l("attn.q_proj"),h,r,[ue,hn,_n],p);const Z=v(r*A*g),le=v(r*A*g);ze(e,"split_head",[["u",r],["u",A],["u",g],["u",0]],[ue],Z,r*A*g),ze(e,"split_head",[["u",r],["u",A],["u",g],["u",g]],[ue],le,r*A*g);const $=v(r*A*g),oe=v(r*A*g);yn(e,Z,`layers.${n}.attn.q_norm`,r*A,g,$),ze(e,"rope_partial",[["u",r],["u",A],["u",g],["u",Y]],[$,o,u],oe,r*A*g);const F=v(r*_*g);dn(e,l("attn.k_proj"),h,r,[F,hn,_n],p);const we=v(r*_*g),qe=v(r*_*g);yn(e,F,`layers.${n}.attn.k_norm`,r*_,g,we),ze(e,"rope_partial",[["u",r],["u",_],["u",g],["u",Y]],[we,o,u],qe,r*_*g);const ce=v(r*_*g);dn(e,l("attn.v_proj"),h,r,[ce,hn,_n],p),pr(e,qe,0,n,r*_,a*_),pr(e,ce,1,n,r*_,a*_);const ye=v(r*A*g),ie=[["u",r],["u",A],["u",_],["u",g],["f",1/Math.sqrt(g)],["u",a],["u",0],["u",0]];be?Re(e,"attention_online_cache_kv8",ie,[oe,Pe[n],Ze[n],Le[n],tn[n]],ye,r*A):Re(e,"attention_online_cache",ie,[oe,Pe[n],Le[n]],ye,r*A);const de=v(r*A*g);ze(e,"gate_sigmoid",[["u",r*A*g],["u",0],["u",0],["u",0]],[ye,le],de,r*A*g),D=v(r*w),jn(e,l("attn.o_proj"),de,t,r,D)}const B=p?rr(r*w):v(r*w);yn(e,D,`layers.${n}.post_attention_layernorm`,r,w,B,p);const j=v(r*We),ae=v(r*We);dn(e,l("mlp.gate_proj"),B,r,[j,hn,_n],p),dn(e,l("mlp.up_proj"),B,r,[ae,hn,_n],p);const G=v(r*We);ze(e,"swiglu",[["u",r*We],["u",0],["u",0],["u",0]],[j,ae],G,r*We);const R=v(r*w);return jn(e,l("mlp.down_proj"),G,D,r,R),R}function ca(e,n,t,r,a,o,u){const i=a+r;if(V.arch.hybrid)return la(e,n,t,r,a,o,u);const l=er&&r===1&&!Qn,f=l?rr(w):v(r*w);yn(e,t,`layers.${n}.input_layernorm`,r,w,f,l);const m=Q[`layers.${n}.attn.qkv`];if(xe&&r===1&&!Qn){const ye=v(A*g),ie=v(_*g),de=v(_*g),W=m.N0+m.N1+m.N2,ge=Math.min(W,65535);en(e,l?"matmul_split_sg_af16":"matmul_split_sg",[["u",m.K],["u",m.nb],["u",m.N0],["u",m.N1],["u",m.N2],["u",ge]],[f,m.sign,m.scales],[ye,ie,de],ge,Math.ceil(W/ge)),pr(e,de,1,n,_,a*_);const I=v(A*g);Re(e,"rmsnorm_rope_sg",[["u",A],["u",g],["f",c.rms_eps],["u",0],["u",g],["u",0]],[ye,Q[`layers.${n}.attn.q_norm`].buf,o,u],I,A);const He=ve?Yr:o,J=ve?Zr:u;be?(je(e,"rmsnorm_rope_sg_kv8",[["u",_],["u",g],["f",c.rms_eps],["u",a*_],["u",0],["u",0]],[ie,Q[`layers.${n}.attn.k_norm`].buf,He,J],[Pe[n],Ze[n]]),e.dispatchWorkgroups(dr("rmsnorm_rope_sg_kv8")?_:1)):Re(e,Xt,[["u",_],["u",g],["f",c.rms_eps],["u",a*_*g],["u",g],["u",0]],[ie,Q[`layers.${n}.attn.k_norm`].buf,He,J],Pe[n],_),Yn(n,"qr",I);const Ee=v(A*g);Re(e,ft,[["u",1],["u",A],["u",_],["u",g],["u",a],["u",i]],Mt(I,n),Ee,A),Yn(n,"att",Ee);const ee=Q[`layers.${n}.attn.o_proj`],U=v(w);jn(e,ee,Ee,t,1,U);const De=l?rr(w):v(w);yn(e,U,`layers.${n}.post_attention_layernorm`,1,w,De,l);const Se=Q[`layers.${n}.mlp.gateup`],Oe=l?rr(We):v(We),Te=Math.ceil(We/_r),Ue=Math.min(Te,65535);en(e,l?"matmul_swiglu_mr_sg_af16":"matmul_swiglu_mr_sg",[["u",Se.K],["u",Se.nb],["u",We],["u",Ue],["u",0],["u",0]],[De,Se.sign,Se.scales],[Oe],Ue,Math.ceil(Te/Ue)),Yn(n,"sw",Oe);const gr=Q[`layers.${n}.mlp.down_proj`],gn=v(w);return jn(e,gr,Oe,U,1,gn,l),gn}const p=v(r*A*g),h=v(r*_*g),D=v(r*_*g);dn(e,m,f,r,[p,h,D]);const B=v(r*A*g),j=v(r*_*g);yn(e,p,`layers.${n}.attn.q_norm`,r*A,g,B),yn(e,h,`layers.${n}.attn.k_norm`,r*_,g,j);const ae=v(r*A*g),G=v(r*_*g);ze(e,"rope",[["u",r],["u",A],["u",g],["u",0]],[B,o,u],ae,r*A*g),ve||ze(e,"rope",[["u",r],["u",_],["u",g],["u",0]],[j,o,u],G,r*_*g),pr(e,ve?j:G,0,n,r*_,a*_),pr(e,D,1,n,r*_,a*_),Yn(n,"qr",ae);const R=v(r*A*g);Re(e,ft,[["u",r],["u",A],["u",_],["u",g],["u",a],["u",i]],Mt(ae,n),R,r*A),Yn(n,"att",R);const ue=Q[`layers.${n}.attn.o_proj`],Z=v(r*w);jn(e,ue,R,t,r,Z);const le=v(r*w);yn(e,Z,`layers.${n}.post_attention_layernorm`,r,w,le);const $=Q[`layers.${n}.mlp.gateup`],oe=v(r*We),F=v(r*We);dn(e,$,le,r,[oe,F,hn]);const we=v(r*We);ze(e,"swiglu",[["u",r*We],["u",0],["u",0],["u",0]],[oe,F],we,r*We),Yn(n,"sw",we);const qe=Q[`layers.${n}.mlp.down_proj`],ce=v(r*w);return jn(e,qe,we,Z,r,ce),ce}function Nn(e,n,t,r){const a=Q.lm_head;if(xe&&t===1){const o=Math.min(a.N,65535);en(e,"matmul_q2_sg",[["u",a.N],["u",a.K],["u",a.nb],["u",a.zp],["u",o],["u",0]],[n,a.codes,a.scales],[r],o,Math.ceil(a.N/o))}else if(t===1){const o=Math.min(a.N,65535);en(e,"matmul_q2_wg",[["u",a.N],["u",a.K],["u",a.nb],["u",a.zp],["u",o],["u",0]],[n,a.codes,a.scales],[r],o,Math.ceil(a.N/o))}else if(xe&&t===Xn){const o=Math.min(a.N,65535);en(e,"matmul_q2_sm",[["u",a.N],["u",a.K],["u",a.nb],["u",a.zp],["u",o],["u",t]],[n,a.codes,a.scales],[r],o,Math.ceil(a.N/o))}else ze(e,"matmul_q2",[["u",t],["u",a.N],["u",a.K],["u",a.nb],["u",128],["u",a.zp]],[n,a.codes,a.scales],r,t*a.N)}function Bn(e,n,t,r){const{cos:a,sin:o}=ta(r,t),u=e.beginComputePass(),i=!sn&&!Ln;let l=null,f=n;i&&(Dn=Dn??new Map);for(let p=0;p<c.layers;p++){const h=[];i&&(Kr=h);const D=f;if(f=ca(u,p,D,t,r,a,o),p===0&&(l=f),i){Kr=null;for(const B of h)B!==f&&mt(B);D!==n&&D!==l&&mt(D)}}c.hybrid&&(lr^=1);const m=v(t*w);return yn(u,f,T,t,w,m),u.end(),Dn=null,{fn:m,layer0:l}}async function da(e){const n=e.length;if(n===0)throw new Error("forward: no tokens to process");if(n>E)throw new Error(`forward: sequence length ${n} exceeds maxSeqLen ${E}`);Ke=[],Ne=0,await kn(n);const t=Q.lm_head.N,r=new Float32Array(n*w),a=new Float32Array(n*w),o=new Float32Array(n*w),u=new Float32Array(n*t);ne=[];const i=Math.min((globalThis.__SEG??0)||_t,32);try{for(let l=0;l<n;l+=i){const f=e.slice(l,l+i),m=s.createCommandEncoder(),p=ar(m,f),{fn:h,layer0:D}=Bn(m,p,f.length,l),B=s.createBuffer({size:f.length*t*4,usage:y|P});ne.push(B);const j=m.beginComputePass();Nn(j,h,f.length,B),j.end(),s.queue.submit([m.finish()]),await s.queue.onSubmittedWorkDone(),r.set(await Je(p,f.length*w),l*w),a.set(await Je(D,f.length*w),l*w),o.set(await Je(h,f.length*w),l*w),u.set(await Je(B,f.length*t),l*t),Ie()}return{embed:r,layer0:a,finalnorm:o,logits:u,vocab:t,sequenceLength:n}}finally{Ie(),ne=null}}async function fr(e,n,t){let r=null,a=0;const o=(globalThis.__SEG??0)||_t;for(let u=0;u<e.length;u+=o){if(u>0&&t?.aborted)return Ke=[],Ne=0,null;const i=e.slice(u,u+o);s.pushErrorScope("out-of-memory");const l=s.createCommandEncoder();if(r=Bn(l,ar(l,i),i.length,n+u).fn,a=i.length-1,s.queue.submit([l.finish()]),await s.popErrorScope())throw new Error(`bitgpu: GPU out of memory during prefill (segment of ${i.length} tokens at position ${n+u}) - the output would have been silently corrupted. Lower maxSeqLen, use kvCache: 'q8', or free GPU memory.`);u+o<e.length&&(await s.queue.onSubmittedWorkDone(),Ie())}return{fn:r,lastRow:a}}async function tt(e,n,t,r=null,a=br,o){await kn(n+e.length+t),Rr=r;const u=Q.lm_head.N,i=s.createBuffer({size:Math.max(1,t)*4,usage:y|P}),l=s.createBuffer({size:w*4,usage:y|P|q}),f=s.createBuffer({size:u*4,usage:y|P});ne=[];try{const m=performance.now(),p=await fr(e,n,o?.signal);if(!p)return{prefillMs:performance.now()-m,decodeMs:0,tokPerSec:0,tokens:[],firstArgmax:-1,recMs:0,gpuMs:0,rbMs:0};const h=s.createCommandEncoder(),D=v(w);h.copyBufferToBuffer(p.fn,p.lastRow*w*4,D,0,w*4);let B=h.beginComputePass();Nn(B,D,1,f),B.end(),B=h.beginComputePass(),Re(B,"argmax",[["u",u],["u",0],["u",0],["u",0]],[f],i,1),B.end(),s.queue.submit([h.finish()]),await s.queue.onSubmittedWorkDone();const j=(await cr(i,1))[0];Ie();const ae=performance.now()-m,G=[];let R=0,ue=0,Z=0,le=0;const $=et&&vr,oe=performance.now();let F=1;const we=o?.stopTokens?new Set(o.stopTokens):null;let qe=we?.has(j)??!1;qe||(G.push(j),o?.onToken?.(j)),nr();let ce=n+e.length;for(;F<t&&!qe&&!o?.signal?.aborted;){const de=Math.min(a,t-F);ce=Jr(ce,de),Mn("decode");let W=performance.now();const ge=s.createCommandEncoder();for(let J=0;J<de;J++){const Ee=F+J,ee=ce+J;let U=ge.beginComputePass($&&J===0?{timestampWrites:{querySet:nt(),beginningOfPassWriteIndex:0}}:void 0);Re(U,"embed_gather",[["u",w],["u",Ee-1],["u",0],["u",0]],[i,Hn,Pr,En,An],l,1),U.end();const De=Bn(ge,l,1,ee),Se=v(w);ge.copyBufferToBuffer(De.fn,0,Se,0,w*4),U=ge.beginComputePass($&&J===de-1?{timestampWrites:{querySet:nt(),endOfPassWriteIndex:1}}:void 0),Nn(U,Se,1,f),Re(U,"argmax",[["u",u],["u",Ee],["u",0],["u",0]],[f],i,1),U.end()}if($&&(ge.resolveQuerySet(nt(),0,2,qt(),0),ge.copyBufferToBuffer(qt(),0,Hr(),0,16)),s.queue.submit([ge.finish()]),R+=performance.now()-W,W=performance.now(),await s.queue.onSubmittedWorkDone(),ue+=performance.now()-W,$){await Hr().mapAsync(GPUMapMode.READ);const J=new BigUint64Array(Hr().getMappedRange());le+=Number(J[1]-J[0]),Hr().unmap()}W=performance.now();const I=await cr(i,F+de);Z+=performance.now()-W;let He=de;for(let J=0;J<de;J++){const Ee=I[F+J];if(we?.has(Ee)){qe=!0,He=J;break}G.push(Ee),o?.onToken?.(Ee)}F+=de,ce+=He}Ne=ce;const ye=performance.now()-oe,ie=Math.max(1,G.length-1);return{prefillMs:ae,decodeMs:ye,tokPerSec:ie/(ye/1e3),tokens:G,firstArgmax:j,recMs:R/ie,gpuMs:ue/ie,rbMs:Z/ie,tsMs:$?le/1e6/ie:0}}finally{Mn(null),Rr=null,Ie(),ne=null,i.destroy(),l.destroy(),f.destroy()}}async function St(e,n,t,r,a,o){await kn(n+e.length+t);const u=r.temperature!=null&&r.temperature>0&&r.temperature!==1,i=Q.lm_head.N,l=Math.max(1,Math.min(r.topK??20,i)),f=Math.max(0,Math.min(Math.floor(r.logprobs??0),32,i)),m=Math.max(l,f),p=r.temperature??1,h=r.repetitionPenalty??1,D=r.presencePenalty??0,B=r.topP??1,j=r.minP??0,ae=r.noRepeatNgramSize??0,G=(r.dryMultiplier??0)>0?{multiplier:r.dryMultiplier,base:r.dryBase??1.75,allowedLength:r.dryAllowedLength??2,range:r.dryRange??0,breakers:new Set(r.dryBreakers??[])}:null,R=r.topNSigma??0,ue=r.stopTokens?new Set(r.stopTokens):null,Z=r.onToken,le=r.signal,$=o??new Rt(r.seed),oe=s.createBuffer({size:Math.max(1,t)*4,usage:y|P|q}),F=s.createBuffer({size:i*4,usage:y|P}),we=s.createBuffer({size:m*4,usage:y|P}),qe=s.createBuffer({size:m*4,usage:y|P}),ce=s.createBuffer({size:4,usage:y|P}),ye=s.createBuffer({size:12,usage:y|P}),ie=m*8+(f?4:0);let de=s.createBuffer({size:E*4,usage:y|q}),W=s.createBuffer({size:E*4,usage:y|q});const ge=(M,z)=>z<=M.size?M:(M.destroy(),s.createBuffer({size:1<<32-Math.clz32(z-1),usage:y|q})),I=s.createBuffer({size:ie+(R>0?12:0),usage:GPUBufferUsage.MAP_READ|q}),He=s.createBuffer({size:w*4,usage:y|P|q}),J=M=>{const z=h!==1||D!==0?it(M):new Uint32Array(0);z.length&&s.queue.writeBuffer(de=ge(de,z.byteLength),0,z);const H=ae>0?st(M,ae):[];return H.length&&s.queue.writeBuffer(W=ge(W,H.length*4),0,Uint32Array.from(H)),{affLen:z.length,banLen:H.length}},Ee=(M,z,H)=>{je(M,"sampler_penalty",[["u",z],["u",H],["f",h],["u",4286578688],["f",D]],[de,W],[F]),M.dispatchWorkgroups(1),f&&(je(M,"logsumexp",[["u",i],["u",0],["u",0],["u",0]],[F],[ce]),M.dispatchWorkgroups(1)),R>0&&(je(M,"sampler_sigma",[["u",i],["u",0],["u",0],["u",0]],[F],[ye]),M.dispatchWorkgroups(1));for(let C=0;C<m;C++)je(M,"argmax_masked",[["u",i],["u",C],["u",0],["u",0]],[F],[we,qe]),M.dispatchWorkgroups(1)},ee=M=>{M.copyBufferToBuffer(we,0,I,0,m*4),M.copyBufferToBuffer(qe,0,I,m*4,m*4),f&&M.copyBufferToBuffer(ce,0,I,m*8,4),R>0&&M.copyBufferToBuffer(ye,0,I,ie,12)};let U=0;const De=async()=>{await I.mapAsync(GPUMapMode.READ);const M=I.getMappedRange(),z=new Uint32Array(M.slice(0,m*4)),H=new Float32Array(M.slice(m*4,m*8)),C=f?new Float32Array(M.slice(m*8,m*8+4))[0]:0;if(R>0){const[O,re,pe]=new Float32Array(M.slice(ie,ie+12));U=pe>0?Math.sqrt(Math.max(0,re/pe-(O/pe)**2)):0}return I.unmap(),{ci:z,cv:H,lse:C}},Se=(M,z)=>{if(!(R>0)||z.length===0)return null;const H=z[0]-R*U;let C=1;for(;C<z.length&&z[C]>=H;)C++;return{ids:Array.prototype.slice.call(M,0,C),vals:Array.prototype.slice.call(z,0,C)}};let Oe=0;const Te=f?[]:null,Ue=(M,z,H)=>{if(!Te)return;const C=[];for(let O=0;O<f;O++)C.push({id:M[O],logprob:z[O]-H});Te.push({logprob:Oe-H,top:C})},gr=(M,z)=>{const H=M.subarray(0,l),C=z.subarray(0,l);let O=H,re=C;const pe=Se(H,C);if(pe&&(O=pe.ids,re=pe.vals),G){const Ae=ut(O,re,a,G);O=Ae.ids,re=Ae.vals}const fe=u?jr(O,re,p,$,B,j):O[0];return Oe=C[H.indexOf(fe)],fe},gn=r.candidateFilter,Tn=async(M,z)=>{if(!gn)return gr(M,z);const H=M.subarray(0,l),C=z.subarray(0,l),O=new Set(gn(H,C));{const re=[],pe=[];for(let L=0;L<H.length;L++)O.has(H[L])&&(re.push(H[L]),pe.push(C[L]));if(re.length===0)return mr(M,z);let fe=re,Ae=pe;const Ce=Se(re,pe);if(Ce&&(fe=Ce.ids,Ae=Ce.vals),G){const L=ut(fe,Ae,a,G);fe=L.ids,Ae=L.vals}const Fe=u?jr(fe,Ae,p,$,B,j):fe[0];return Oe=pe[re.indexOf(Fe)],Fe}},mr=async(M,z)=>{const H=await Je(F,i);for(let L=0;L<M.length;L++)H[M[L]]=z[L];const C=Array.from(H.keys()).sort((L,Be)=>H[Be]-H[L]||L-Be),O=[],re=[],pe=512;for(let L=0;L<C.length&&O.length<l&&!(H[C[L]]===-1/0&&O.length>0);L+=pe){const Be=C.slice(L,L+pe),_e=new Set(gn(Uint32Array.from(Be),Float32Array.from(Be.map(nn=>H[nn]))));for(const nn of Be)if(_e.has(nn)&&(O.push(nn),re.push(H[nn]),O.length>=l))break;if(re[0]===-1/0){O.length=1,re.length=1;break}}if(O.length===0)throw new Error("bitgpu: candidateFilter permitted no token in the entire vocabulary");let fe=O,Ae=re;const Ce=Se(O,re);if(Ce&&(fe=Ce.ids,Ae=Ce.vals),G){const L=ut(fe,Ae,a,G);fe=L.ids,Ae=L.vals}const Fe=u?jr(fe,Ae,p,$,B,j):fe[0];return Oe=re[O.indexOf(Fe)],Fe};ne=[];try{const M=performance.now(),z=await fr(e,n,le);if(!z)return{prefillMs:performance.now()-M,decodeMs:0,tokPerSec:0,tokens:[],firstArgmax:-1,recMs:0,gpuMs:0,rbMs:0,rng:$};const H=s.createCommandEncoder(),C=v(w);H.copyBufferToBuffer(z.fn,z.lastRow*w*4,C,0,w*4);const O=J(a);let re=H.beginComputePass();Nn(re,C,1,F),Ee(re,O.affLen,O.banLen),re.end(),ee(H),s.queue.submit([H.finish()]);const pe=await De(),fe=await Tn(pe.ci,pe.cv);Ie();const Ae=performance.now()-M,Ce=[];let Fe=ue?.has(fe)??!1;Fe||(Ce.push(fe),a.push(fe),Ue(pe.ci,pe.cv,pe.lse),Z?.(fe),s.queue.writeBuffer(oe,0,new Uint32Array([fe])));let L=0,Be=0,_e=0;const nn=performance.now();let mn=1;nr();let Pn=n+e.length;for(;mn<t&&!Fe&&!le?.aborted;){Mn("decode"),Pn=Jr(Pn,1);const me=mn,Ge=Pn;let $e=performance.now();const{affLen:Un,banLen:Wr}=J(a),Zn=s.createCommandEncoder();let Jn=Zn.beginComputePass();Re(Jn,"embed_gather",[["u",w],["u",me-1],["u",0],["u",0]],[oe,Hn,Pr,En,An],He,1),Jn.end();const wa=Bn(Zn,He,1,Ge),Gt=v(w);Zn.copyBufferToBuffer(wa.fn,0,Gt,0,w*4),Jn=Zn.beginComputePass(),Nn(Jn,Gt,1,F),Ee(Jn,Un,Wr),Jn.end(),ee(Zn),s.queue.submit([Zn.finish()]),L+=performance.now()-$e,$e=performance.now();const{ci:Pt,cv:zt,lse:va}=await De();Be+=performance.now()-$e,$e=performance.now();const hr=await Tn(Pt,zt);if(_e+=performance.now()-$e,mn+=1,ue?.has(hr)){Fe=!0;break}Ce.push(hr),a.push(hr),Ue(Pt,zt,va),Z?.(hr),s.queue.writeBuffer(oe,me*4,new Uint32Array([hr])),Pn+=1}Ne=Pn;const Ar=performance.now()-nn,Qe=Math.max(1,Ce.length-1);return{prefillMs:Ae,decodeMs:Ar,tokPerSec:Qe/(Ar/1e3),tokens:Ce,firstArgmax:fe,recMs:L/Qe,gpuMs:Be/Qe,rbMs:_e/Qe,rng:$,...Te?{lp:Te}:{}}}finally{Mn(null),Ie(),ne=null;for(const M of[oe,F,we,qe,ce,ye,de,W,I,He])M.destroy()}}async function at(e,n,t,r,a,o){await kn(n+e.length+t);const u=r.temperature!=null&&r.temperature>0&&r.temperature!==1,i=Q.lm_head.N,l=Math.max(1,Math.min(r.topK??20,i)),f=r.temperature??1,m=r.repetitionPenalty??1,p=r.presencePenalty??0,h=r.topP??1,D=r.minP??0,B=r.noRepeatNgramSize??0,j=typeof r.promptLookup=="object"&&r.promptLookup!==null?r.promptLookup:{},ae=Math.max(2,j.ngramSize??3),G=Math.max(1,Math.min(j.maxDraft??8,31)),R=r.stopTokens?new Set(r.stopTokens):null,ue=r.onToken,Z=r.signal,le=o??new Rt(r.seed),$=u||m!==1||p!==0||B>0,oe=s.createBuffer({size:i*4,usage:y|P|q}),F=s.createBuffer({size:(G+1)*i*4,usage:y|P}),we=s.createBuffer({size:(G+1)*4,usage:y|P}),qe=s.createBuffer({size:l*4,usage:y|P}),ce=s.createBuffer({size:l*4,usage:y|P});let ye=s.createBuffer({size:(E+G+1)*4,usage:y|q}),ie=s.createBuffer({size:(E+G+1)*4,usage:y|q});const de=(ee,U)=>U<=ee.size?ee:(ee.destroy(),s.createBuffer({size:1<<32-Math.clz32(U-1),usage:y|q})),W=s.createBuffer({size:(G+1)*l*8,usage:GPUBufferUsage.MAP_READ|q}),ge=s.createBuffer({size:(G+1)*4,usage:y|q}),I=s.createBuffer({size:(G+1)*w*4,usage:y|q}),He=ee=>{const U=m!==1||p!==0?it(ee):new Uint32Array(0);U.length&&s.queue.writeBuffer(ye=de(ye,U.byteLength),0,U);const De=B>0?st(ee,B):[];return De.length&&s.queue.writeBuffer(ie=de(ie,De.length*4),0,Uint32Array.from(De)),{affLen:U.length,banLen:De.length}},J=(ee,U,De)=>{je(ee,"sampler_penalty",[["u",U],["u",De],["f",m],["u",4286578688],["f",p]],[ye,ie],[oe]),ee.dispatchWorkgroups(1);for(let Se=0;Se<l;Se++)je(ee,"argmax_masked",[["u",i],["u",Se],["u",0],["u",0]],[oe],[qe,ce]),ee.dispatchWorkgroups(1)},Ee=(ee,U)=>u?jr(new Uint32Array(ee,U*l*8,l),new Float32Array(ee,U*l*8+l*4,l),f,le,h,D):new Uint32Array(ee,U*l*8,1)[0];ne=[];try{const ee=performance.now(),U=await fr(e,n,Z);if(!U)return{prefillMs:performance.now()-ee,decodeMs:0,tokPerSec:0,tokens:[],firstArgmax:-1,recMs:0,gpuMs:0,rbMs:0,spec:{steps:0,drafted:0,accepted:0},rng:le};const De=s.createCommandEncoder(),Se=v(w);De.copyBufferToBuffer(U.fn,U.lastRow*w*4,Se,0,w*4);const Oe=$?He(a):null;let Te=De.beginComputePass();Nn(Te,Se,1,oe),Oe?J(Te,Oe.affLen,Oe.banLen):Re(Te,"argmax",[["u",i],["u",0],["u",0],["u",0]],[oe],we,1),Te.end(),Oe&&(De.copyBufferToBuffer(qe,0,W,0,l*4),De.copyBufferToBuffer(ce,0,W,l*4,l*4)),s.queue.submit([De.finish()]),await s.queue.onSubmittedWorkDone();let Ue;if(Oe){await W.mapAsync(GPUMapMode.READ);const L=W.getMappedRange().slice(0);W.unmap(),Ue=Ee(L,0)}else Ue=(await cr(we,1))[0];Ie();const gr=performance.now()-ee,gn=[];let Tn=R?.has(Ue)??!1;Tn||(gn.push(Ue),a.push(Ue),ue?.(Ue)),nr();let mr=1,M=Ue,z=n+e.length,H=0,C=0,O=0,re=0,pe=0,fe=0;const Ae=performance.now();for(;mr<t&&!Tn&&!Z?.aborted;){z=Jr(z,G+1);const L=Math.min(G,t-mr-1,E-1-z),Be=L>0?ya(a,ae,L):[],_e=Be.length+1;await kn(z+_e);let nn=performance.now();_e===1?Mn("pld1"):xe&&_e<=9?Mn("pldm",_e,9):Mn(null),s.queue.writeBuffer(ge,0,new Uint32Array([M,...Be])),Xn=xe&&_e>=2&&_e<=9?_e:0;const mn=s.createCommandEncoder(),Pn=mn.beginComputePass();ze(Pn,"embed_gather_batch",[["u",_e],["u",w],["u",0],["u",0]],[ge,Hn,Pr,En,An],I,_e*w),Pn.end();const Ar=Bn(mn,I,_e,z);if(Te=mn.beginComputePass(),Nn(Te,Ar.fn,_e,F),Te.end(),Xn=0,$){s.queue.submit([mn.finish()]);for(let me=0;me<_e;me++){const{affLen:Ge,banLen:$e}=He(me===0?a:[...a,...Be.slice(0,me)]),Un=s.createCommandEncoder();Un.copyBufferToBuffer(F,me*i*4,oe,0,i*4);const Wr=Un.beginComputePass();J(Wr,Ge,$e),Wr.end(),Un.copyBufferToBuffer(qe,0,W,me*l*8,l*4),Un.copyBufferToBuffer(ce,0,W,me*l*8+l*4,l*4),s.queue.submit([Un.finish()])}}else{for(let me=0;me<_e;me++){mn.copyBufferToBuffer(F,me*i*4,oe,0,i*4);const Ge=mn.beginComputePass();Re(Ge,"argmax",[["u",i],["u",me],["u",0],["u",0]],[oe],we,1),Ge.end()}s.queue.submit([mn.finish()])}re+=performance.now()-nn,nn=performance.now(),await s.queue.onSubmittedWorkDone(),pe+=performance.now()-nn,nn=performance.now();const Qe=[];if($){await W.mapAsync(GPUMapMode.READ);const me=W.getMappedRange().slice(0);W.unmap();for(let Ge=0;Ge<_e;Ge++){const $e=Ee(me,Ge);if(R?.has($e)){Tn=!0;break}if(Qe.push($e),Ge<Be.length&&$e!==Be[Ge])break}}else{const me=await cr(we,_e);for(let Ge=0;Ge<_e;Ge++){const $e=me[Ge];if(R?.has($e)){Tn=!0;break}if(Qe.push($e),Ge<Be.length&&$e!==Be[Ge])break}}fe+=performance.now()-nn,H++,C+=Be.length,O+=Math.max(0,Qe.length-1);for(const me of Qe)gn.push(me),a.push(me),ue?.(me);if(mr+=Qe.length,Ie(),Qe.length===0)break;z+=Qe.length,M=Qe[Qe.length-1]}Ne=z;const Ce=performance.now()-Ae,Fe=Math.max(1,gn.length-1);return{prefillMs:gr,decodeMs:Ce,tokPerSec:Fe/(Ce/1e3),tokens:gn,firstArgmax:Ue,recMs:re/Fe,gpuMs:pe/Fe,rbMs:fe/Fe,spec:{steps:H,drafted:C,accepted:O},rng:le}}finally{Xn=0,Mn(null),Ie(),ne=null;for(const ee of[oe,F,we,qe,ce,ye,ie,W,I,ge])ee.destroy()}}async function pa(e){Ke=[],Ne=0,await kn(e.length+1),ne=[];const n=s.createCommandEncoder();Bn(n,ar(n,e),e.length,0),s.queue.submit([n.finish()]),await s.queue.onSubmittedWorkDone();const t=e.length,r=e[e.length-1],a=async o=>{Qn=o,Ln={};const u=s.createCommandEncoder(),i=Bn(u,ar(u,[r]),1,t),l=s.createBuffer({size:Q.lm_head.N*4,usage:y|P});ne?.push(l);const f=u.beginComputePass();Nn(f,i.fn,1,l),f.end(),s.queue.submit([u.finish()]),await s.queue.onSubmittedWorkDone();const m={};for(const[h,D]of Object.entries(Ln))m[h]=await Je(D,D.size/4);const p=t*_*g;if(!on&&!be&&cn.length){const h=cn[0];m.kc=(await Je(Pe[h],ln*_*g)).slice(p,p+_*g),m.vc=(await Je(Le[h],ln*_*g)).slice(p,p+_*g)}return m.fn=await Je(i.fn,w),m.logits=await Je(l,Q.lm_head.N),Qn=!1,Ln=null,m};try{return{fast:await a(!1),slow:await a(!0)}}finally{Qn=!1,Ln=null,Ie(),ne=null}}async function fa(e,n){Ke=[],Ne=0,ne=[];const t=Q.lm_head.N,r=Math.max(1,Math.min(n.topK??20,t)),a=n.repetitionPenalty??1,o=n.presencePenalty??0,u=n.noRepeatNgramSize??0,i=s.createBuffer({size:t*4,usage:y|P}),l=s.createBuffer({size:r*4,usage:y|P}),f=s.createBuffer({size:r*4,usage:y|P});ne?.push(i,l,f);const m=a!==1||o!==0?it(e):new Uint32Array(0),p=u>0?st(e,u):[],h=Nr(m.length?m:new Uint32Array(1),y|q),D=Nr(p.length?Uint32Array.from(p):new Uint32Array(1),y|q),B=s.createCommandEncoder(),{fn:j}=Bn(B,ar(B,e),e.length,0),ae=s.createBuffer({size:w*4,usage:y|P|q});ne?.push(ae),B.copyBufferToBuffer(j,(e.length-1)*w*4,ae,0,w*4);let G=B.beginComputePass();Nn(G,ae,1,i),G.end(),s.queue.submit([B.finish()]),await s.queue.onSubmittedWorkDone();const R=await Je(i,t),ue=s.createCommandEncoder();G=ue.beginComputePass(),je(G,"sampler_penalty",[["u",m.length],["u",p.length],["f",a],["u",4286578688],["f",o]],[h,D],[i]),G.dispatchWorkgroups(1);for(let Z=0;Z<r;Z++)je(G,"argmax_masked",[["u",t],["u",Z],["u",0],["u",0]],[i],[l,f]),G.dispatchWorkgroups(1);G.end(),s.queue.submit([ue.finish()]),await s.queue.onSubmittedWorkDone();try{return{base:R,penalized:await Je(i,t),candIds:await cr(l,r),candVals:await Je(f,r)}}finally{Ie(),ne=null}}const Er={useSubgroups:xe,subgroupSize:Ve,kvCache:on?"f16":be?"q8":"f32",activation:er?"f16":"f32",overflow:ve?"sinks":"error",maxSeqLen:E,adapter:{vendor:Cn.vendor,architecture:Cn.architecture,device:Cn.device,description:Cn.description},limits:{maxStorageBufferBindingSize:Number(s.limits.maxStorageBufferBindingSize),maxComputeWorkgroupStorageSize:s.limits.maxComputeWorkgroupStorageSize},timestampQuery:vr};async function ga(e,n={}){const t=n.temperature!=null&&n.temperature>0&&n.temperature!==1,r=(n.repetitionPenalty??1)!==1||(n.noRepeatNgramSize??0)>0||(n.presencePenalty??0)!==0||(n.dryMultiplier??0)>0;if(((n.dryMultiplier??0)>0||(n.topNSigma??0)>0)&&n.promptLookup&&n.promptLookup!=="auto")throw new Error("bitgpu: dryMultiplier/topNSigma are not supported with promptLookup (they need per-position statistics; auto simply disables lookup)");if(c.hybrid&&n.promptLookup&&n.promptLookup!=="auto")throw new Error("bitgpu: promptLookup is not supported on the qwen3_5 hybrid backbone (rejected drafts would corrupt the linear-attention recurrent state); use promptLookup: 'auto' or omit it");const a=(n.reuseCache??!1)&&Ke.length>0;if(n.signal?.aborted)return{tokens:[],prefillMs:0,decodeMs:0,tokensPerSecond:0,timing:{recordMs:0,gpuMs:0,readbackMs:0}};let o=a?ve?Ne:Ke.length-1:0;const u=a?[Ke[Ke.length-1],...e]:e,i=a?Ke:[...e];if(u.length===0)throw new Error("generate: no tokens to process");if(ve&&o+u.length+1>E){if(Xe+u.length+1>E)throw new Error(`generate: prompt length ${u.length} exceeds the rolling window (maxSeqLen ${E} minus ${Xe} sinks); trim the prompt`);await kn(Math.min(E,o+u.length)),o=Nt(o,o+u.length+1-E),Ne=o}const l=E-o-u.length;if(l<1)throw new Error(`generate: prompt length ${o+u.length} exceeds maxSeqLen ${E}; trim history or raise maxSeqLen`);const f=ve?n.maxTokens??256:Math.min(n.maxTokens??256,l);a?i.push(...e):Ke=i;try{if(f<1){await kn(o+u.length),ne=[];try{const h=performance.now();return u.length>1&&(await fr(u.slice(0,-1),o,n.signal),await s.queue.onSubmittedWorkDone()),Ne=o+u.length-1,{tokens:[],prefillMs:performance.now()-h,decodeMs:0,tokensPerSecond:0,timing:{recordMs:0,gpuMs:0,readbackMs:0}}}finally{Ie(),ne=null}}const m=!!n.candidateFilter||(n.logprobs??0)>0;let p;if(!m&&!c.hybrid&&(n.dryMultiplier??0)===0&&(n.topNSigma??0)===0&&n.promptLookup==="auto"&&f>24){const h=await at(u,o,24,n,i),D=h.tokens.length;if(D<24)p=h;else{const B=_a(D,h.spec?.steps??0,t||r),j=[h.tokens[D-1]],ae=o+u.length+D-1,G=f-D;let R;B?R=await at(j,ae,G,n,i,h.rng):t||r?R=await St(j,ae,G,n,i,h.rng):(R=await tt(j,ae,G,null,br,{stopTokens:n.stopTokens,onToken:n.onToken,signal:n.signal}),i.push(...R.tokens));const ue=D+R.tokens.length,Z=h.decodeMs+R.prefillMs+R.decodeMs,le=Math.max(1,D-1),$=Math.max(0,R.tokens.length);p={prefillMs:h.prefillMs,decodeMs:Z,tokPerSec:Math.max(1,ue-1)/(Z/1e3),tokens:[...h.tokens,...R.tokens],firstArgmax:h.firstArgmax,recMs:(h.recMs*le+R.recMs*$)/(le+$),gpuMs:(h.gpuMs*le+R.gpuMs*$)/(le+$),rbMs:(h.rbMs*le+R.rbMs*$)/(le+$),spec:{steps:(h.spec?.steps??0)+(R.spec?.steps??0),drafted:(h.spec?.drafted??0)+(R.spec?.drafted??0),accepted:(h.spec?.accepted??0)+(R.spec?.accepted??0),bailed:!B}}}}else!m&&!c.hybrid&&(n.dryMultiplier??0)===0&&(n.topNSigma??0)===0&&n.promptLookup?p=await at(u,o,f,n,i):t||r||m?p=await St(u,o,f,n,i):(p=await tt(u,o,f,null,br,{stopTokens:n.stopTokens,onToken:n.onToken,signal:n.signal}),i.push(...p.tokens));return{tokens:p.tokens,prefillMs:p.prefillMs,decodeMs:p.decodeMs,tokensPerSecond:p.tokPerSec,timing:{recordMs:p.recMs,gpuMs:p.gpuMs,readbackMs:p.rbMs},...p.spec?{speculation:p.spec}:{},...p.lp?{logprobs:p.lp}:{}}}catch(m){throw Ke=[],Ne=0,m}}async function ma(e){if(e.length===0)throw new Error("prefill: no tokens to process");if(e.length>E)throw new Error(`prefill: sequence length ${e.length} exceeds maxSeqLen ${E}`);Ke=[],Ne=0,await kn(e.length),ne=[];try{const n=performance.now();return e.length>1&&(await fr(e.slice(0,-1),0),await s.queue.onSubmittedWorkDone()),Ke=[...e],Ne=e.length-1,{prefillMs:performance.now()-n}}finally{Ie(),ne=null}}const pn=_*g*qn,fn=be?_*(g/32)*4:0,ot=e=>(c.hybrid?cn.length:c.layers)*2*e*(pn+fn)+ir.length*(In+On);async function ha(e){if(Ke.length===0)return null;const n=ve?Ne:Ke.length-1,t=Math.max(0,Math.min(Math.floor(e?.from??0),n));if(t>0&&ve)throw new Error("saveCache: delta snapshots ({ from }) are not supported under overflow 'sinks'");if(t>0&&c.hybrid)throw new Error("saveCache: delta snapshots ({ from }) are not supported for the qwen3_5 hybrid backbone");const r=n-t,a=ot(r),o=new ArrayBuffer(a);if(a>0){const u=Math.max(4,Math.min((globalThis.__RBCAP??0)||134217728,s.limits.maxBufferSize)&-4),i=[];if(r>0)for(const p of cn)i.push({src:Pe[p],off:t*pn,size:r*pn}),i.push({src:Le[p],off:t*pn,size:r*pn}),be&&(i.push({src:Ze[p],off:t*fn,size:r*fn}),i.push({src:tn[p],off:t*fn,size:r*fn}));for(const p of ir)i.push({src:sr[p][lr],off:0,size:In}),i.push({src:ur[p][lr],off:0,size:On});const l=s.createBuffer({size:Math.min(a,u),usage:GPUBufferUsage.MAP_READ|q});let f=0,m=0;for(let p=0;p<a;){const h=Math.min(u,a-p),D=s.createCommandEncoder();for(let B=0;B<h;){const j=i[f],ae=Math.min(j.size-m,h-B);D.copyBufferToBuffer(j.src,j.off+m,l,B,ae),B+=ae,m+=ae,m===j.size&&(f++,m=0)}s.queue.submit([D.finish()]),await l.mapAsync(GPUMapMode.READ,0,h),new Uint8Array(o,p,h).set(new Uint8Array(l.getMappedRange(0,h))),l.unmap(),p+=h}l.destroy()}return{version:ve?2:1,kvCache:Er.kvCache,model:{layers:c.layers,kvHeads:_,headDim:g,hidden:w,vocab:c.vocab},ids:[...Ke],...t>0?{base:t}:{},...ve?{roll:{sinkTokens:Xe,cacheLen:Ne}}:{},data:o}}async function ba(e){if(!e||e.version!==1&&e.version!==2)throw new Error(`restoreCache: unsupported snapshot version ${e?.version}`);if(e.version===2!==ve)throw new Error(e.version===2?"restoreCache: snapshot was saved under overflow 'sinks' (unroped keys); this engine runs overflow 'error'":"restoreCache: snapshot was saved under overflow 'error' (roped keys); this engine runs overflow 'sinks'");if(e.version===2&&e.roll?.sinkTokens!==Xe)throw new Error(`restoreCache: snapshot uses ${e.roll?.sinkTokens} sink tokens but this engine uses ${Xe}`);if(e.kvCache!==Er.kvCache)throw new Error(`restoreCache: snapshot was saved under kvCache '${e.kvCache}' but this engine runs '${Er.kvCache}' - snapshots do not convert across modes`);const n=e.model;if(!n||n.layers!==c.layers||n.kvHeads!==_||n.headDim!==g||n.hidden!==w||n.vocab!==c.vocab)throw new Error("restoreCache: snapshot is from a different model (architecture mismatch)");if(!Array.isArray(e.ids)||e.ids.length===0)throw new Error("restoreCache: snapshot holds no tokens");const t=e.version===2?e.roll.cacheLen:e.ids.length-1,r=Math.max(0,Math.floor(e.base??0));if(r>0&&c.hybrid)throw new Error("restoreCache: delta snapshots are not supported for the qwen3_5 hybrid backbone");const a=t-r;if(t+(e.version===2?0:1)>E)throw new Error(`restoreCache: snapshot needs ${t+(e.version===2?0:1)} cache slots but maxSeqLen is ${E}`);if(e.data.byteLength!==ot(a))throw new Error(`restoreCache: snapshot data is ${e.data.byteLength} bytes, expected ${ot(a)}`);if(r>0){if(Ne!==r)throw new Error(`restoreCache: delta snapshot expects the cache at position ${r} (prewarm the shared prefix first); it is at ${Ne}`);for(let u=0;u<r;u++)if(Ke[u]!==e.ids[u])throw new Error(`restoreCache: delta snapshot prefix does not match the current prewarm (token ${u})`)}await kn(t);let o=0;if(a>0)for(const u of cn)s.queue.writeBuffer(Pe[u],r*pn,e.data,o,a*pn),o+=a*pn,s.queue.writeBuffer(Le[u],r*pn,e.data,o,a*pn),o+=a*pn,be&&(s.queue.writeBuffer(Ze[u],r*fn,e.data,o,a*fn),o+=a*fn,s.queue.writeBuffer(tn[u],r*fn,e.data,o,a*fn),o+=a*fn);for(const u of ir)s.queue.writeBuffer(sr[u][0],0,e.data,o,In),o+=In,s.queue.writeBuffer(ur[u][0],0,e.data,o,On),o+=On;c.hybrid&&(lr=0),Ke=[...e.ids],Ne=t}let Bt=Promise.resolve();const Gn=e=>(...n)=>{const t=Bt.then(()=>e(...n),()=>e(...n));return Bt=t.catch(()=>{}),t};return{generate:Gn(ga),prefill:Gn(ma),forward:Gn(da),saveCache:Gn(ha),restoreCache:Gn(ba),resetCache:Zt,capabilities:Er,lost:Ft,dispose:()=>s.destroy(),device:s,adapter:Me,profileDecode:Gn(async(e,n,t=null,r=br)=>{Ke=[],Ne=0,et=vr;try{return await tt(e,0,n,t,r)}finally{et=!1}}),debugDecode:Gn(pa),debugSampler:Gn(fa)}}export{Lr as GpuOutOfMemoryError,Vt as WebGPUUnavailableError,Ga as createEngine};
