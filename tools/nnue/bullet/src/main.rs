//! askaig NNUE trainer on bullet (Rust, Metal on Apple Silicon) — replaces the PyTorch/MPS
//! pipeline. Reproduces src/nnue.h EXACTLY so a bullet-trained net converts to AKNN v2 and
//! loads bit-for-bit:
//!
//!   (768 x 8 king buckets -> 512)x2 -> 1 of 8 material output buckets, SCReLU.
//!
//! ChessBucketsMirrored + MaterialCount::<8> give a feature index and output bucket that are
//! bit-identical to the engine's `idx = 768*bucket + 384*enemy + 64*pt + oriented(sq)` and
//! `(popcount(occ)-2)/4` — verified against bullet source — so no permutation is needed and the
//! converter (bullet_to_nnue.py) is a straight reshape + one transpose.
//!
//! Config is via env (all optional except ASKAIG_DATA):
//!   ASKAIG_DATA=a.bf[,b.bf]   shuffled bulletformat file(s)          (required)
//!   ASKAIG_SB=160             superbatches (LR cosine spans this)
//!   ASKAIG_BPS=6104           batches per superbatch (~100M positions)
//!   ASKAIG_BATCH=16384        batch size
//!   ASKAIG_LAMBDA=0.0         WDL blend: 0 = pure teacher eval, 0.2-0.4 with real results
//!   ASKAIG_LR=0.001           initial LR (cosine -> lr*0.3^5)
//!   ASKAIG_THREADS=4          data-loader (CPU decode) threads
//!   ASKAIG_ID=askaig          net id / checkpoint prefix
//!   ASKAIG_OUT=checkpoints    output directory

use bullet_lib::{
    game::{inputs::ChessBucketsMirrored, outputs::MaterialCount},
    nn::optimiser::AdamW,
    trainer::{
        save::SavedFormat,
        schedule::{lr, wdl, TrainingSchedule, TrainingSteps},
        settings::LocalSettings,
    },
    value::{loader::DirectSequentialDataLoader, ValueTrainerBuilder},
};

// Architecture constants — MUST match src/nnue.h.
const HL: usize = 512; // hidden per perspective
const IN_BUCKETS: usize = 8; // king buckets -> 768*8 = 6144 features/perspective
const OUT_BUCKETS: usize = 8; // material output buckets
const SCALE: f32 = 400.0; // eval scale (score_cp / SCALE)
const QA: i16 = 255; // FT quant (ft_w, ft_b)
const QB: i16 = 64; // output-weight quant

// Left half (files a-d) of the engine's symmetric KING_BUCKET[64], rank-major.
// ChessBucketsMirrored folds files e-h onto d-a via [0,1,2,3,3,2,1,0] and flips sq^7 for a
// king on files e-h — exactly the engine's mirror convention.
#[rustfmt::skip]
const KING_BUCKETS: [usize; 32] = [
    0, 1, 2, 3,
    4, 4, 5, 5,
    6, 6, 6, 6,
    6, 6, 6, 6,
    7, 7, 7, 7,
    7, 7, 7, 7,
    7, 7, 7, 7,
    7, 7, 7, 7,
];

fn env_or<T: std::str::FromStr>(key: &str, default: T) -> T {
    std::env::var(key).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

fn main() {
    let data_env = std::env::var("ASKAIG_DATA").expect("set ASKAIG_DATA=path1.bf[,path2.bf]");
    let paths: Vec<String> = data_env.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
    let path_refs: Vec<&str> = paths.iter().map(String::as_str).collect();

    let superbatches: usize = env_or("ASKAIG_SB", 160);
    let bps: usize = env_or("ASKAIG_BPS", 6104);
    let batch: usize = env_or("ASKAIG_BATCH", 16_384);
    let lambda: f32 = env_or("ASKAIG_LAMBDA", 0.0);
    let init_lr: f32 = env_or("ASKAIG_LR", 0.001);
    let threads: usize = env_or("ASKAIG_THREADS", 4);
    let net_id: String = std::env::var("ASKAIG_ID").unwrap_or_else(|_| "askaig".to_string());
    let out_dir: String = std::env::var("ASKAIG_OUT").unwrap_or_else(|_| "checkpoints".to_string());

    println!(
        "askaig-trainer: {} file(s), {} superbatches x {} batches x {} = {:.0}M pos/sb, lambda {}, lr {}",
        paths.len(),
        superbatches,
        bps,
        batch,
        (bps * batch) as f64 / 1e6,
        lambda,
        init_lr
    );

    let mut trainer = ValueTrainerBuilder::default()
        .dual_perspective()
        .optimiser(AdamW) // default clamps every param to [-1.98, 1.98] -> QA*w, QB*w fit i16
        .inputs(ChessBucketsMirrored::new(KING_BUCKETS))
        .output_buckets(MaterialCount::<OUT_BUCKETS>)
        // quantised.bin comes out in exact AKNN body order; we convert from raw.bin instead
        // (bullet_to_nnue.py, Option B) so export.py's quantization + overflow asserts stay the
        // single source of truth. This save_format also emits a ready-to-slice quantised.bin.
        .save_format(&[
            SavedFormat::id("l0w").round().quantise::<i16>(QA), // ft_w  [6144][512] i16
            SavedFormat::id("l0b").round().quantise::<i16>(QA), // ft_b  [512]       i16
            SavedFormat::id("l1w").round().quantise::<i16>(QB).transpose(), // out_w [8][1024] i16
            SavedFormat::id("l1b").round().quantise::<i32>(QA as i32 * QB as i32), // out_b [8] i32
        ])
        .loss_fn(|out, target| out.sigmoid().squared_error(target))
        .build(|builder, stm, ntm, buckets| {
            let l0 = builder.new_affine("l0", 768 * IN_BUCKETS, HL); // shared FT: 6144 -> 512
            let l1 = builder.new_affine("l1", 2 * HL, OUT_BUCKETS); // output: 1024 -> 8
            let stm_h = l0.forward(stm).screlu();
            let ntm_h = l0.forward(ntm).screlu();
            l1.forward(stm_h.concat(ntm_h)).select(buckets) // [stm(512), ntm(512)] -> this bucket
        });

    let schedule = TrainingSchedule {
        net_id,
        eval_scale: SCALE,
        steps: TrainingSteps {
            batch_size: batch,
            batches_per_superbatch: bps,
            start_superbatch: 1,
            end_superbatch: superbatches,
        },
        wdl_scheduler: wdl::ConstantWDL { value: lambda },
        lr_scheduler: lr::CosineDecayLR {
            initial_lr: init_lr,
            final_lr: init_lr * 0.3f32.powi(5),
            final_superbatch: superbatches,
        },
        save_rate: 10,
    };

    let settings =
        LocalSettings { threads, test_set: None, output_directory: out_dir.as_str(), batch_queue_size: 64 };

    let data = DirectSequentialDataLoader::new(&path_refs);
    trainer.run(&schedule, &settings, &data);

    println!("done -> {}/{}-{}/  (raw.bin, quantised.bin)", out_dir, "<id>", superbatches);
}
