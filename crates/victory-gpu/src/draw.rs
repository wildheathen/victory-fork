//! Draws on the host GPU (decision 0007, step 3). A list keeps the state
//! the title sets for its draws (pipeline, descriptor heaps, root
//! arguments, vertex and index buffers, targets); each draw then brings the
//! host list up to date: the host pipeline and our root signature, the
//! user data the translated shaders read, the host copies of the buffers
//! and descriptors they read, and the states of the textures involved.
//! State the host keeps across draws (viewports, scissors, blend factor,
//! stencil reference, topology, targets) is set on the host list as the
//! title sets it.

use std::sync::atomic::{AtomicU64, Ordering};

use victory_shader::resource::RootIndexedView;
use victory_shader::{ResourceType, RootType, UsedResource};

use crate::descriptor::View;
use windows::Win32::Foundation::RECT;
use windows::Win32::Graphics::Direct3D::D3D_PRIMITIVE_TOPOLOGY;
use windows::Win32::Graphics::Direct3D12::*;
use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
use windows::core::Interface;

use crate::host::HostList;
use crate::object::This;
use crate::pipeline::host::HostPipeline;

/// A root argument as the title set it.
#[derive(Debug, Clone, Default)]
enum Root {
    #[default]
    None,
    /// A descriptor table's first descriptor (an address in a heap).
    Table(u64),
    Constants(Vec<u32>),
    /// A root view's GPU address.
    View(u64),
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct VertexView {
    address: u64,
    size: u32,
    stride: u32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct IndexView {
    address: u64,
    size: u32,
    format: u32,
}

/// Host states of a depth target, and of a texture shaders read.
const DEPTH_WRITE: u32 = 0x10;
const DEPTH_READ: u32 = 0x20;
const SHADER_RESOURCE: u32 = 0x40 | 0x80;

thread_local! {
    /// While a draw is prepared (`State::bind`, then its targets' states):
    /// the depth target the draw only tests (0: none, or one it writes),
    /// and whether a view its shaders read is of that same texture.
    static TESTED_DEPTH: std::cell::Cell<(usize, bool)> =
        const { std::cell::Cell::new((0, false)) };
    /// While a stage's tables are walked (`State::tables`): whether that
    /// stage's shader can store to buffers (`Translated::stores`).
    static STORES: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
    /// While `State::early_textures` runs the texture pass: the
    /// milliseconds and the bytes that run may take, in place of the
    /// pass's own per frame.
    static PASS_BUDGET: std::cell::Cell<Option<(u64, u64)>> = const { std::cell::Cell::new(None) };
    /// For the run of the texture pass `State::bake_sources_pass` makes
    /// before a draw of the title's skin bake: the title slots that draw
    /// samples and that wait in the queue. That run serves them alone,
    /// whatever their turn, and none of the queue's upkeep. Set and taken
    /// on the thread that records the draw only: the worker's passes
    /// (`crate::heap_worker`) never see it.
    static PASS_SOURCES: std::cell::RefCell<Option<Vec<u32>>> = const { std::cell::RefCell::new(None) };
    /// What that run did with them: uploaded, their bytes, left because
    /// their memory is not mapped, and left because no more of their
    /// memory is mapped than at their last upload.
    static PASS_SOURCED: std::cell::Cell<[u64; 4]> = const { std::cell::Cell::new([0; 4]) };
    /// The tables a pixel shader reads as a skin bake does
    /// ([`bake_tables`]), by the shader's hash.
    static BAKE_TABLES: std::cell::RefCell<std::collections::HashMap<String, Option<BakeTables>>> =
        std::cell::RefCell::new(std::collections::HashMap::new());
}

/// The milliseconds of every frame the pass that uploads textures picked
/// by index takes before a shader that picks by index has asked for it
/// (`VICTORY_EARLY_TEXTURES_MS`, 12; 0: never, as before), and the
/// milliseconds it may take just before a draw of the title's decoration
/// bake (`VICTORY_BAKE_TEXTURES_MS`, 1500; 0: nothing special).
///
/// The pass ran only from the first world frame on, where it met some
/// 24,000 slots at once and needed 60-80 frames for them. The title bakes
/// its characters' skin and decorations into textures of its own while it
/// loads, 159-1,165 frames before that (runs 156-174), with a pixel shader
/// that picks its sources by index: whatever was not on the card then was
/// baked as nothing, once, for the run. On the console a texture is there
/// when its bytes are.
fn early_budgets() -> (u64, u64) {
    static MS: std::sync::OnceLock<(u64, u64)> = std::sync::OnceLock::new();
    *MS.get_or_init(|| {
        let ms = |name: &str, default: u64| {
            std::env::var(name).ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(default)
        };
        (ms("VICTORY_EARLY_TEXTURES_MS", 12), ms("VICTORY_BAKE_TEXTURES_MS", 1500))
    })
}

/// The pixel shaders of the title's decoration bake (the title:
/// `PS_BakeDecorations_*`), by the first 16 hex digits of their hash as the
/// frame listings print it. They draw into a 256x256 corner of 2048x2048
/// targets and name no texture in a table.
const BAKE_PIXEL_SHADERS: [&str; 4] =
    ["c0a7fe77b1862e10", "0d1575a396dfe513", "b7cfd8dda91ebb6b", "280967ad158cee0b"];

/// Where a pixel shader of the title's skin bake finds what it samples
/// (fork pull request #1264, head 4c14ca41): the root parameter of its one
/// table, and in that table the dwords of its buffer of decoration records
/// and of its constants.
///
/// The bake names no texture in a table. Its constants hold the view heap
/// indices of up to five textures (bytes 0x20, 0x30, 0x40, 0x50 and 0x60)
/// and the records it walks (first at 0x78, the one past the last at 0x7c);
/// a record is 304 bytes and holds five more indices (bytes 0, 0x10, 0x20,
/// 0x30 and 0x40). The same in all 322 bake programs but for the two table
/// offsets (by that pull request); an index counts only if the slot it
/// names holds a texture view.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct BakeTables {
    slot: u8,
    records: u32,
    constants: u32,
}

/// The stride of the bake's records.
const BAKE_RECORD: u64 = 304;
/// Where its constants and each of its records hold view heap indices.
const BAKE_CONSTANT_INDICES: [usize; 5] = [0x20, 0x30, 0x40, 0x50, 0x60];
const BAKE_RECORD_INDICES: [usize; 5] = [0, 0x10, 0x20, 0x30, 0x40];

/// [`BakeTables`] of pipeline `p`, if its pixel shader reads one table
/// with a buffer of 304-byte records and constants and nothing else
/// through tables.
fn bake_tables(p: &HostPipeline) -> Option<BakeTables> {
    let stage = p.stages.iter().find(|s| s.shader.stage == "pixel")?;
    let read = || {
        let mut tabled = stage.translated().used.iter().filter(|u| u.root == RootType::Table);
        let (a, b) = (tabled.next()?, tabled.next()?);
        if tabled.next().is_some() {
            return None;
        }
        let (records, constants) = if a.kind == ResourceType::Srv { (a, b) } else { (b, a) };
        (records.kind == ResourceType::Srv
            && u64::from(records.stride) == BAKE_RECORD
            && constants.kind == ResourceType::Cbv
            && records.slot == constants.slot)
            .then_some(BakeTables {
                slot: records.slot,
                records: records.table_offset_dwords,
                constants: constants.table_offset_dwords,
            })
    };
    BAKE_TABLES.with_borrow_mut(|known| match known.get(stage.shader.hash.as_str()) {
        Some(&tables) => tables,
        None => {
            let tables = read();
            known.insert(stage.shader.hash.clone(), tables);
            tables
        }
    })
}

/// The view heap indices a bake draw names, in order and each once, and
/// the records it walks with how many of them were read: from its
/// constants, and from the records `first..end` they give and the one
/// after them, which the program looks at as well; no more records than
/// the buffer holds (`held`), and no more than 256. `record` reads the
/// start of one.
fn bake_named(
    constants: &[u8; 0x80],
    held: u32,
    mut record: impl FnMut(u32) -> Option<[u8; 0x44]>,
) -> (Vec<u32>, (u32, u32)) {
    let word = |bytes: &[u8], at: usize| u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
    let mut named: Vec<u32> = BAKE_CONSTANT_INDICES.iter().map(|&at| word(constants, at)).collect();
    let (first, end) = (word(constants, 0x78), word(constants, 0x7c));
    let mut read = 0;
    // (No record is looked at where none is walked.)
    let last = if end > first { end.saturating_add(1) } else { first };
    for i in first..last.min(held).min(first.saturating_add(256)) {
        let Some(record) = record(i) else {
            break;
        };
        named.extend(BAKE_RECORD_INDICES.iter().map(|&at| word(&record, at)));
        read += 1;
    }
    named.sort_unstable();
    named.dedup();
    (named, (end.saturating_sub(first), read))
}

/// What is done for a skin-bake draw's own textures
/// (`VICTORY_BAKE_SOURCES`): unset or anything else, nothing (no draw is
/// even looked at); `count`, the textures each such draw samples are
/// counted, and those it samples while they still wait for their upload
/// are logged; `1`, those are also uploaded before the draw, on the thread
/// that records it, up to `VICTORY_BAKE_SOURCES_MB` (64) megabytes a draw,
/// and only while the card has room for textures (`resource::room`).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum BakeMode {
    Off,
    Count,
    Upload,
    /// [`BakeMode::Upload`] also while the card has no room
    /// (`VICTORY_BAKE_SOURCES=full`), as fork pull request #1264 had it:
    /// at the boat save every such upload was skipped for room (run 316: 55
    /// of 55), the card being at its line through the whole world. Still
    /// within `VICTORY_BAKE_SOURCES_MB` a draw.
    Full,
}

/// [`BakeMode`], and the bytes one draw may have uploaded.
fn bake_sources_mode() -> (BakeMode, u64) {
    static MODE: std::sync::OnceLock<(BakeMode, u64)> = std::sync::OnceLock::new();
    *MODE.get_or_init(|| {
        let mode = match std::env::var("VICTORY_BAKE_SOURCES").ok().as_deref().map(str::trim) {
            Some("count") => BakeMode::Count,
            Some("1") => BakeMode::Upload,
            Some("full") => BakeMode::Full,
            _ => BakeMode::Off,
        };
        let mb = std::env::var("VICTORY_BAKE_SOURCES_MB").ok().and_then(|v| v.trim().parse::<u64>().ok()).unwrap_or(64);
        if mode != BakeMode::Off {
            tracing::info!(?mode, megabytes_a_draw = mb, "skin-bake draws' own textures (VICTORY_BAKE_SOURCES)");
        }
        (mode, mb << 20)
    })
}

/// What a bake draw samples ([`State::bake_sources`]).
#[derive(Default, Debug)]
struct BakeSources {
    /// The records the draw walks, and how many of them were read.
    records: (u32, u32),
    /// Slots named that hold a texture view, and their handles.
    textures: Vec<(u32, u64)>,
    /// Of them: with their host texture made from that view and nothing
    /// waiting.
    ready: u32,
    /// Waiting in the queue of textures picked by index.
    queued: Vec<u32>,
    /// Neither, and a texture the title placed in memory that is there:
    /// the host slot shows nothing, and nothing would make it (the title
    /// wrote a newer view of memory this one clashes with, which made this
    /// one stale to the layer; the bake reads it all the same).
    unmade: Vec<u32>,
    /// Neither, and something else: a texture the host itself writes.
    other: u32,
}

/// Since the start: early runs of the texture pass, and those of them
/// made for a bake draw.
pub(crate) static EARLY_TEXTURE_PASSES: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Whether a buffer range a shader names through an unordered access view
/// is given to shaders as written (snapshot before the submission, compare
/// and write back after) only if that shader's code can store to buffers.
/// The title binds buffers its shaders only read through such views, and
/// every range declared costs a snapshot and a compare of its bytes: 3.97
/// GB a frame in 31 submissions, 0.49 s of snapshots and 0.28 s of
/// comparing (run 28). On by default; `VICTORY_WRITE_SKIP=0` declares
/// every such range, as before.
fn write_skip() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_WRITE_SKIP").is_none_or(|v| v != "0"))
}

/// A run of indirect records clamped in one pass ([`windows`]): the host
/// copy they came from (its number), their kind and stride, the offset of
/// the first in that copy, how many, and the list's scratch buffer that
/// holds them clamped, record `i` at `i * step`.
#[derive(Clone)]
struct Window {
    copy: u64,
    kind: u32,
    stride: u32,
    /// How far apart the records are in the copy, and in the scratch
    /// buffer: what the title walks them by ([`windows`]).
    step: u32,
    base: u64,
    records: u32,
    scratch: ID3D12Resource,
}

/// The most records clamped in one pass ([`windows`]).
const WINDOW_MOST: u64 = 64;

/// Whether indirect draws that read record after record of one argument
/// buffer are clamped a window at a time (`VICTORY_INDIRECT_WINDOW=1`; off
/// by default).
///
/// The title issues its mesh draws one record per call, each from the next
/// slot of the same argument buffer: 3,460 a frame in gameplay, and before
/// each our clamp ran as a compute dispatch of its own between two waits,
/// so the card went from drawing to compute and back for every record. In
/// the timing run of the lit scene (run 79) a shadow draw of next to
/// nothing cost 14-16 microseconds, 4,126 records a frame.
///
/// With the switch, when a draw asks for a record a short step after the
/// one drawn before, the clamp takes that record and the ones that step
/// apart after it (16, then twice as many each time, [`WINDOW_MOST`] at
/// most, never past the buffer) into
/// the list's scratch buffer in one pass, and the draws that follow read
/// their record there with no pass of ours and no barrier (the scratch
/// buffer stays readable as arguments until the next clamp), until one of
/// these: a compute
/// dispatch, a direct draw, an indirect call with a count buffer or more
/// than one record, a draw whose shaders can store to buffers, a record
/// outside the window, another host list.
///
/// The step is the one the title is seen to take, not the stride its
/// command signature names: it draws one record a call, for which the
/// stride means nothing, and the title names 12 bytes (the three
/// words of a mesh dispatch) for records it keeps 32 bytes apart (2,173 of
/// a frame's 3,349 mesh records in run 87's listed frame; 283 are 16
/// apart, which ends a window). Walked by the named stride the window
/// never opened: run 87, with the switch, still clamped 5,068 records a
/// frame one at a time.
///
/// What it costs the guard against garbage records: nothing against a
/// hang (every record is bounded as before and charged to the list's
/// budget as before), but the budget is charged when a window is clamped,
/// so also for records of the window's tail that are never drawn. Those
/// hold garbage in this title (slots of all ones: run 93 spent lists'
/// budgets on them and emptied the draws after), so in a window a record
/// that asks for more than can run is left empty and charges nothing
/// (`crate::indirect::bound_ahead`); a tail record that looks real is
/// charged what it asks, a few thousand of 2^27. And arguments the title's CPU side changes are seen as
/// before, at the next submission.
///
/// Off for the rest of the run from the first error the card's driver
/// reports to the system (`crate::drvwatch`): every loss of the card in a
/// run with windows was announced that way seconds ahead.
fn windows() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_INDIRECT_WINDOW").is_some_and(|v| v == "1")) && !crate::drvwatch::tripped()
}

/// Whether a record that is not on the window's walk opens a window of its
/// own at once, with the step the title was last seen to walk that
/// argument buffer by (`VICTORY_WINDOW_JUMP=1`; `alt`: twenty seconds
/// with, twenty without, in turn, each change said in the log). Off by
/// default. Needs [`windows`].
///
/// Run 243 counted, per frame, 2,370 indirect commands read from a window
/// and 2,370 clamped by a pass of ours; 960 of those passes were for
/// records "off the window's walk": the title draws a few records one
/// after the other and then jumps to another place in the same argument
/// buffer, some 480 times a frame. The record after a jump was clamped
/// alone (its step is not known from one record), and the one after it
/// opened a window of sixteen: two passes a jump, each one a trip of the
/// card from drawing to compute and back (9 microseconds in the game).
///
/// With the switch the step of the last window that draws did read from
/// is remembered with its argument buffer, kind and stride, and a record
/// that would be clamped alone on that buffer opens sixteen records at
/// that step instead: what the next record would have opened anyway, one
/// record earlier, so no more of the list's budget is spent than before.
/// If no draw reads from a window opened that way before the next pass,
/// the step is forgotten until a walk shows it again: a title that draws
/// lone records all over a buffer gets what it had.
fn jump_windows() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_WINDOW_JUMP").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    }) {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), std::sync::atomic::Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "a record off the window's walk opens a window at the remembered step from now (VICTORY_WINDOW_JUMP=alt): {}",
                    if on { "ON" } else { "off, as usual" }
                );
            }
            on
        }
    }
}

/// Whether a single record that misses its window is gathered: given a
/// slot that the list's prelude fills in one pass with the list's other
/// such records, in place of a pass of ours in front of its draw
/// (`crate::indirect::gather`; `VICTORY_INDIRECT_GATHER=1`; `alt`: twenty
/// seconds with, twenty without, in turn, each change said in the log).
/// Off by default. Needs [`windows`].
fn gathers() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_INDIRECT_GATHER").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        _ => 0,
    }) {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), std::sync::atomic::Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "single indirect records are gathered for their list's prelude from now (VICTORY_INDIRECT_GATHER=alt): {}",
                    if on { "ON" } else { "off, as usual" }
                );
            }
            on
        }
    }
}

/// The records of a window opened at a remembered step ([`jump_windows`]):
/// what the record after a lone one opens.
const WINDOW_JUMP: u32 = 16;

/// What a record that would be clamped alone opens instead, if the step
/// its argument buffer is walked by is remembered: that step and how many
/// records (never past the buffer, of which `left` bytes are left from
/// the record on). None if nothing is remembered for this buffer, kind
/// and stride, or no second record fits.
fn jump_window(remembered: Option<(u64, u32, u32, u32)>, (copy, kind, stride): (u64, u32, u32), left: u64) -> Option<(u32, u32)> {
    let (c, k, s, step) = remembered?;
    if (c, k, s) != (copy, kind, stride) || step == 0 {
        return None;
    }
    let fit = (left / u64::from(step)).min(u64::from(WINDOW_JUMP)) as u32;
    (fit > 1).then_some((step, fit))
}

/// Diagnostics of [`windows`]: indirect commands since the last line, by
/// what became of their clamp. In order: read from a window (no pass of
/// ours); then clamped by a pass of ours because: it is a dispatch; it
/// asks for more than one record; it has a count buffer; its shaders can
/// store to buffers; it is the list's first; there was no window on its
/// argument buffer (another buffer, kind or stride, or the window was
/// ended); its record does not continue the window's walk; it is past the
/// window's end (the next window: as meant). And the records those passes
/// clamped in all; the passes that opened a window at a remembered step
/// ([`jump_windows`]), and those of them no draw read from; the commands
/// that read a gathered record ([`gathers`]: no pass of ours either), and
/// those that asked for one and got none (the list no longer gathers, or
/// its set is full).
#[cfg(feature = "diagnostics")]
pub(crate) static WINDOW_WHY: [std::sync::atomic::AtomicU64; 14] = [const { std::sync::atomic::AtomicU64::new(0) }; 14];

/// The line of [`WINDOW_WHY`], per frame over `frames` frames.
#[cfg(feature = "diagnostics")]
pub(crate) fn window_line(frames: f64) {
    let n: Vec<f64> = WINDOW_WHY.iter().map(|n| n.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames.max(1.0)).collect();
    let g: Vec<f64> = crate::indirect::GATHER_NONE.iter().map(|n| n.swap(0, std::sync::atomic::Ordering::Relaxed) as f64 / frames.max(1.0)).collect();
    tracing::info!(
        "indirect commands per frame by their clamp: {:.0} read from a window (no pass of ours); a pass of ours for {:.0}: {:.0} dispatches, {:.0} of more than one record, {:.0} with a count \
         buffer, {:.0} whose shaders can store to buffers, {:.0} first on their list, {:.0} with no window on their argument buffer, {:.0} off the window's walk, {:.0} past the window's end; \
         {:.0} records clamped by those passes; {:.0} of the passes opened a window at a remembered step (VICTORY_WINDOW_JUMP=1), {:.0} of those windows were read by no draw; \
         {:.0} read a record gathered for their list's prelude (VICTORY_INDIRECT_GATHER=1), {:.0} asked for one and got none          (no fit for the clamp {:.0}, the list had stopped gathering {:.0}, too many buffers {:.0}, the set full {:.0}, no buffer {:.0})",
        n[0],
        n[1..9].iter().sum::<f64>(),
        n[1],
        n[2],
        n[3],
        n[4],
        n[5],
        n[6],
        n[7],
        n[8],
        n[9],
        n[10],
        n[11],
        n[12],
        n[13],
        g[0],
        g[1],
        g[2],
        g[3],
        g[4],
    );
}

/// What one title list has set, and what its host list was last given.
#[derive(Default)]
pub(crate) struct State {
    pipeline: usize,
    /// View heap, sampler heap.
    heaps: [usize; 2],
    graphics: Vec<Root>,
    compute: Vec<Root>,
    vertex: [VertexView; 32],
    index: IndexView,
    /// Bound render target and depth resources, for their host states.
    targets: Vec<usize>,
    depth: usize,
    /// The host state the bound depth view needs ([`depth_state`]).
    depth_state: u32,
    /// The title's primitive topology (the host's may differ).
    topology: u32,
    /// What the title set that goes straight onto the host list, kept for
    /// the host list that continues this one after a cut ([`State::replay`]):
    /// whether a topology was set, the render target and depth handles, the
    /// viewports, the scissors, the blend factor and the stencil reference.
    topology_set: bool,
    target_handles: Vec<u64>,
    depth_handle: u64,
    targets_set: bool,
    /// The formats of the views [`State::set_targets`] bound on the host
    /// list: a draw's pipeline is one for them
    /// (`crate::pipeline::host_pipeline_for`).
    target_formats: crate::pipeline::host::TargetFormats,
    viewports: Vec<D3D12_VIEWPORT>,
    scissors: Vec<RECT>,
    blend: Option<[f32; 4]>,
    stencil: Option<u32>,
    depth_bounds: Option<[f32; 2]>,
    /// The indirect records clamped last, if later draws may still read
    /// them ([`windows`]).
    window: Option<Window>,
    /// With [`jump_windows`]: the argument buffer (its copy's number), kind
    /// and stride of the last window of several records that a draw read
    /// from, and that window's step.
    walk_step: Option<(u64, u32, u32, u32)>,
    /// Whether [`State::window`] was opened at the remembered step, and
    /// whether a draw has read from it since.
    window_jumped: bool,
    window_read: bool,
    /// A direct dispatch spread across z, for [`State::host_word`] at its
    /// bind: `[row, asked x]` (`[0, 0]`: not spread; an indirect one gets
    /// its words from the clamp through its command signature).
    dispatch_spread: [u32; 2],
    /// A `DispatchX` that counts threads, for [`State::host_word`] at its
    /// bind: the count per axis where it is not a whole number of groups,
    /// else 0 (`USER_DISPATCH_THREADS`; fork pull request #646).
    dispatch_threads: [u32; 3],
    /// Diagnostics: the argument buffer and the offset of the indirect
    /// command before this one on the list.
    #[cfg(feature = "diagnostics")]
    last_indirect: (u64, u64),
    /// What the host list has: pipeline, root signature, heaps.
    host_pipeline: usize,
    /// Graphics, compute.
    host_roots: [usize; 2],
    host_heaps: [usize; 2],
    /// Tables made afresh in this list: (table, slot count, contents) → first
    /// dynamic slot.
    fresh_tables: std::collections::HashMap<(u64, u32, u64), u32>,
    /// The dynamic slots the list holds: title heap, that heap's identity
    /// (`HostHeap::id`), first, count; given back when the list has run, is
    /// reset or released.
    dynamic: Vec<(usize, u64, u32, u32)>,
    /// Root SRVs bound as one-view tables in this list: (title heap, title
    /// address; 0 for none) → dynamic slot, and the host copy the slot
    /// shows: later draws of the list bring that one up to date, whatever
    /// copy the address would find by then (`shadow.rs` `tails`).
    root_srv_slots: std::collections::HashMap<(usize, u64), RootSlot>,
    /// Root UAVs bound as one-view tables in this list, likewise.
    root_uav_slots: std::collections::HashMap<(usize, u64), RootSlot>,
    /// The list ran and its dynamic slots were given back.
    tables_gone: bool,
    /// Diagnostics: the number of this recording of the list, by which the
    /// tables it makes twice are told ([`table_asked`]); 0 until its first.
    #[cfg(feature = "diagnostics")]
    recording: u64,
}

impl Drop for State {
    fn drop(&mut self) {
        for &(heap, id, first, n) in &self.dynamic {
            crate::descriptor::release_dynamic(heap, id, first, n);
        }
    }
}

impl State {
    /// The list's submission has run (submissions are waited for): the
    /// dynamic slots its tables were made in are given back, not kept
    /// until the title resets the list. A frame of the opening cutscene
    /// asks for 12,500 tables; held until the resets they ran the 131,072
    /// slots out, and up to 4,900 draws a frame went without their tables.
    /// shortcut: a list the title runs a second time without a reset then
    /// reads slots others may have taken (logged once).
    pub(crate) fn tables_done(&mut self) {
        if self.tables_gone && self.dynamic.is_empty() {
            return;
        }
        if self.tables_gone {
            static ONCE: std::sync::Once = std::sync::Once::new();
            ONCE.call_once(|| tracing::warn!("a list ran again after its descriptor tables were given back"));
        }
        for (heap, id, first, n) in self.dynamic.drain(..) {
            crate::descriptor::release_dynamic(heap, id, first, n);
        }
        self.fresh_tables.clear();
        self.root_srv_slots.clear();
        self.root_uav_slots.clear();
        self.tables_gone = true;
        #[cfg(feature = "diagnostics")]
        {
            self.recording = 0;
        }
    }
}

/// Diagnostics, since the last line ([`tables_line`]): the descriptor tables
/// draws and dispatches have made afresh ([`State::tables`]).
/// 0: tables asked of the list's own memory of them (`fresh_tables`); 1: of
/// them, found there (the same table holding the same records). The others
/// are made, and each is counted once in 2 to 7 or in none of them:
/// 2: its records are those of a table this recording of the list made
/// before, at another address; 6: not that, but they differ from such a
/// table's only in where the constants are; 3, 4, 5: neither, but its
/// records are those of a table some list made earlier in this frame, in
/// the frame before, 2 to 8 frames before; 7: none of those, but they
/// differ only in where the constants are from a table made in this frame
/// or the 8 before.
/// 8 to 11: the tables to make by their slots (1, 2 to 4, 5 to 16, more);
/// 12: those slots; 13 to 16: the slots made as a constant view, a texture
/// view, another view (buffers), a null view; 17: entries made readable in
/// tables made; 18: in tables found; 19: entries of tables that are not
/// made afresh ([`State::table_entry`]: the title's own slots).
/// 20: tables timed (one in 16 of each thread); their nanoseconds: 21
/// finding the slots (`HostHeap::dynamic`), 22 looking the views up and
/// making them, 23 making entries readable, 24 the title's own slots of
/// texture views (`HostHeap::texture_slot_tabled`, `HostHeap::mirror`).
/// 25: tables to make that found no slots.
#[cfg(feature = "diagnostics")]
static TABLES: [AtomicU64; 26] = [const { AtomicU64::new(0) }; 26];

#[cfg(feature = "diagnostics")]
thread_local! {
    /// This thread's counts for [`TABLES`] not yet added to it, and its
    /// calls of [`State::tables`] since they last were.
    static TALLY: std::cell::RefCell<([u64; 26], u32)> = const { std::cell::RefCell::new(([0; 26], 0)) };
    /// This thread's tables to make, for timing one in 16.
    static TABLES_MADE: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
}

/// Diagnostics: `n` more of count `at` of [`TABLES`], kept by the thread
/// until [`tally_done`] (no shared memory is written here).
#[cfg(feature = "diagnostics")]
fn tally(at: usize, n: u64) {
    TALLY.with(|t| t.borrow_mut().0[at] += n);
}

/// Diagnostics: a call of [`State::tables`] ends; every 64th of a thread
/// adds the thread's counts to [`TABLES`].
#[cfg(feature = "diagnostics")]
fn tally_done() {
    TALLY.with(|t| {
        let t = &mut *t.borrow_mut();
        t.1 += 1;
        if t.1 < 64 {
            return;
        }
        t.1 = 0;
        for (total, sum) in TABLES.iter().zip(t.0.iter_mut()) {
            if *sum != 0 {
                total.fetch_add(std::mem::take(sum), Ordering::Relaxed);
            }
        }
    });
}

/// Diagnostics: the clock of a timed table: each lap goes to a count of
/// [`TABLES`]. Without a clock (15 tables of 16) a lap does nothing.
#[cfg(feature = "diagnostics")]
struct Lap(Option<std::time::Instant>);

#[cfg(feature = "diagnostics")]
impl Lap {
    /// The time since the lap before (or the start) goes to count `at`.
    fn lap(&mut self, at: usize) {
        if let Some(last) = &mut self.0 {
            let now = std::time::Instant::now();
            tally(at, now.duration_since(*last).as_nanos() as u64);
            *last = now;
        }
    }
}

/// Diagnostics: slots of each of the four parts of [`seen`].
#[cfg(feature = "diagnostics")]
const SEEN_SLOTS: usize = 1 << 19;

/// Diagnostics: where the fingerprints of the tables made are kept
/// ([`table_asked`]): four parts of [`SEEN_SLOTS`] slots, each slot the
/// last fingerprint that fell on it. Made at the first table (16 MB).
#[cfg(feature = "diagnostics")]
fn seen() -> &'static [AtomicU64] {
    static SEEN: std::sync::OnceLock<Box<[AtomicU64]>> = std::sync::OnceLock::new();
    SEEN.get_or_init(|| (0..4 * SEEN_SLOTS).map(|_| AtomicU64::new(0)).collect())
}

/// Diagnostics: counts the table of `n` records at `table` that a list in
/// its recording number `recording` (given one here if it has none) is
/// about to make, by where the same records were seen before ([`TABLES`] 2
/// to 12), and starts the clock of one table in 16.
///
/// A table is told by a fingerprint of its records
/// (`crate::descriptor::host::table_prints`), and a fingerprint is found
/// again only while no other has fallen on its slot since: the counts of
/// tables seen before are lower bounds. With 30,000 tables a frame on
/// 524,288 slots, about 6% of the tables last made a whole frame ago are
/// missed, fewer of the nearer ones.
#[cfg(feature = "diagnostics")]
fn table_asked(recording: &mut u64, table: u64, n: u32) -> Lap {
    const K: u64 = 0x9e37_79b9_7f4a_7c15;
    static NEXT: AtomicU64 = AtomicU64::new(1);
    if *recording == 0 {
        *recording = NEXT.fetch_add(1, Ordering::Relaxed);
    }
    let (whole, shape) = crate::descriptor::host::table_prints(table, n);
    let seen = seen();
    let mix = |v: u64| {
        let v = v.wrapping_mul(K);
        v ^ (v >> 31)
    };
    // In this recording of this list: the slot holds the fingerprint mixed
    // with the recording's number.
    let own = recording.wrapping_mul(0xff51_afd7_ed55_8ccd);
    let in_list = |part: usize, print: u64| {
        let tag = mix(print ^ own) | (1 << 63);
        let slot = &seen[part * SEEN_SLOTS + (tag as usize & (SEEN_SLOTS - 1))];
        slot.swap(tag, Ordering::Relaxed) == tag
    };
    // By any list: the slot holds the fingerprint's high half and the frame
    // it was last seen in. The frames since then, if it is there.
    let frame = crate::frame::current() as u32;
    let frames_ago = |part: usize, print: u64| {
        let tag = mix(print);
        let slot = &seen[part * SEEN_SLOTS + (tag as usize & (SEEN_SLOTS - 1))];
        let was = slot.swap((tag & !0xffff_ffff) | u64::from(frame), Ordering::Relaxed);
        (was != 0 && was >> 32 == tag >> 32).then(|| frame.wrapping_sub(was as u32))
    };
    // All four are noted for every table, whichever it is counted as.
    let (list_whole, list_shape) = (in_list(0, whole), in_list(1, shape));
    let (ago_whole, ago_shape) = (frames_ago(2, whole), frames_ago(3, shape));
    let class = if list_whole {
        Some(2)
    } else if list_shape {
        Some(6)
    } else {
        match (ago_whole, ago_shape) {
            (Some(0), _) => Some(3),
            (Some(1), _) => Some(4),
            (Some(2..=8), _) => Some(5),
            (_, Some(0..=8)) => Some(7),
            _ => None,
        }
    };
    if let Some(class) = class {
        tally(class, 1);
    }
    tally(
        match n {
            1 => 8,
            2..=4 => 9,
            5..=16 => 10,
            _ => 11,
        },
        1,
    );
    tally(12, u64::from(n));
    let timed = TABLES_MADE.with(|made| {
        made.set(made.get().wrapping_add(1));
        made.get().is_multiple_of(16)
    });
    if timed {
        tally(20, 1);
    }
    Lap(timed.then(std::time::Instant::now))
}

/// Diagnostics, with the line of the recording: the tables made afresh
/// ([`TABLES`]) and the host heap's locks (`crate::descriptor::host::LOCKS`),
/// per frame over `frames` frames.
#[cfg(feature = "diagnostics")]
pub(crate) fn tables_line(frames: f64) {
    let frames = frames.max(1.0);
    let t: Vec<f64> = TABLES.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    let l: Vec<f64> = crate::descriptor::host::LOCKS.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    let (asked, found) = (t[0], t[1]);
    let to_make = asked - found;
    let (in_list, in_list_but_constants) = (t[2], t[6]);
    let (this_frame, frame_before, older, but_constants) = (t[3], t[4], t[5], t[7]);
    let new = to_make - t[2..8].iter().sum::<f64>();
    let (one, to_4, to_16, more, slots) = (t[8], t[9], t[10], t[11], t[12]);
    let (constants, textures, others, nulls) = (t[13], t[14], t[15], t[16]);
    let (readable_made, readable_found, entries) = (t[17], t[18], t[19]);
    // One table in 16 was timed: the milliseconds of all of them, at that rate.
    let timed = t[20];
    let all = |ns: f64| if timed > 0.0 { ns / timed * to_make / 1e6 } else { 0.0 };
    let (finding, views, readable, title) = (all(t[21]), all(t[22]), all(t[23]), all(t[24]));
    let refused = t[25];
    tracing::info!(
        "descriptor tables per frame ({frames:.0} frames; counts of a thread join after 64 of its binds): {asked:.0} asked of their list's \
         memory, {found:.0} found there; of the {to_make:.0} to make: {in_list:.0} hold the records of a table made earlier in the same \
         recording of their list at another address (a table cache within a list would serve these), {in_list_but_constants:.0} more \
         differ from one only in where their constants are; of the rest {this_frame:.0} hold the records of a table another list made \
         earlier in the frame, {frame_before:.0} of one made the frame before, {older:.0} of one made 2 to 8 frames before, \
         {but_constants:.0} more differ from one of those only in where their constants are; {new:.0} are new (the counts of repeats \
         are lower bounds); {refused:.0} found no slots; tables of 1 slot {one:.0}, of 2 to 4 {to_4:.0}, of 5 to 16 {to_16:.0}, of more \
         {more:.0}; {slots:.0} slots: {constants:.0} constant views, {textures:.0} texture views, {others:.0} other views, {nulls:.0} null; \
         entries made readable {readable_made:.0} in tables made, {readable_found:.0} in tables found, {entries:.0} entries of tables \
         not made afresh; one table in 16 timed ({timed:.0}), all of them at that rate: {finding:.1} ms finding slots, {views:.1} ms \
         looking up and making views, {readable:.1} ms making entries readable, {title:.1} ms on the title's own slots of textures"
    );
    let (mirrors_other, mirrors_draw) = (l[0], l[1]);
    let (changed_other, changed_draw) = (l[2], l[3]);
    let (made_waits, made_waited) = (l[4], l[5] / 1e6);
    // One call in 16 was timed: the mean of those, and all calls at that mean.
    let mean = |ns: f64, n: f64| if n > 0.0 { ns / n } else { 0.0 };
    let (same_ns, changed_ns) = (mean(l[6], l[8]), mean(l[7], l[9]));
    let changed = changed_other + changed_draw;
    let same_ms = same_ns * (mirrors_other + mirrors_draw - changed) / 1e6;
    let changed_ms = changed_ns * changed / 1e6;
    let (keeps, kept_waits, kept_waited) = (l[10], l[11], l[12] / 1e6);
    let (next_waits, next_waited) = (l[13], l[14] / 1e6);
    tracing::info!(
        "the host heap's locks per frame, all threads: the slots' lock (HostHeap::mirror) taken {mirrors_draw:.0} times for draws \
         ({changed_draw:.0} found the slot's record changed) and {mirrors_other:.0} times for writes and the pass over copied records \
         ({changed_other:.0} changed); {made_waits:.0} takes found it held and waited {made_waited:.2} ms in all (every wait timed); it \
         was held {same_ns:.0} ns a call where the slot was as it was and {changed_ns:.0} ns where it had changed (one call in 16 timed), \
         {same_ms:.1} ms and {changed_ms:.1} ms for all calls at those means; the lock of the resources slots hold (HostHeap::keep) \
         taken {keeps:.0} times, {kept_waits:.0} found it held and waited {kept_waited:.2} ms; the lock of the dynamic slots \
         (HostHeap::dynamic; its takes are the line above's tables asked): {next_waits:.0} found it held and waited {next_waited:.2} ms"
    );
    let why: Vec<f64> =
        crate::descriptor::host::DYNAMIC_WHY.iter().map(|n| n.swap(0, Ordering::Relaxed) as f64 / frames).collect();
    let low = crate::descriptor::host::DYNAMIC_LOW.swap(u32::MAX, Ordering::Relaxed);
    let low = if low == u32::MAX { "no table asked".to_string() } else { format!("{low}") };
    tracing::info!(
        "dynamic slots per frame, tables refused by cause: {:.0} asked more slots than were free (the ring was full), {:.0} had room \
         but a search for a table no longer had failed since the last release (not searched), {:.0} were searched for and no run of \
         free slots was that long (the free slots lay apart); {:.0} tables got a run lying across the place the search starts from \
         (missed by the search as it was before); the fewest free slots seen since the last line: {low} of {}",
        why[0],
        why[1],
        why[2],
        why[3],
        crate::descriptor::host::DYNAMIC,
    );
}

/// An experiment on the title's scissor rectangles (`VICTORY_SCISSOR_TEST=full`):
/// they are all widened to the whole target. The title's room at
/// the start of the prologue lost its walls in five runs of seven, and in
/// those runs the player's body and the world outside were cut by one
/// rectangle around the window (run 85, frame 2046): one run with this on
/// says whether the walls are drawn and cut away by such a rectangle, or
/// not drawn at all. Off by default: a rectangle the title sets is part of
/// its picture.
fn scissor_test() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("VICTORY_SCISSOR_TEST").as_deref() == Ok("full"))
}

/// An experiment on mesh draws (`VICTORY_MESH_TEST`): 1 for `skip` (none
/// is issued), 2 for `scissor` (each runs its shaders but rasterises one
/// pixel), 0 otherwise. It tells a card hung by a mesh shader from one hung
/// by what the shader's triangles cover.
fn mesh_test() -> u8 {
    static ON: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    *ON.get_or_init(|| match std::env::var("VICTORY_MESH_TEST").as_deref() {
        Ok("skip") => 1,
        Ok("scissor") => 2,
        _ => 0,
    })
}

/// `VICTORY_MESH_TEST=keep:<hex digits>[:<hex digits>]`: only the mesh
/// draws whose mesh shader's hash ends in one of the first digits (and
/// whose digit before it is one of the second, if given) are issued: a
/// search for the shaders that stop the card.
fn mesh_kept(pipeline: usize) -> bool {
    static KEEP: std::sync::OnceLock<Option<(String, String)>> = std::sync::OnceLock::new();
    let Some((last, before)) = KEEP.get_or_init(|| {
        let value = std::env::var("VICTORY_MESH_TEST").ok()?;
        let mut parts = value.strip_prefix("keep:")?.split(':');
        Some((parts.next()?.to_owned(), parts.next().unwrap_or("").to_owned()))
    }) else {
        return true;
    };
    let Some(p) = crate::pipeline::host_pipeline(pipeline as This) else {
        return true;
    };
    let Some(at) = p.name.find("mesh ") else {
        return true;
    };
    let hash = p.name[at + 5..].as_bytes();
    if hash.len() < 16 {
        return true;
    }
    last.as_bytes().contains(&hash[15]) && (before.is_empty() || before.as_bytes().contains(&hash[14]))
}

/// Draws recorded on the host GPU since the last frame line.
static DRAWN: AtomicU64 = AtomicU64::new(0);

/// Draws run on the host so far.
pub(crate) fn drawn() -> u64 {
    DRAWN.load(Ordering::Relaxed)
}

/// The draws traced since the last present (diagnostics builds, with
/// debug logging for this module): logged when the frame comes out black.
#[cfg(feature = "diagnostics")]
static FRAME: std::sync::Mutex<Vec<String>> = std::sync::Mutex::new(Vec::new());

/// The draws traced since the last call.
#[cfg(feature = "diagnostics")]
pub(crate) fn take_frame() -> Vec<String> {
    std::mem::take(&mut *FRAME.lock().unwrap_or_else(|e| e.into_inner()))
}

/// Draws, mesh dispatches and dispatches recorded but not run on the host
/// (no host pipeline) since the last call.
pub(crate) fn take_dropped() -> u64 {
    DROPPED.swap(0, Ordering::Relaxed)
}
/// A root SRV's dynamic slot in a list, and the host copy it shows.
type RootSlot = (u32, Option<std::sync::Arc<crate::shadow::Shadow>>);

/// Buffer views of the heap whose slot showed another host buffer than
/// the copy the pass brought up to date for them, since the start
/// (diagnostics; for the memory line). It stays 0 while the pass and the
/// slots agree; with `VICTORY_VIEW_EXTENT=0` it counts the views that were
/// read stale.
#[cfg(feature = "diagnostics")]
pub(crate) static HEAP_VIEWS_ELSEWHERE: AtomicU64 = AtomicU64::new(0);
/// Passes over the view heap's buffer views for shaders that pick them by
/// index ([`State::heap_buffers`]), and the microseconds they took.
pub(crate) static HEAP_BUFFER_PASSES: AtomicU64 = AtomicU64::new(0);
/// Per host heap, for [`State::heap_buffers`]: its passes so far, and the
/// host copy and range each slot's buffer view reads. (At the module's
/// level so that the marked pass can have this one start anew:
/// [`State::heap_buffers_marked`].)
type Ranges = std::collections::HashMap<u32, (std::sync::Arc<crate::shadow::Shadow>, u64, u64)>;
static RANGES: std::sync::Mutex<Option<std::collections::HashMap<usize, (u64, Ranges)>>> =
    std::sync::Mutex::new(None);
/// Per host heap, for [`State::heap_buffers_marked`]: what the marked pass
/// keeps (`VICTORY_HEAP_PASS_MARKED`; nothing while that is unset).
type Marked = crate::heap_pass::Picked<std::sync::Arc<crate::shadow::Shadow>>;
static MARKED: std::sync::Mutex<Option<std::collections::HashMap<usize, Marked>>> = std::sync::Mutex::new(None);
pub(crate) static HEAP_BUFFER_MICROS: AtomicU64 = AtomicU64::new(0);
/// Where the pass over textures picked by index records its uploads: into
/// a list of the title's, or into lists of the thread that runs it ahead.
pub(crate) enum Into<'a> {
    List(&'a HostList),
    Ahead(&'a mut crate::heap_worker::Batch),
}

/// Of [`HEAP_BUFFER_MICROS`], the microseconds of the pass over textures
/// picked by index that runs inside it (`heap_textures`), and the buffer
/// slots the pass over buffer views looked at; since the last frame line.
pub(crate) static HEAP_PASS_PARTS: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];
/// Textures brought up to date for shaders that pick them by index
/// ([`State::heap_textures`]), the bytes of their memory, and the slots
/// still waiting.
pub(crate) static HEAP_TEXTURES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_TEXTURE_BYTES: AtomicU64 = AtomicU64::new(0);
pub(crate) static HEAP_TEXTURES_WAITING: AtomicU64 = AtomicU64::new(0);
/// Passes that made no texture because the card had no room
/// (`crate::resource::room`), for the memory line.
pub(crate) static HEAP_TEXTURES_FULL: AtomicU64 = AtomicU64::new(0);

/// Frames after a whole upload of a texture picked by index at which its
/// memory is looked at again, each counted from the look before.
const RECHECK_FRAMES: [u64; 5] = [4, 12, 40, 120, 400];

/// Frames between looks at an uploaded texture picked by index once
/// [`RECHECK_FRAMES`] is through (`VICTORY_HEAP_RECHECK_EVERY=<frames>`; unset
/// or 0: it is not looked at again, as before). The title streams a
/// texture's memory in and rewrites it long after its view was written:
/// runs 338-339 logged changes found at the schedule's last look, and what
/// changes later stays as it was on the card for the rest of the run (skin
/// that turns black or red). After RyanCase94's 920423a9.
fn recheck_every() -> u64 {
    static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *EVERY.get_or_init(|| {
        let every = std::env::var("VICTORY_HEAP_RECHECK_EVERY").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
        if every > 0 {
            tracing::info!("uploaded textures picked by index are looked at again every {every} frames (VICTORY_HEAP_RECHECK_EVERY)");
        }
        every
    })
}

/// Looks at uploaded textures picked by index that found the memory as it
/// was, and that found it changed (uploaded again), since the start.
pub(crate) static HEAP_TEXTURES_RECHECKED: [AtomicU64; 2] = [AtomicU64::new(0), AtomicU64::new(0)];

/// Uploads of textures already on the card made while it had no room for
/// new ones, since the start.
pub(crate) static HEAP_TEXTURES_REUPLOADED_FULL: AtomicU64 = AtomicU64::new(0);

/// Whether a pass over the textures picked by index on a full card holds
/// back the slots that need a new host copy (`VICTORY_FULL_CARD_PASS=0`:
/// no, as before: they stay due, each pass looks at the same ones again
/// and stops at the 257th). Held back, such a slot is looked at again
/// after about four seconds or as soon as the card has room, and the pass
/// goes on to the slots behind it: the ones whose host copy exists and
/// only wants bringing up to date, and the ones whose memory went (letting
/// their host copies go is what makes room). What is uploaded while the
/// card has room is the same either way.
fn full_card_pass() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_FULL_CARD_PASS").is_none_or(|v| v != "0"))
}

/// Whether a texture that has its host copy is still brought up to date
/// while the card has no room for new ones (`VICTORY_REUPLOAD_WHEN_FULL=0`:
/// no).
fn reupload_when_full() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_REUPLOAD_WHEN_FULL").is_none_or(|v| v != "0"))
}

/// Whether textures picked by index are looked at again after their upload
/// (`VICTORY_HEAP_RECHECK=0`: no).
fn heap_recheck() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_RECHECK").is_none_or(|v| v != "0"))
}

/// Whether a texture picked by index that a look found changed after its
/// upload is looked at again in the next frame, not after
/// `RECHECK_FRAMES[0]` (`VICTORY_RECHECK_SOON=1`; off by default). A
/// texture whose bytes land in several pieces, or that the title keeps
/// writing, was shown four frames behind at each change. (The AMD sibling
/// repository's switch of that name, on by default there.)
fn recheck_soon() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_RECHECK_SOON").is_some_and(|v| v == "1"))
}

/// Uploads of textures picked by index looked at again in the next frame
/// because a look had found them changed ([`recheck_soon`]), since the
/// start.
pub(crate) static HEAP_TEXTURES_SOON: AtomicU64 = AtomicU64::new(0);
/// Times a placed texture's slot was passed over because its memory is not
/// mapped.
pub(crate) static HEAP_TEXTURES_AWAY: AtomicU64 = AtomicU64::new(0);
/// Slots the pass found up to date already: their texture is uploaded by
/// the draws that read it through a table (it has a staging buffer).
pub(crate) static HEAP_TEXTURES_TABLED: AtomicU64 = AtomicU64::new(0);
/// The queue of textures picked by index at the last pass
/// (`TextureQueue::census`): slots in it, due, waiting for their memory,
/// uploaded in part, and the frames the oldest due one has waited.
pub(crate) static HEAP_TEXTURES_QUEUE: [AtomicU64; 5] = [const { AtomicU64::new(0) }; 5];
/// Since the start: looks at a slot uploaded in part that found no more of
/// its memory mapped; uploads of textures not all mapped; and of those,
/// the ones whose first page was not there (asked by the first page alone,
/// as before, they counted as not there at all).
pub(crate) static HEAP_TEXTURES_PARTLY: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
static DROPPED: AtomicU64 = AtomicU64::new(0);

/// Marks the last traced draw as not run on the host.
#[cfg(feature = "diagnostics")]
fn dropped(pipeline: usize) {
    crate::pipeline::note_drop(pipeline as This);
    if let Some(last) = FRAME.lock().unwrap_or_else(|e| e.into_inner()).last_mut() {
        last.push_str(" DROPPED");
    }
}

/// Dispatches run on the host so far.
static DISPATCHED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn dispatched() -> u64 {
    DISPATCHED.load(Ordering::Relaxed)
}

/// The host address of a root view: its buffer's host copy, or for a
/// constant buffer not 256-byte aligned there (the console needs 16) a
/// copy in upload memory.
/// shortcut: a root view has no size; constant buffers are taken as 64 KiB
/// (cut at their buffer's end), other views as all that is left of their
/// buffer. (They were taken as 1 MiB: 1,452 of the 2,021 mesh vertex
/// streams of a cutscene frame reach further, and what lay past the first
/// megabyte was never brought up to date - meshes without bodies, exploded
/// triangles. `VICTORY_ROOT_VIEW_MB` sets a limit again.)
fn root_view(list: &HostList, address: u64, srv: bool) -> Option<u64> {
    static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let limit = *LIMIT.get_or_init(|| {
        std::env::var("VICTORY_ROOT_VIEW_MB")
            .ok()
            .and_then(|v| v.parse::<u64>().ok())
            .map_or(u64::MAX, |mb| mb << 20)
    });
    let shadow = crate::shadow::find(address, 1)?;
    let len = shadow
        .left(address)
        .min(if srv { limit } else { 1 << 16 });
    let gpu = shadow.gpu(address);
    if srv || gpu % 256 == 0 {
        shadow.refresh(list, address, len);
        return Some(gpu);
    }
    if crate::host::list_reuse() {
        return list.fill_scratch(address as *const u8, len);
    }
    let (gpu, to) = list.scratch(len)?;
    list.defer_fill(to, address as *const u8, len as usize);
    Some(gpu)
}

macro_rules! once {
    ($($arg:tt)*) => {{
        static ONCE: std::sync::Once = std::sync::Once::new();
        ONCE.call_once(|| tracing::warn!($($arg)*));
    }};
}

/// Whether the stages of one draw share a table they both need made afresh
/// (`VICTORY_TABLE_STAGES=1`; off by default: each stage makes its own, as
/// before). `alt`: twenty seconds with, twenty without, in turn, each
/// change said in the log; `alt40`: forty and forty, so that a run can
/// carry this switch and another one in turn and tell the two apart (the
/// four pairs of states each come up for twenty seconds).
///
/// A draw's stages read the same root tables, and each asked for slots up
/// to the last entry it reads itself. The count is in the key of
/// `State::fresh_tables`, so a mesh stage and a pixel stage that read one
/// table to different depths made it twice, in dynamic slots of its own
/// each time. With this both ask for the larger count ([`stage_spans`]),
/// the second finds the first one's table, and one table is made.
fn table_stages() -> bool {
    /// 0 off, 1 on, 2 in turn every 20 s, 3 in turn every 40 s.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    match *MODE.get_or_init(|| match std::env::var("VICTORY_TABLE_STAGES").as_deref() {
        Ok("1") => 1,
        Ok("alt") => 2,
        Ok("alt40") => 3,
        _ => 0,
    }) {
        0 => false,
        1 => true,
        mode => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let period = if mode == 3 { 40 } else { 20 };
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / period) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "the stages of a draw share the tables they need made afresh from now (VICTORY_TABLE_STAGES=alt): {}",
                    if on { "ON" } else { "off, each stage makes its own as usual" }
                );
            }
            on
        }
    }
}

/// For [`table_stages`]: per root table the stages of `p` read, the slots
/// it has up to the last entry read by a stage that also reads a constant
/// view in it (0: no stage does), and up to the last entry any stage reads:
/// (root parameter, with constants, all).
fn stage_spans(p: &HostPipeline) -> Vec<(u8, u32, u32)> {
    let mut spans: Vec<(u8, u32, u32)> = Vec::new();
    for stage in &p.stages {
        let t = stage.translated();
        let mut constant = [false; 256];
        for u in t.used.iter().filter(|u| u.root == RootType::Table && u.kind == ResourceType::Cbv) {
            constant[usize::from(u.slot)] = true;
        }
        for u in t.used.iter().filter(|u| u.root == RootType::Table) {
            let n = u.table_offset_dwords / 8 + 1;
            let with = if constant[usize::from(u.slot)] { n } else { 0 };
            match spans.iter_mut().find(|s| s.0 == u.slot) {
                Some(s) => {
                    s.1 = s.1.max(with);
                    s.2 = s.2.max(n);
                }
                None => spans.push((u.slot, with, n)),
            }
        }
    }
    spans
}

fn set(roots: &mut Vec<Root>, slot: u32, root: Root) {
    let slot = slot as usize;
    if roots.len() <= slot {
        roots.resize(slot + 1, Root::None);
    }
    roots[slot] = root;
}

impl State {
    pub(crate) fn reset(&mut self, pipeline: This) {
        *self = State::default();
        self.pipeline = pipeline as usize;
    }

    /// The title's topology; console-only ones become host lists (see
    /// [`crate::primitive`]).
    pub(crate) fn set_topology(&mut self, list: &HostList, topology: u32) {
        self.topology = topology;
        self.topology_set = true;
        let host = match topology {
            RECTANGLES => 4,
            TRIANGLE_FAN | LINE_LOOP | QUADS | QUAD_STRIP => {
                once!(topology, "console topology drawn as a triangle list");
                4
            }
            t => t,
        };
        // SAFETY: an open list.
        unsafe {
            list.list
                .IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY(host as i32))
        };
    }

    pub(crate) fn set_pipeline(&mut self, pipeline: This) {
        self.pipeline = pipeline as usize;
    }

    pub(crate) fn set_heaps(&mut self, heaps: &[This]) {
        self.heaps = [0; 2];
        for &h in heaps {
            if !h.is_null() {
                let samplers = crate::descriptor::heap(h).samplers();
                self.heaps[usize::from(samplers)] = h as usize;
            }
        }
    }

    fn roots(&mut self, compute: bool) -> &mut Vec<Root> {
        if compute {
            &mut self.compute
        } else {
            &mut self.graphics
        }
    }

    pub(crate) fn set_table(&mut self, compute: bool, slot: u32, handle: u64) {
        set(self.roots(compute), slot, Root::Table(handle));
    }

    pub(crate) fn set_view(&mut self, compute: bool, slot: u32, address: u64) {
        set(self.roots(compute), slot, Root::View(address));
    }

    pub(crate) fn set_constants(&mut self, compute: bool, slot: u32, values: &[u32], offset: u32) {
        let roots = self.roots(compute);
        let s = slot as usize;
        if !matches!(roots.get(s), Some(Root::Constants(_))) {
            set(roots, slot, Root::Constants(Vec::new()));
        }
        if let Root::Constants(c) = &mut roots[s] {
            let end = offset as usize + values.len();
            if c.len() < end {
                c.resize(end, 0);
            }
            c[offset as usize..end].copy_from_slice(values);
        }
    }

    pub(crate) fn set_vertex_buffers(&mut self, start: u32, views: &[VertexView]) {
        for (i, v) in views.iter().enumerate() {
            if let Some(slot) = self.vertex.get_mut(start as usize + i) {
                *slot = *v;
            }
        }
    }

    pub(crate) fn clear_vertex_buffers(&mut self, start: u32, count: u32) {
        for slot in self.vertex.iter_mut().skip(start as usize).take(count as usize) {
            *slot = VertexView::default();
        }
    }

    pub(crate) fn set_index_buffer(&mut self, view: Option<&IndexView>) {
        self.index = view.copied().unwrap_or_default();
    }

    /// Render targets and depth, set on the host list at once.
    pub(crate) fn set_targets(&mut self, list: &HostList, handles: &[u64], depth: u64) {
        let Some(host) = crate::host::host() else {
            return;
        };
        self.target_handles.clear();
        self.target_handles.extend_from_slice(handles);
        self.depth_handle = depth;
        self.targets_set = true;
        self.targets.clear();
        // The formats of the host views, for the draws' pipelines.
        let mut formats = [0; 9];
        let mut rtvs = Vec::new();
        for &h in handles {
            let Some(record) = crate::descriptor::rtv_record(h) else {
                once!(
                    handle = format_args!("{h:#x}"),
                    "render target without a recorded view"
                );
                continue;
            };
            let Some(texture) = crate::resource::host_resource(record.resource) else {
                // A typed null view (or a target without a host copy) still
                // takes its slot: left out, the next live target would move
                // into it and the pixel shader's outputs reach the wrong
                // targets.
                // shortcut: a view without a description or a format has no
                // null view to stand for it and is left out, as before.
                let null = record
                    .desc()
                    .filter(|d| d.Format.0 != 0 && d.ViewDimension.0 != 0);
                if let Some(desc) = null {
                    if rtvs.len() < 8 {
                        formats[rtvs.len()] = desc.Format.0 as u32;
                    }
                    rtvs.push(host.rtv(None, Some(&desc)));
                }
                continue;
            };
            self.targets.push(record.resource as usize);
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            let desc = record.desc();
            if rtvs.len() < 8 {
                formats[rtvs.len()] = bound_format(texture, desc.as_ref().map(|d| d.Format));
            }
            rtvs.push(host.rtv(Some(texture), desc.as_ref()));
        }
        self.depth = 0;
        self.depth_state = DEPTH_WRITE;
        let dsv = crate::descriptor::dsv_record(depth).and_then(|record| {
            let texture = crate::resource::host_resource(record.resource)?;
            self.depth = record.resource as usize;
            let desc = record.depth_desc();
            self.depth_state = depth_state(texture, desc.as_ref());
            formats[8] = bound_format(texture, desc.as_ref().map(|d| d.Format));
            if crate::resource::depth_and_unordered(record.resource) {
                crate::resource::DEPTH_UNORDERED[1].fetch_add(1, Ordering::Relaxed);
            }
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(record.resource);
            Some(host.dsv(texture, desc.as_ref()))
        });
        self.target_formats = formats;
        // SAFETY: an open list; the views were just made.
        unsafe {
            list.list.OMSetRenderTargets(
                rtvs.len() as u32,
                (!rtvs.is_empty()).then_some(rtvs.as_ptr()),
                false,
                dsv.as_ref().map(std::ptr::from_ref),
            )
        };
    }

    /// Viewports, scissors, the blend factor and the stencil reference: set
    /// on the host list at once, and kept for [`State::replay`].
    pub(crate) fn set_viewports(&mut self, list: &HostList, viewports: &[D3D12_VIEWPORT]) {
        self.viewports.clear();
        self.viewports.extend_from_slice(viewports);
        crate::draw::viewports(list, viewports);
    }

    pub(crate) fn set_scissors(&mut self, list: &HostList, rects: &[RECT]) {
        self.scissors.clear();
        self.scissors.extend_from_slice(rects);
        if scissor_test() {
            // The experiment: every rectangle the title sets is widened to
            // the largest target there is, so nothing is cut by one. What
            // the listing shows (`op_text`) stays the title's own.
            let full = RECT {
                left: 0,
                top: 0,
                right: 16384,
                bottom: 16384,
            };
            crate::draw::scissors(list, &vec![full; rects.len()]);
            return;
        }
        crate::draw::scissors(list, rects);
    }

    pub(crate) fn set_blend_factor(&mut self, list: &HostList, factor: Option<&[f32; 4]>) {
        self.blend = factor.copied();
        crate::draw::blend_factor(list, factor);
    }

    pub(crate) fn set_stencil_ref(&mut self, list: &HostList, reference: u32) {
        self.stencil = Some(reference);
        crate::draw::stencil_ref(list, reference);
    }

    pub(crate) fn set_depth_bounds(&mut self, list: &HostList, min: f32, max: f32) {
        self.depth_bounds = Some([min, max]);
        crate::draw::depth_bounds(list, min, max);
    }

    /// `list` is a new host list that continues the title's list (it was
    /// cut at a wait for a memory value, `crate::list::cut`): a new host
    /// list has nothing set, so what the title set so far is set again.
    /// The pipeline, the root signatures and the heaps are set by the next
    /// draw or dispatch (`bind`, which is told here that the host list has
    /// none); root arguments, vertex and index buffers are set by every
    /// draw anyway. The dynamic descriptor slots and the tables made in
    /// them stay: the parts before the cut have not run yet, and they are
    /// given back when the title's list is reset.
    /// shortcut: the render target views are made again from the handles'
    /// descriptors as they are now, not as they were when the title set
    /// the targets (a descriptor rewritten in between names another view).
    pub(crate) fn replay(&mut self, list: &HostList) {
        self.host_pipeline = 0;
        self.host_roots = [0; 2];
        self.host_heaps = [0; 2];
        if self.topology_set {
            self.set_topology(list, self.topology);
        }
        if self.targets_set {
            let handles = std::mem::take(&mut self.target_handles);
            self.set_targets(list, &handles, self.depth_handle);
        }
        if !self.viewports.is_empty() {
            crate::draw::viewports(list, &self.viewports);
        }
        if !self.scissors.is_empty() {
            let rects = std::mem::take(&mut self.scissors);
            self.set_scissors(list, &rects);
        }
        if let Some(factor) = self.blend {
            crate::draw::blend_factor(list, Some(&factor));
        }
        if let Some(reference) = self.stencil {
            crate::draw::stencil_ref(list, reference);
        }
        if let Some([min, max]) = self.depth_bounds {
            crate::draw::depth_bounds(list, min, max);
        }
    }

    /// The user-data value of a root argument, as translated shaders read
    /// it: a table's byte offset in its heap, a constant's value.
    fn user_value(&self, compute: bool, slot: u8, root: RootType, dword: u32) -> u32 {
        match (self.roots_ref(compute).get(usize::from(slot)), root) {
            (Some(Root::Table(handle)), RootType::Table) => match self.heap_of(*handle) {
                Some(heap) => (handle - heap.start) as u32,
                None => {
                    once!(
                        handle = format_args!("{handle:#x}"),
                        "descriptor table outside the bound heaps"
                    );
                    0
                }
            },
            (Some(Root::Constants(c)), RootType::Constants) => {
                c.get(dword as usize).copied().unwrap_or(0)
            }
            // The host's own ([`Self::host_word`]): no root argument.
            (_, RootType::Other(_)) => 0,
            _ => {
                once!(
                    slot,
                    ?root,
                    "a shader reads a root argument the title did not set"
                );
                0
            }
        }
    }

    /// A user-data dword the host computes for a translated shader
    /// (`victory_shader::emit::USER_MEMORY_BASE`, `USER_VIEW_ADDRESS`).
    fn host_word(
        &self,
        compute: bool,
        kind: u32,
        slot: u8,
        dword: u32,
        memory: Option<&crate::memory::Ready>,
    ) -> u32 {
        use victory_shader::emit::{
            USER_DISPATCH_SPREAD, USER_DISPATCH_THREADS, USER_MEMORY_BASE, USER_VIEW_ADDRESS, USER_VIEW_BIAS,
        };
        if kind == USER_DISPATCH_SPREAD {
            return self.dispatch_spread[usize::from(dword != 0)];
        }
        if kind == USER_DISPATCH_THREADS {
            return self.dispatch_threads.get(dword as usize).copied().unwrap_or(0);
        }
        if kind == USER_MEMORY_BASE {
            return memory.map_or(0, |m| m.base);
        }
        if kind == USER_VIEW_BIAS {
            // The view begins on 16 bytes of the copy the address finds
            // (`root_uav`, `root_srv_table`).
            // shortcut: found again here, not taken from the list's slot.
            return match self.roots_ref(compute).get(usize::from(slot)) {
                Some(&Root::View(address)) => crate::shadow::find(address, 1).map_or(0, |s| {
                    // SAFETY: a live buffer.
                    let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                    ((s.gpu(address) - base) & 15) as u32
                }),
                _ => 0,
            };
        }
        if kind != USER_VIEW_ADDRESS && kind != USER_VIEW_ADDRESS + 1 {
            once!(kind, "a shader reads a host value this build does not know");
            return 0;
        }
        // The title address of the buffer view at that dword of the table.
        let address = match self.roots_ref(compute).get(usize::from(slot)) {
            Some(&Root::Table(table)) => crate::descriptor::view(table + u64::from(dword) * 4)
                .and_then(|view| crate::descriptor::host::buffer_range(&view))
                .map_or(0, |range| range.0),
            _ => 0,
        };
        if address == 0 {
            once!(slot, dword, "a shader takes the address of a buffer view that is not there: it reads 0");
        }
        if kind == USER_VIEW_ADDRESS {
            address as u32
        } else {
            (address >> 32) as u32
        }
    }

    /// For a shader that reads title memory by address
    /// ([`crate::memory`]): its directory, fault notes and view slots,
    /// with the pages of the root views the list has set made ready.
    fn memory_ready(&self, list: &HostList, compute: bool, writes: bool) -> Option<crate::memory::Ready> {
        let heap = match self.heaps[0] {
            0 => None,
            h => crate::descriptor::heap(h as This).host(),
        }?;
        let (mut addresses, mut constants) = (Vec::new(), Vec::new());
        for root in self.roots_ref(compute) {
            match root {
                Root::View(address) => addresses.push(*address),
                Root::Constants(values) => constants.extend_from_slice(values),
                _ => {}
            }
        }
        crate::memory::ready(list, heap, &addresses, &constants, writes)
    }

    /// The bound heap holding `handle`.
    fn heap_of(&self, handle: u64) -> Option<&'static crate::descriptor::Heap> {
        self.heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| h.index(handle).is_some())
    }

    /// Brings the host list up to date for a draw; false if the draw cannot
    /// run on the host.
    fn prepare(&mut self, list: &HostList, indexed: bool) -> bool {
        self.prepare_for(list, indexed, false)
    }

    /// As [`State::prepare`], for draws (`mesh` false) or mesh dispatches.
    fn prepare_for(&mut self, list: &HostList, indexed: bool, mesh: bool) -> bool {
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            return false;
        }
        let Some(p) = crate::pipeline::host_pipeline_for(self.pipeline as This, &self.target_formats) else {
            return false;
        };
        if p.compute || p.mesh != mesh {
            once!(mesh, "a draw or mesh dispatch with another kind of pipeline");
            return false;
        }
        if p.rects && self.topology != RECTANGLES {
            // Its geometry stage would turn each triangle into a rectangle.
            once!("a rectangle pipeline drawn with another topology");
        }
        if !self.bind(list, p, false) {
            return false;
        }
        let l = &list.list;
        // Vertex buffers the input layout reads.
        let mut views = [D3D12_VERTEX_BUFFER_VIEW::default(); 32];
        let mut count = 0;
        for (slot, (view, &v)) in views.iter_mut().zip(&self.vertex).enumerate() {
            if p.slots & 1 << slot == 0 {
                continue;
            }
            count = slot + 1;
            if v.address == 0 || v.size == 0 {
                continue;
            }
            let Some(shadow) = crate::shadow::find_any(v.address, u64::from(v.size)) else {
                continue;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            *view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                StrideInBytes: v.stride,
            };
        }
        // SAFETY: an open list; the views name live host buffers.
        unsafe {
            if count > 0 {
                l.IASetVertexBuffers(0, Some(&views[..count]));
            }
        }
        if indexed {
            let v = self.index;
            let Some(shadow) = crate::shadow::find_any(v.address, u64::from(v.size)) else {
                return false;
            };
            shadow.refresh(list, v.address, u64::from(v.size));
            let view = D3D12_INDEX_BUFFER_VIEW {
                BufferLocation: shadow.gpu(v.address),
                SizeInBytes: v.size,
                Format: DXGI_FORMAT(v.format as i32),
            };
            // SAFETY: an open list; the view names a live host buffer.
            unsafe { l.IASetIndexBuffer(Some(&view)) };
        }
        const RENDER_TARGET: u32 = 0x4;
        for &t in &self.targets {
            crate::list::ensure_state(list, t as This, RENDER_TARGET);
        }
        if self.depth != 0 {
            // A depth texture shaders also write: what they wrote, first.
            list.twin_use(self.depth as This, false, true);
            // A view that writes neither plane keeps the texture readable:
            // the depth read state, together with the shader read states
            // if this draw's shaders read the same texture (the write state
            // put it out of the state those views need, and a move to it
            // and back for every such draw).
            let state = if self.depth_state == DEPTH_READ {
                let (tested, read) = TESTED_DEPTH.get();
                if read && tested == self.depth {
                    DEPTH_READ | SHADER_RESOURCE
                } else {
                    DEPTH_READ
                }
            } else {
                DEPTH_WRITE
            };
            crate::list::ensure_state(list, self.depth as This, state);
        }
        true
    }

    /// A descriptor a shader reads through a table: its host slot made if
    /// it changed, constant data copied, textures made readable.
    /// What draws and dispatches share: heaps, our root signature, the
    /// pipeline, and per stage the user data, root views and the
    /// descriptors its tables hold.
    /// False if the draw or dispatch must not run: a root SRV it reads
    /// could not be given a view.
    pub(crate) fn bind(&mut self, list: &HostList, p: &HostPipeline, compute: bool) -> bool {
        let l = &list.list;
        // Textures picked by index, before anything asks for them.
        self.early_textures(list, p, compute);
        if crate::host::walls_trace() {
            self.walls_census(p, compute);
        }
        // Heaps first: a root signature with directly indexed heaps must be
        // set after them.
        if self.host_heaps != self.heaps {
            let heaps: Vec<Option<ID3D12DescriptorHeap>> = self
                .heaps
                .iter()
                .filter(|&&h| h != 0)
                .filter_map(|&h| crate::descriptor::heap(h as This).host())
                .map(|h| Some(h.heap.clone()))
                .collect();
            heaps.iter().flatten().for_each(|h| list.keep(h));
            // SAFETY: an open list and shader-visible heaps.
            unsafe { l.SetDescriptorHeaps(&heaps) };
            self.host_heaps = self.heaps;
            self.host_roots = [0; 2];
        }
        // The depth target a draw only tests, for the views of it that the
        // draw's shaders read (`readable`).
        let tested = if !compute && self.depth_state == DEPTH_READ {
            self.depth
        } else {
            0
        };
        TESTED_DEPTH.set((tested, false));
        let root = p.root.as_raw() as usize;
        if self.host_roots[usize::from(compute)] != root {
            // SAFETY: an open list and a live root signature.
            unsafe {
                if compute {
                    l.SetComputeRootSignature(&p.root);
                } else {
                    l.SetGraphicsRootSignature(&p.root);
                }
            }
            self.host_roots[usize::from(compute)] = root;
        }
        // By host object: a title pipeline released and another made at its
        // address is another host pipeline (the list keeps the old one).
        if self.host_pipeline != p.pso.as_raw() as usize {
            // SAFETY: an open list and a live pipeline.
            unsafe { l.SetPipelineState(&p.pso) };
            self.host_pipeline = p.pso.as_raw() as usize;
            list.used_pipeline(&p.name);
            // The title may release the pipeline before it submits the list.
            // (`VICTORY_LIST_REUSE`: kept at the list's first bind of it only.)
            if !crate::host::list_reuse() || list.first_bind(p.pso.as_raw() as usize) {
                list.keep(&p.pso);
                list.keep(&p.root);
            }
        }
        // With `table_stages`: what the stages of this draw read of each
        // table, and the tables its earlier stages made afresh (address,
        // slots).
        let spans = if table_stages() { stage_spans(p) } else { Vec::new() };
        let mut staged: Vec<(u64, u32)> = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            STORES.set(t.stores);
            let made = self.tables(list, &t.used, &t.host.root_indexed_views, &t.host.half_views, compute, &spans, &mut staged);
            STORES.set(true);
            if made.iter().any(|m| m.1 == u32::MAX) {
                once!("a draw whose descriptor table found no free slots is dropped");
                // Which draws those are, in the frame line's list of drops.
                #[cfg(feature = "diagnostics")]
                crate::pipeline::note_why(format!("no free dynamic slots for a table of {}", p.name));
                return false;
            }
            // What the shader stores through root views and raw addresses is
            // copied back into title memory afterwards.
            let writes = (t.stores || !write_skip())
                && !crate::pipeline::builder_quarantined(self.pipeline as This);
            // Title memory by address: what the shader finds it by.
            let memory = if t.host.root_srvs.contains(&victory_shader::emit::SLOT_MEMORY) {
                let Some(ready) = self.memory_ready(list, compute, writes) else {
                    once!("a shader that reads title memory by address without a view heap to hold its views: the draw is dropped");
                    return false;
                };
                Some(ready)
            } else {
                None
            };
            if let Some((param, dwords)) = stage.user_data {
                let mut values = vec![0u32; dwords as usize];
                for u in &t.host.user_data {
                    if let RootType::Other(kind) = u.root {
                        values[u.index as usize] =
                            self.host_word(compute, kind, u.slot, u.dword, memory.as_ref());
                        continue;
                    }
                    let fresh = made
                        .iter()
                        .find(|m| m.0 == u.slot && u.root == RootType::Table);
                    values[u.index as usize] = match fresh {
                        Some(&(_, offset)) => offset,
                        None => self.user_value(compute, u.slot, u.root, u.dword),
                    };
                }
                // SAFETY: an open list; `values` covers the parameter.
                unsafe {
                    if compute {
                        l.SetComputeRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    } else {
                        l.SetGraphicsRoot32BitConstants(param, dwords, values.as_ptr().cast(), 0);
                    }
                }
            }
            for (views, srv) in [(&stage.root_cbvs, false), (&stage.root_srvs, true)] {
                for &(slot, param) in views {
                    if srv && slot == victory_shader::emit::SLOT_MEMORY {
                        // Ours: the directory of title memory by address.
                        if let Some(m) = &memory {
                            // SAFETY: an open list; the directory lives as
                            // long as the process and the list holds it.
                            unsafe {
                                if compute {
                                    l.SetComputeRootShaderResourceView(param, m.directory);
                                } else {
                                    l.SetGraphicsRootShaderResourceView(param, m.directory);
                                }
                            }
                        }
                        continue;
                    }
                    if srv && slot == victory_shader::emit::SLOT_STRIDES {
                        // Ours, not the title's: the shader picks buffers
                        // from the view heap by an index it computes.
                        match self.heap_buffers(list) {
                            // SAFETY: an open list; the table lives as long
                            // as the heap the list holds.
                            Some(gpu) => unsafe {
                                if compute {
                                    l.SetComputeRootShaderResourceView(param, gpu);
                                } else {
                                    l.SetGraphicsRootShaderResourceView(param, gpu);
                                }
                            },
                            None => once!("a shader picks buffers from a view heap that is not set"),
                        }
                        continue;
                    }
                    let tabled = srv && crate::pipeline::host::bounded_root_srvs();
                    let Some(&Root::View(address)) = self.roots_ref(compute).get(usize::from(slot))
                    else {
                        once!(slot, "a shader reads a root view the title did not set");
                        // A table left unset is undefined: a null view.
                        if tabled && !self.root_srv_table(list, None, param, compute, false) {
                            return false;
                        }
                        continue;
                    };
                    // The shader adds the bytes a raw view begins early by.
                    let biased = srv
                        && t.host.user_data.iter().any(|u| {
                            u.slot == slot
                                && u.root == RootType::Other(victory_shader::emit::USER_VIEW_BIAS)
                        });
                    if tabled {
                        if !self.root_srv_table(list, Some(address), param, compute, biased) {
                            return false;
                        }
                        continue;
                    }
                    let Some(mut gpu) = root_view(list, address, srv) else {
                        continue;
                    };
                    if biased {
                        gpu -= u64::from(self.host_word(
                            compute,
                            victory_shader::emit::USER_VIEW_BIAS,
                            slot,
                            0,
                            None,
                        ));
                    }
                    // SAFETY: an open list; the address is in live memory.
                    unsafe {
                        match (compute, srv) {
                            (false, false) => l.SetGraphicsRootConstantBufferView(param, gpu),
                            (false, true) => l.SetGraphicsRootShaderResourceView(param, gpu),
                            (true, false) => l.SetComputeRootConstantBufferView(param, gpu),
                            (true, true) => l.SetComputeRootShaderResourceView(param, gpu),
                        }
                    }
                }
            }
            for &(slot, param) in &stage.root_uavs {
                if slot == victory_shader::emit::SLOT_FAULTS {
                    // Ours: where shaders note the pages they found empty.
                    if let Some(m) = &memory {
                        // SAFETY: an open list; the buffer lives as long as
                        // the process and the list holds it.
                        unsafe {
                            if compute {
                                l.SetComputeRootUnorderedAccessView(param, m.faults);
                            } else {
                                l.SetGraphicsRootUnorderedAccessView(param, m.faults);
                            }
                        }
                    }
                    continue;
                }
                let address = match self.roots_ref(compute).get(usize::from(slot)) {
                    Some(&Root::View(address)) => Some(address),
                    _ => {
                        once!(slot, "a shader writes through a root view the title did not set");
                        None
                    }
                };
                if !self.root_uav(list, address, param, compute, writes) {
                    return false;
                }
            }
        }
        // Depth textures it writes through unordered access: their colour
        // twins hold what they lack from here on.
        list.twins_written();
        true
    }

    /// Binds a root UAV: a raw unordered access view of the title buffer's
    /// host copy from `address` on, as a one-view table (a null view
    /// without an address, without a host copy shaders can write, or at an
    /// address not on 16 bytes), or as a root descriptor with
    /// `VICTORY_ROOT_SRV_DIRECT=1` (the dispatch is dropped then if there
    /// is nothing to bind). With `writes`, what the shader stored is copied
    /// back into title memory when the list has run. False if the draw
    /// must be dropped.
    /// shortcut: a root view has no size: the view, the refresh and the
    /// copy back cover all that is left of the title's buffer
    /// (`VICTORY_ROOT_UAV_MB` sets a limit for all three).
    fn root_uav(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
        writes: bool,
    ) -> bool {
        const UNORDERED: i32 = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0;
        static LIMIT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let limit = *LIMIT.get_or_init(|| {
            std::env::var("VICTORY_ROOT_UAV_MB")
                .ok()
                .and_then(|v| v.parse::<u64>().ok())
                .map_or(u64::MAX, |mb| mb << 20)
        });
        let find = |a: u64| {
            let s = crate::shadow::find(a, 1)?;
            // SAFETY: a live buffer.
            let flags = unsafe { s.buffer.GetDesc() }.Flags;
            if flags.0 & UNORDERED == 0 {
                once!("a root UAV of a buffer whose host copy shaders cannot write");
                return None;
            }
            Some(s)
        };
        let ready = |a: u64, s: &std::sync::Arc<crate::shadow::Shadow>| {
            let len = s.left(a).min(limit);
            s.refresh(list, a, len);
            if writes {
                s.gpu_writes(list, a, len);
            }
        };
        if !crate::pipeline::host::bounded_root_srvs() {
            let Some((a, s)) = address.and_then(|a| Some((a, find(a)?))) else {
                once!("a root UAV with nothing to bind: the draw is dropped");
                return false;
            };
            ready(a, &s);
            // From the 16 bytes below: the shader adds the rest.
            // SAFETY: a live buffer.
            let base = unsafe { s.buffer.GetGPUVirtualAddress() };
            let gpu = base + ((s.gpu(a) - base) & !15);
            // SAFETY: an open list; the address is in a live buffer.
            unsafe {
                if compute {
                    list.list.SetComputeRootUnorderedAccessView(param, gpu);
                } else {
                    list.list.SetGraphicsRootUnorderedAccessView(param, gpu);
                }
            }
            return true;
        }
        let heap = match self.heaps[0] {
            0 => None,
            h => Some(crate::descriptor::heap(h as This)),
        };
        let Some((heap, host)) = heap.and_then(|h| Some((h, h.host()?))) else {
            once!("a root UAV without a bound view heap: the draw is dropped");
            return false;
        };
        let key = (std::ptr::from_ref(heap) as usize, address.unwrap_or(0));
        let (index, shadow) = match self.root_uav_slots.get(&key) {
            Some((index, shadow)) => (*index, address.zip(shadow.clone())),
            None => {
                let shadow = address.and_then(|a| find(a).map(|s| (a, s)));
                let Some(index) = host.dynamic(1) else {
                    #[cfg(feature = "diagnostics")]
                    crate::pipeline::note_why("no free dynamic slot for the one-view table of a root view".to_owned());
                    return false;
                };
                self.dynamic.push((key.0, host.id(), index, 1));
                if let Some((a, s)) = &shadow {
                    // SAFETY: a live buffer.
                    let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                    let offset = s.gpu(*a) - base;
                    // Raw views start on 16 bytes: from the 16 bytes
                    // below, and the shader adds the rest to its offsets
                    // (`victory_shader::emit::USER_VIEW_BIAS`). (An address
                    // not on 16 bytes got a null view at first: the
                    // builder's counters are at such addresses.)
                    let bias = offset & 15;
                    host.raw_uav_at(index, &s.buffer, offset - bias, s.left(*a).min(limit) + bias);
                }
                if shadow.is_none() {
                    host.null_raw_uav_at(index);
                }
                self.root_uav_slots
                    .insert(key, (index, shadow.as_ref().map(|(_, s)| s.clone())));
                (index, shadow)
            }
        };
        if let Some((a, s)) = &shadow {
            ready(*a, s);
        }
        let handle = host.gpu_handle(index);
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }

    /// Binds a root SRV as a one-view table
    /// ([`crate::pipeline::host::bounded_root_srvs`]): a raw view of the
    /// title buffer's host copy from `address` to the copy's end, or a null
    /// view (no address, no host copy, or an address not on 16 bytes). One
    /// dynamic slot per address per list, given back with the list's
    /// others. False if no slot could be had or no view heap is bound.
    fn root_srv_table(
        &mut self,
        list: &HostList,
        address: Option<u64>,
        param: u32,
        compute: bool,
        biased: bool,
    ) -> bool {
        let heap = match self.heaps[0] {
            0 => None,
            h => Some(crate::descriptor::heap(h as This)),
        };
        let Some((heap, host)) = heap.and_then(|h| Some((h, h.host()?))) else {
            once!("a root SRV without a bound view heap: the draw is dropped");
            return false;
        };
        // A view that begins early is another view than one that does not
        // (title addresses stay below bit 63).
        let key = (
            std::ptr::from_ref(heap) as usize,
            address.unwrap_or(0) | u64::from(biased) << 63,
        );
        let (index, shadow) = match self.root_srv_slots.get(&key) {
            Some((index, shadow)) => (*index, address.zip(shadow.clone())),
            None => {
                let shadow = address.and_then(|a| crate::shadow::find(a, 1).map(|s| (a, s)));
                let Some(index) = host.dynamic(1) else {
                    #[cfg(feature = "diagnostics")]
                    crate::pipeline::note_why("no free dynamic slot for the one-view table of a root view".to_owned());
                    return false;
                };
                self.dynamic.push((key.0, host.id(), index, 1));
                self.root_srv_slots
                    .insert(key, (index, shadow.as_ref().map(|(_, s)| s.clone())));
                match &shadow {
                    Some((a, s)) => {
                        // SAFETY: a live buffer.
                        let base = unsafe { s.buffer.GetGPUVirtualAddress() };
                        let offset = s.gpu(*a) - base;
                        if biased {
                            // From the 16 bytes below; the shader adds the
                            // rest (`victory_shader::emit::USER_VIEW_BIAS`).
                            let bias = offset & 15;
                            host.raw_at(index, &s.buffer, offset - bias, s.left(*a) + bias);
                        } else if offset.is_multiple_of(16) {
                            host.raw_at(index, &s.buffer, offset, s.left(*a));
                        } else {
                            // Raw views start on 16 bytes; one from the
                            // aligned start would shift every read.
                            once!(
                                address = format_args!("{a:#x}"),
                                "a root SRV not on 16 bytes reads zeros"
                            );
                            host.null_raw_at(index);
                        }
                    }
                    None => host.null_raw_at(index),
                }
                (index, shadow)
            }
        };
        // As for root descriptors (`root_view`): all that is left of the
        // buffer is brought up to date. The first 1 MiB only left what lay
        // past it stale (most mesh vertex streams reach further).
        if let Some((a, s)) = &shadow {
            s.refresh(list, *a, s.left(*a));
        }
        let handle = host.gpu_handle(index);
        // SAFETY: an open list; the slot is in the bound view heap.
        unsafe {
            if compute {
                list.list.SetComputeRootDescriptorTable(param, handle);
            } else {
                list.list.SetGraphicsRootDescriptorTable(param, handle);
            }
        }
        true
    }

    /// For a shader that picks buffers from the view heap by an index it
    /// computes (SM 6.6 dynamic resources): the GPU address of the heap's
    /// table of element strides, with every buffer view of the heap given
    /// its host slot and its memory brought up to date when `list` is
    /// submitted. Tables tell us which views a draw reads; nothing tells
    /// us which of the heap's a computed index names, so all of them are
    /// made ready, once per list.
    /// shortcut: the buffers are read, not written: what such a shader
    /// writes through a picked unordered access view stays in the host
    /// copy. A list refreshes them where it first runs such a shader;
    /// title writes recorded later in the same list are not seen.
    fn heap_buffers(&self, list: &HostList) -> Option<u64> {
        let heap = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())?;
        let host = heap.host()?;
        let gpu = host.strides_gpu()?;
        // One key per list, as textures have: no title resource is there.
        const ONCE: usize = usize::MAX - 0x5e;
        if !list.first_use(ONCE) {
            return Some(gpu);
        }
        // The list may outlive the title's heap object, which owns the table.
        if let Some(buffer) = host.strides_buffer() {
            list.keep(buffer);
        }
        // The pass itself once per frame: per list it ran up to 51 times a
        // frame over 81,000 views and 4,100 host copies, 18 s of every
        // frame (2026-11-24, run 11).
        // shortcut: what the title writes into such buffers between lists
        // of one frame is seen a frame late.
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        let frame = crate::frame::current();
        if FRAME.swap(frame, Ordering::Relaxed) == frame {
            return Some(gpu);
        }
        // `VICTORY_HEAP_PASS_MARKED` (off unless set): the pass that looks
        // only at marked slots, in place of everything below.
        if crate::heap_pass::marked() {
            self.heap_buffers_marked(list, heap, host, frame);
            return Some(gpu);
        }
        // (`alt`, the marked pass's turn just over: the host copies it
        // kept are let go.)
        if crate::heap_pass::marked_paused() {
            *MARKED.lock().unwrap_or_else(|e| e.into_inner()) = None;
        }
        let started = std::time::Instant::now();
        // Per host heap: the host copy and range each slot's buffer view
        // reads, kept from pass to pass. Only the slots written since the
        // last pass are looked at again (83,000 slots a pass otherwise, 0.1 s
        // at best); all of them every 16 passes, because a view of a
        // released resource goes stale without a write, and a slot written
        // before its memory had a host copy is only made then.
        let mut kept = RANGES.lock().unwrap_or_else(|e| e.into_inner());
        let (passes, ranges) = kept
            .get_or_insert_with(Default::default)
            .entry(std::ptr::from_ref(host) as usize)
            .or_insert_with(|| (0, Ranges::new()));
        let changed = host.buffer_changes();
        let slots = if *passes % 16 == 0 {
            ranges.clear();
            host.buffer_slots()
        } else {
            changed
        };
        *passes += 1;
        // Whether the device was there before the pass: a view that removes
        // it is named below.
        static GONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let mut alive = !GONE.load(Ordering::Relaxed) && crate::descriptor::host::device_removed().is_none();
        HEAP_PASS_PARTS[1].fetch_add(slots.len() as u64, Ordering::Relaxed);
        for (n, &index) in slots.iter().enumerate() {
            ranges.remove(&index);
            let Some(handle) = heap.handle(index) else {
                continue;
            };
            let Some(view) = host.mirror(index, handle, true) else {
                continue;
            };
            // Every 64 views while the device lives (a cheap query).
            if alive
                && n % 64 == 63
                && let Some(why) = crate::descriptor::host::device_removed()
            {
                alive = false;
                GONE.store(true, Ordering::Relaxed);
                let last: Vec<String> = slots[n - 63..=n]
                    .iter()
                    .filter_map(|&i| crate::descriptor::view(heap.handle(i)?).map(|v| format!("{i}: {v:?}")))
                    .map(|s| s.chars().take(200).collect())
                    .collect();
                tracing::error!(why, slot = index, "the host device went while buffer views of the heap were made; one of: {last:#?}");
                crate::host::log_removal_context();
            }
            // The copy the slot shows (`buffer_copy`), not the one the
            // view's range alone would find.
            let Some((shadow, address, len)) = crate::descriptor::host::buffer_copy(&view) else {
                continue;
            };
            #[cfg(feature = "diagnostics")]
            if host.shows(index, &shadow.buffer) == Some(false) {
                HEAP_VIEWS_ELSEWHERE.fetch_add(1, Ordering::Relaxed);
            }
            ranges.insert(index, (shadow, address, address + len));
        }
        let views = ranges.len() as u32;
        // Per host copy, the span its picked views cover.
        let mut spans: std::collections::HashMap<usize, (std::sync::Arc<crate::shadow::Shadow>, u64, u64)> =
            std::collections::HashMap::new();
        for (shadow, from, to) in ranges.values() {
            let key = std::sync::Arc::as_ptr(shadow) as usize;
            let span = spans.entry(key).or_insert_with(|| (shadow.clone(), *from, *to));
            span.1 = span.1.min(*from);
            span.2 = span.2.max(*to);
        }
        drop(kept);
        let bytes: u64 = spans.values().map(|s| s.2 - s.1).sum();
        // Memory that is not write-watched is copied whole by every refresh.
        // It was once refreshed only every 120 frames (unmapped pages read
        // one by one cost 14 GB a frame in run 11, before they were
        // remembered): streamed geometry then drew from stale buffers, and
        // the garbage ended runs (the card stopped answering 14-90 frames
        // into the opening cutscene; with every frame it ran on). The pass
        // costs some 30 ms a frame this way.
        static WHOLE: std::sync::Mutex<Option<std::collections::HashMap<usize, u64>>> = std::sync::Mutex::new(None);
        let mut whole = WHOLE.lock().unwrap_or_else(|e| e.into_inner());
        let whole = whole.get_or_insert_with(Default::default);
        for (shadow, from, to) in spans.values() {
            if shadow.unwatched() {
                // By the copy's number: a copy made where one let go was
                // (`shadow::let_go_idle`) has never been refreshed.
                let key = shadow.id() as usize;
                // `VICTORY_UNWATCHED_FRAMES` sets the interval (1: every frame).
                static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
                let every = *EVERY.get_or_init(|| {
                    std::env::var("VICTORY_UNWATCHED_FRAMES")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(1)
                });
                match whole.get(&key) {
                    Some(&last) if frame < last + every => continue,
                    _ => {
                        whole.insert(key, frame);
                    }
                }
            }
            shadow.refresh(list, *from, to - from);
        }
        let textures_began = std::time::Instant::now();
        if !crate::heap_worker::kick(heap) {
            Self::heap_textures(Into::List(list), heap, host);
        }
        HEAP_PASS_PARTS[0].fetch_add(textures_began.elapsed().as_micros() as u64, Ordering::Relaxed);
        HEAP_BUFFER_PASSES.fetch_add(1, Ordering::Relaxed);
        HEAP_BUFFER_MICROS.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        static LOGGED: AtomicU64 = AtomicU64::new(0);
        // The first pass and every growth of the set by half.
        let seen = LOGGED.load(Ordering::Relaxed);
        if u64::from(views) > seen + seen / 2 {
            LOGGED.store(u64::from(views), Ordering::Relaxed);
            tracing::info!(
                slots = slots.len(),
                views,
                copies = spans.len(),
                mb = bytes >> 20,
                ms = started.elapsed().as_millis(),
                "buffer views of the heap made ready for shaders that pick them by index"
            );
        }
        Some(gpu)
    }

    /// [`State::heap_buffers`]' pass with `VICTORY_HEAP_PASS_MARKED`
    /// (`crate::heap_pass`, after fork pull request #523): a slot is looked
    /// at again only when its view was written, the table entry its record
    /// names was freed, its record in title memory is not the one seen at
    /// its last look, or it found no host copy (every 16 passes); every slot
    /// every 256 passes. A copy's span is worked out again only when one of
    /// its slots went. What follows the slots (the refreshes, the textures,
    /// the counts) is the other pass's, line for line.
    fn heap_buffers_marked(
        &self,
        list: &HostList,
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
        frame: u64,
    ) {
        let started = std::time::Instant::now();
        let mut kept = MARKED.lock().unwrap_or_else(|e| e.into_inner());
        let heaps = kept.get_or_insert_with(Default::default);
        // `alt`, back from the other pass's turn: what is kept here is from
        // before it, and it took the heap's changes meanwhile. Both start
        // anew: this one now, the other at its next turn (with no entry it
        // begins at pass 0, which looks at every slot).
        if crate::heap_pass::RESUMED.swap(false, Ordering::Relaxed) {
            heaps.clear();
            *RANGES.lock().unwrap_or_else(|e| e.into_inner()) = None;
        }
        // One list of freed table entries for all heaps: each heap's pass is told.
        let forgotten = crate::descriptor::take_forgotten();
        for picked in heaps.values_mut() {
            picked.note_forgotten(forgotten.as_deref());
        }
        let ranges = heaps.entry(std::ptr::from_ref(host) as usize).or_insert_with(|| Marked::new(false));
        let changed = host.buffer_changes();
        let record = |index: u32| heap.handle(index).map(crate::descriptor::record_at);
        let slots = ranges.plan(changed, || host.buffer_slots(), record, crate::descriptor::record_entry);
        // Whether the device was there before the pass: a view that removes
        // it is named below.
        static GONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
        let mut alive = !GONE.load(Ordering::Relaxed) && crate::descriptor::host::device_removed().is_none();
        HEAP_PASS_PARTS[1].fetch_add(slots.len() as u64, Ordering::Relaxed);
        // Of the slots looked at, those whose record is the one seen at
        // their last look (a view written again as it was).
        let mut same = 0u64;
        for (n, &index) in slots.iter().enumerate() {
            let Some(handle) = heap.handle(index) else {
                ranges.forget(index);
                continue;
            };
            // Read before the slot is made: a write after it is seen next pass.
            let record = crate::descriptor::record_at(handle);
            let entry = crate::descriptor::record_entry(record);
            same += u64::from(ranges.record_of(index) == Some(record));
            ranges.forget(index);
            let Some(view) = host.mirror(index, handle, true) else {
                continue;
            };
            // Every 64 views while the device lives (a cheap query).
            if alive
                && n % 64 == 63
                && let Some(why) = crate::descriptor::host::device_removed()
            {
                alive = false;
                GONE.store(true, Ordering::Relaxed);
                let last: Vec<String> = slots[n - 63..=n]
                    .iter()
                    .filter_map(|&i| crate::descriptor::view(heap.handle(i)?).map(|v| format!("{i}: {v:?}")))
                    .map(|s| s.chars().take(200).collect())
                    .collect();
                tracing::error!(why, slot = index, "the host device went while buffer views of the heap were made; one of: {last:#?}");
                crate::host::log_removal_context();
            }
            // The copy the slot shows (`buffer_copy`), not the one the
            // view's range alone would find.
            let Some((shadow, address, len)) = crate::descriptor::host::buffer_copy(&view) else {
                // A buffer view whose memory has no host copy (yet): tried
                // again every 16 passes. (Other views: left out until
                // written, as in the other pass.)
                if crate::descriptor::host::buffer_range(&view).is_some_and(|(_, len)| len > 0) {
                    ranges.orphan(index, record, entry);
                }
                continue;
            };
            #[cfg(feature = "diagnostics")]
            if host.shows(index, &shadow.buffer) == Some(false) {
                HEAP_VIEWS_ELSEWHERE.fetch_add(1, Ordering::Relaxed);
            }
            let key = std::sync::Arc::as_ptr(&shadow) as usize;
            ranges.keep(index, key, &shadow, address, address + len, record, entry);
        }
        let views = ranges.views() as u32;
        // Per host copy, the span its picked views cover.
        let spans = ranges.spans();
        drop(kept);
        let bytes: u64 = spans.iter().map(|s| s.2 - s.1).sum();
        // Memory that is not write-watched is copied whole by every refresh
        // (see the other pass).
        static WHOLE: std::sync::Mutex<Option<std::collections::HashMap<usize, u64>>> = std::sync::Mutex::new(None);
        let mut whole = WHOLE.lock().unwrap_or_else(|e| e.into_inner());
        let whole = whole.get_or_insert_with(Default::default);
        for (shadow, from, to) in &spans {
            if shadow.unwatched() {
                let key = shadow.id() as usize;
                // `VICTORY_UNWATCHED_FRAMES` sets the interval (1: every frame).
                static EVERY: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
                let every = *EVERY.get_or_init(|| {
                    std::env::var("VICTORY_UNWATCHED_FRAMES")
                        .ok()
                        .and_then(|v| v.parse().ok())
                        .unwrap_or(1)
                });
                match whole.get(&key) {
                    Some(&last) if frame < last + every => continue,
                    _ => {
                        whole.insert(key, frame);
                    }
                }
            }
            shadow.refresh(list, *from, *to - *from);
        }
        let textures_began = std::time::Instant::now();
        if !crate::heap_worker::kick(heap) {
            Self::heap_textures(Into::List(list), heap, host);
        }
        HEAP_PASS_PARTS[0].fetch_add(textures_began.elapsed().as_micros() as u64, Ordering::Relaxed);
        HEAP_BUFFER_PASSES.fetch_add(1, Ordering::Relaxed);
        HEAP_BUFFER_MICROS.fetch_add(started.elapsed().as_micros() as u64, Ordering::Relaxed);
        // Every 256 passes: how many of the slots looked at held the record
        // of their last look (what a further cut could leave alone).
        static SEEN: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];
        SEEN[1].fetch_add(slots.len() as u64, Ordering::Relaxed);
        SEEN[2].fetch_add(same, Ordering::Relaxed);
        if SEEN[0].fetch_add(1, Ordering::Relaxed) % 256 == 255 {
            tracing::info!(
                "the marked pass over the heap's buffer views (VICTORY_HEAP_PASS_MARKED), its last 256 passes: {} slots looked at, {} of them held the record seen at their last look; {} views kept now",
                SEEN[1].swap(0, Ordering::Relaxed),
                SEEN[2].swap(0, Ordering::Relaxed),
                views,
            );
        }
        static LOGGED: AtomicU64 = AtomicU64::new(0);
        // The first pass and every growth of the set by half.
        let seen = LOGGED.load(Ordering::Relaxed);
        if u64::from(views) > seen + seen / 2 {
            LOGGED.store(u64::from(views), Ordering::Relaxed);
            tracing::info!(
                slots = slots.len(),
                views,
                copies = spans.len(),
                mb = bytes >> 20,
                ms = started.elapsed().as_millis(),
                "buffer views of the heap made ready for shaders that pick them by index (the marked pass)"
            );
        }
    }

    /// Runs the pass that uploads textures picked by index
    /// ([`State::heap_textures`]) where nothing has asked for it yet
    /// ([`early_budgets`]): once a frame for a few milliseconds, at the
    /// frame's first draw with a view heap set, and with a larger budget
    /// of its own before every draw whose pixel shader is one of the
    /// title's bake programs. The uploads are recorded into `list` ahead
    /// of the draw being bound, as the world's pass records them.
    /// Then, for a draw of the title's skin bake, what
    /// `VICTORY_BAKE_SOURCES` asks for its own textures
    /// ([`State::bake_sources_pass`]).
    fn early_textures(&self, list: &HostList, p: &HostPipeline, compute: bool) {
        if compute {
            return;
        }
        self.early_textures_budgeted(list, p);
        if bake_sources_mode().0 != BakeMode::Off {
            self.bake_sources_pass(list, p);
        }
    }

    /// [`State::early_textures`]' runs within budgets of their own.
    fn early_textures_budgeted(&self, list: &HostList, p: &HostPipeline) {
        let (early_ms, bake_ms) = early_budgets();
        if early_ms == 0 && bake_ms == 0 {
            return;
        }
        let bake = bake_ms > 0
            && p.stages.iter().any(|s| BAKE_PIXEL_SHADERS.contains(&s.shader.hash.as_str()));
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        // The milliseconds this frame's bake draws have taken so far: a
        // frame has several, and together they get `bake_ms`.
        static BAKED_MS: AtomicU64 = AtomicU64::new(0);
        let frame = crate::frame::current();
        let first = FRAME.swap(frame, Ordering::Relaxed) != frame;
        if first {
            BAKED_MS.store(0, Ordering::Relaxed);
        }
        let left = bake_ms.saturating_sub(BAKED_MS.load(Ordering::Relaxed));
        let budget = if bake && left > 0 {
            // What a bake may read must be there now: up to 2 GiB.
            (left, 2 << 30)
        } else if first && early_ms > 0 {
            (early_ms, 64 << 20)
        } else {
            return;
        };
        let Some(heap) = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())
        else {
            return;
        };
        let Some(host) = heap.host() else {
            return;
        };
        let bake = bake && left > 0;
        // What can wait for a later list is the worker's.
        if !bake && crate::heap_worker::kick(heap) {
            return;
        }
        let _pass = crate::heap_worker::on().then(crate::heap_worker::urgent);
        let n = EARLY_TEXTURE_PASSES[usize::from(bake)].fetch_add(1, Ordering::Relaxed) + 1;
        if bake && n.is_power_of_two() {
            let [slots, due, ..] = HEAP_TEXTURES_QUEUE.each_ref().map(|c| c.load(Ordering::Relaxed));
            tracing::info!(
                frame,
                bakes = n,
                queued_at_the_last_pass = slots,
                due_at_the_last_pass = due,
                uploaded_so_far = HEAP_TEXTURES.load(Ordering::Relaxed),
                "a draw of the title's decoration bake: textures picked by index are uploaded first (VICTORY_BAKE_TEXTURES_MS=0: not)"
            );
        }
        let began = std::time::Instant::now();
        PASS_BUDGET.set(Some(budget));
        Self::heap_textures(Into::List(list), heap, host);
        PASS_BUDGET.set(None);
        if bake {
            BAKED_MS.fetch_add(began.elapsed().as_millis() as u64, Ordering::Relaxed);
        }
    }

    /// For a draw of the title's skin bake (any pixel shader with the
    /// bake's tables, [`bake_tables`]): the textures it samples by index,
    /// read from its own constants and records, after the budgeted runs
    /// above. `VICTORY_BAKE_SOURCES=count`: counted and logged
    /// ([`State::log_bake_sources`]). `=1`: those still waiting in the
    /// queue are also uploaded now, into `list` ahead of the draw, while
    /// the card has room, up to `VICTORY_BAKE_SOURCES_MB` a draw. That run
    /// is this thread's: the worker, if it runs, stops and hands over what
    /// it recorded first ([`crate::heap_worker::urgent`]), as for the
    /// budgeted bake run. A source missing here is baked as nothing for the
    /// rest of the run: black or blotched skin.
    fn bake_sources_pass(&self, list: &HostList, p: &HostPipeline) {
        let (mode, _) = bake_sources_mode();
        let Some(tables) = bake_tables(p) else {
            return;
        };
        let Some(heap) = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())
        else {
            return;
        };
        let Some(host) = heap.host() else {
            return;
        };
        let Some(sources) = self.bake_sources(tables, heap, host) else {
            return;
        };
        let waiting = !sources.queued.is_empty() || !sources.unmade.is_empty();
        let mut done = [0u64; 4];
        let mut skipped = false;
        if matches!(mode, BakeMode::Upload | BakeMode::Full) && waiting {
            if mode == BakeMode::Full || crate::resource::room() {
                let _pass = crate::heap_worker::on().then(crate::heap_worker::urgent);
                host.texture_slots_requeue(&sources.unmade, crate::frame::current());
                PASS_SOURCES.set(Some(sources.queued.iter().chain(&sources.unmade).copied().collect()));
                PASS_SOURCED.set([0; 4]);
                // No budget of the frame's: the run serves the sources
                // alone, within their own bytes.
                PASS_BUDGET.set(Some((0, 0)));
                Self::heap_textures(Into::List(list), heap, host);
                PASS_BUDGET.set(None);
                PASS_SOURCES.set(None);
                done = PASS_SOURCED.get();
            } else {
                skipped = true;
            }
        }
        Self::log_bake_sources(p, &sources, done, Self::bake_missing(&sources, host), skipped);
    }

    /// The textures a draw of the title's skin bake samples
    /// ([`BakeTables`]), by what the queue of textures picked by index
    /// holds of them. `None`: the draw's table does not hold what a bake's
    /// does.
    fn bake_sources(
        &self,
        tables: BakeTables,
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
    ) -> Option<BakeSources> {
        let Some(&Root::Table(table)) = self.roots_ref(false).get(usize::from(tables.slot)) else {
            return None;
        };
        let View::Constant(constants, size) = crate::descriptor::view(table + u64::from(tables.constants) * 4)?
        else {
            return None;
        };
        let mut words = [0u8; 0x80];
        if (size as usize) < words.len() || !crate::host::read_into(constants as *const u8, &mut words) {
            return None;
        }
        let (address, len) = crate::descriptor::view(table + u64::from(tables.records) * 4)
            .and_then(|view| crate::descriptor::host::buffer_range(&view))
            .unwrap_or((0, 0));
        let held = u32::try_from(len / BAKE_RECORD).unwrap_or(u32::MAX);
        let (named, records) = bake_named(&words, held, |i| {
            let mut record = [0u8; 0x44];
            crate::host::read_into((address + u64::from(i) * BAKE_RECORD) as *const u8, &mut record).then_some(record)
        });
        let mut sources = BakeSources { records, ..Default::default() };
        let mut textures = Vec::new();
        for index in named {
            let Some(handle) = heap.handle(index) else {
                continue;
            };
            if crate::descriptor::view(handle).is_some_and(|view| crate::descriptor::host::is_texture_view(&view)) {
                textures.push((index, handle));
            }
        }
        let indices: Vec<u32> = textures.iter().map(|t| t.0).collect();
        sources.queued = host.texture_slots_queued(&indices).into_iter().map(|q| q.0).collect();
        sources.textures = textures;
        for &(index, handle) in &sources.textures {
            if sources.queued.contains(&index) {
                continue;
            }
            if host.texture_slot_made_from(index, handle) {
                sources.ready += 1;
            } else if let Some(View::Placed(false, resource, desc)) = crate::descriptor::view(handle)
                && crate::texture::mapped_bytes(
                    (u64::from(desc[10]) | u64::from(desc[11]) << 32) as *const u8,
                    crate::resource::placed_bytes(&resource).unwrap_or(1) as usize,
                ) > 0
            {
                sources.unmade.push(index);
            } else {
                sources.other += 1;
            }
        }
        Some(sources)
    }

    /// Of a bake draw's textures, the ones the draw samples wrong: placed
    /// by the title in memory that is there, with a host slot that shows
    /// nothing of them. Counted when the runs before the draw are done.
    fn bake_missing(sources: &BakeSources, host: &crate::descriptor::host::HostHeap) -> u64 {
        let missing = |&&(index, handle): &&(u32, u64)| {
            !host.texture_slot_made_from(index, handle)
                && matches!(
                    crate::descriptor::view(handle),
                    Some(View::Placed(false, resource, desc)) if crate::texture::mapped_bytes(
                        (u64::from(desc[10]) | u64::from(desc[11]) << 32) as *const u8,
                        crate::resource::placed_bytes(&resource).unwrap_or(1) as usize,
                    ) > 0
                )
        };
        sources.textures.iter().filter(missing).count() as u64
    }

    /// The lines of `VICTORY_BAKE_SOURCES`: one for each of the first 64
    /// bake draws that sample a texture wrong ([`State::bake_missing`]),
    /// and while bake draws come, one a second with the totals so far.
    /// `skipped`: the upload was not made, the card having no room.
    fn log_bake_sources(p: &HostPipeline, sources: &BakeSources, done: [u64; 4], missing: u64, skipped: bool) {
        // Bake draws; of them sampling something wrong; sources waiting at
        // the draw; made stale; sampled wrong; uploaded; their bytes;
        // uploads not made for lack of room.
        static TOTALS: [AtomicU64; 8] = [const { AtomicU64::new(0) }; 8];
        static LINES: AtomicU64 = AtomicU64::new(0);
        static SAID_MS: AtomicU64 = AtomicU64::new(0);
        let waiting = sources.queued.len() as u64;
        let unmade = sources.unmade.len() as u64;
        let [uploaded, bytes, unmapped, as_before] = done;
        for (total, now) in TOTALS.iter().zip([
            1,
            u64::from(missing > 0),
            waiting,
            unmade,
            missing,
            uploaded,
            bytes,
            u64::from(skipped),
        ]) {
            total.fetch_add(now, Ordering::Relaxed);
        }
        let frame = crate::frame::current();
        if missing > 0 && LINES.fetch_add(1, Ordering::Relaxed) < 64 {
            let pixel = p.stages.iter().find(|s| s.shader.stage == "pixel").map_or("", |s| s.shader.hash.as_str());
            tracing::warn!(
                frame,
                pixel,
                records = sources.records.0,
                records_read = sources.records.1,
                textures = sources.textures.len(),
                on_the_card = sources.ready,
                waiting,
                made_stale = unmade,
                the_host_s_own = sources.other,
                uploaded,
                kilobytes = bytes >> 10,
                as_far_as_mapped_already = as_before,
                not_mapped = unmapped,
                skipped_no_room = skipped,
                sampled_wrong = missing,
                "a skin-bake draw samples textures that are not on the card (VICTORY_BAKE_SOURCES)"
            );
        }
        let now = crate::host::now_ms();
        let said = SAID_MS.load(Ordering::Relaxed);
        if now.saturating_sub(said) >= 1000
            && SAID_MS.compare_exchange(said, now, Ordering::Relaxed, Ordering::Relaxed).is_ok()
        {
            let [draws, short, waited, stale, wrong, ups, up_bytes, no_room] =
                TOTALS.each_ref().map(|t| t.load(Ordering::Relaxed));
            tracing::info!(
                frame,
                bake_draws = draws,
                draws_sampling_wrong = short,
                sources_waiting = waited,
                made_stale = stale,
                sampled_wrong = wrong,
                uploaded = ups,
                megabytes = up_bytes >> 20,
                skipped_no_room = no_room,
                "skin-bake draws and their own textures, so far (VICTORY_BAKE_SOURCES)"
            );
        }
    }

    /// The queue's side of the card accounting line
    /// (`crate::resource::card_accounting`, which says what the line
    /// holds): the slots that are due, by what their texture has on the
    /// card. Off unless `VICTORY_CARD_ACCOUNTING=1`; then at the first pass
    /// that finds the card without room, and every five seconds after it.
    /// Nothing is made, uploaded or counted as used here. For 18,000 due
    /// slots: 18,000 looks at a title slot and at the texture it views, a
    /// lock each, some milliseconds of the thread that records the frame.
    fn card_accounting(
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
        frame: u64,
        room: bool,
    ) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_CARD_ACCOUNTING").is_some_and(|v| v == "1")) {
            return;
        }
        // When the next line is due (0: none yet, the card has had room).
        static NEXT: AtomicU64 = AtomicU64::new(0);
        let now = crate::host::now_ms().max(1);
        let next = NEXT.load(Ordering::Relaxed);
        // (With levels on demand the card may never be without room: the
        // lines then start with the first texture made that way.)
        let by_level = crate::tiles::STATS[4].load(Ordering::Relaxed) > 0;
        let due_now = if next == 0 { !room || by_level } else { now >= next };
        if !due_now || NEXT.compare_exchange(next, now + 5000, Ordering::Relaxed, Ordering::Relaxed).is_err() {
            return;
        }
        // Due slots; of them uploaded before; of them viewing nothing the
        // title fills; and the textures (each once) whose host copy is
        // alive, with their bytes on the card.
        let mut due = [0u64; 5];
        // Textures (each once) that never had a host copy, and ones whose
        // host copy was let go: where their memory is, and its bytes.
        let (mut never, mut gone) = (Vec::new(), Vec::new());
        let mut seen = std::collections::BTreeSet::new();
        for (index, uploaded) in host.texture_slots_all_due(frame) {
            due[0] += 1;
            due[1] += u64::from(uploaded);
            let state = match heap.handle(index).and_then(crate::descriptor::view) {
                Some(View::Placed(false, resource, desc)) if resource[0] != 1 => {
                    let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                    match crate::resource::placed_known(address, &resource) {
                        Some(texture) => crate::resource::copy_state(texture),
                        // No title texture was made for it: the pass never
                        // got as far as making one.
                        None => Some((0, crate::resource::placed_bytes(&resource).unwrap_or(0), address)),
                    }
                }
                Some(View::Shader(object, _)) if crate::resource::title_fills(object as This) => {
                    crate::resource::copy_state(object as This)
                }
                _ => None,
            };
            let Some((state, bytes, memory)) = state else {
                due[2] += 1;
                continue;
            };
            if !seen.insert(memory) {
                continue;
            }
            match state {
                2 => {
                    due[3] += 1;
                    due[4] += bytes;
                }
                1 => gone.push((memory, bytes)),
                _ => never.push((memory, bytes)),
            }
        }
        crate::resource::card_accounting(room, due, never, gone);
    }

    /// The texture side of [`State::heap_buffers`]: textures the title fills
    /// itself (streamed pages, layout-library copies) whose views sit in the
    /// view heap are created (placed textures have no host texture until
    /// something reads them) and uploaded, as a table read would do, for
    /// shaders that pick them by a computed index. Within a budget of bytes
    /// per frame and in all; the rest waits for later lists and frames.
    /// The most recently written slots first (`TextureQueue::due`); slots
    /// draws read through tables are not in the queue.
    /// shortcut: uploaded once, and again only when the title rewrites the
    /// slot. A placed texture not all mapped yet is uploaded with what is
    /// there and looked at again (1, 2, 4, ... frames later, at most about
    /// a second): uploaded again when more of it is mapped. Pages mapped
    /// in place of others, and contents changed with no descriptor write,
    /// are missed (hook the page-mapping events for those).
    /// `VICTORY_HEAP_TEXTURES=0` turns it off.
    /// On a thread of its own the pass records into lists of its own, and
    /// the slots show what they upload from then on
    /// (`crate::heap_worker`).
    pub(crate) fn heap_textures(
        mut into: Into<'_>,
        heap: &crate::descriptor::Heap,
        host: &crate::descriptor::host::HostHeap,
    ) {
        static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        if !*ON.get_or_init(|| std::env::var_os("VICTORY_HEAP_TEXTURES").is_none_or(|v| v != "0")) {
            return;
        }
        /// Bytes of texture memory staged per frame.
        // `VICTORY_TEXTURE_STAGE_MB` raises it: a backlog of tens of thousands of
        // textures (a scene's first minutes) then drains in fewer frames.
        static PER_FRAME_MB: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let per_frame: u64 = *PER_FRAME_MB.get_or_init(|| {
            std::env::var("VICTORY_TEXTURE_STAGE_MB").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(192)
        }) << 20;
        static FRAME: AtomicU64 = AtomicU64::new(u64::MAX);
        static SPENT: AtomicU64 = AtomicU64::new(0);
        let frame = crate::frame::current();
        if FRAME.swap(frame, Ordering::Relaxed) != frame {
            SPENT.store(0, Ordering::Relaxed);
        }
        // No more of them are made while the card has no room
        // (`resource::room`): by what is alive now, not by what was ever
        // staged (4 GB in all, after which later textures stayed empty for
        // the run). The ones nothing names any more are let go, which
        // makes room again: so the slots are still looked at (the pass
        // returned here, and a slot whose texture had been streamed out
        // kept its host copy for as long as the card was full).
        // A skin-bake draw's own textures (`State::bake_sources_pass`): a
        // run for them alone, on the thread that records the draw. Nothing
        // of the queue's upkeep is done in it, which a pass of this frame
        // did or will do.
        let sources = PASS_SOURCES.take();
        let only = sources.is_some();
        // A bake draw's own run with `VICTORY_BAKE_SOURCES=full` goes on
        // as if the card had room, within its own bytes.
        let forced = only && bake_sources_mode().0 == BakeMode::Full;
        let mut room = forced || crate::resource::room();
        // Bytes uploaded by this pass since the card was last asked
        // whether it has room: asked again every 32 MiB (a pass may upload
        // 192 MB and more, and one asking a pass let the queue run some
        // 400 MB past its line while the world loaded: run 193).
        let mut since_asked = 0u64;
        if !room && !only {
            HEAP_TEXTURES_FULL.fetch_add(1, Ordering::Relaxed);
            once!("textures picked by index: the card has no room for more; they wait");
        }
        let mut refused = 0;
        if !only {
            // Room again: the slots held back while the card was full are
            // due now (`full_card_pass`).
            if room {
                host.texture_slots_room(frame);
            }
            // Slots whose texture's memory the runtime mapped, unmapped or
            // wrote since the last pass are due again.
            host.texture_slots_remapped(frame);
            // Levels on demand: the slots of textures whose first level
            // with memory changed since the last pass are made again from
            // it.
            host.texture_views_remade();
            // And with room, the textures whose upload left a held level
            // without the card's memory are uploaded again.
            if room {
                host.texture_slots_waiting(frame);
            }
            // Slots that read zeros for good and nothing asks for again
            // (`VICTORY_REASK_BLANK=1`; off by default; diagnostics builds).
            #[cfg(feature = "diagnostics")]
            host.texture_slots_blank(frame);
        }
        let due = match &sources {
            // The bake draw's own, due or not, held or not.
            Some(slots) => host.texture_slots_queued(slots),
            None => {
                let (due, census) = host.texture_slots_due(frame, 4096);
                // What the slots shaders pick by index show, every five seconds.
                #[cfg(feature = "diagnostics")]
                host.placed_slot_census();
                HEAP_TEXTURES_WAITING.store(census[0], Ordering::Relaxed);
                for (count, now) in HEAP_TEXTURES_QUEUE.iter().zip(census) {
                    count.store(now, Ordering::Relaxed);
                }
                // Where the card's memory is, and what the due slots would
                // need (`VICTORY_CARD_ACCOUNTING=1`; a line when the card
                // first has no room, and every five seconds after it).
                Self::card_accounting(heap, host, frame, room);
                due
            }
        };
        // What a run for a bake draw's own may upload.
        let sourced_at_most = bake_sources_mode().1;
        let mut sourced_bytes = 0u64;
        // The frames in about four seconds, between 16 and 256: how long a
        // slot with nothing to upload waits at most for its next look. In
        // frames alone it was 120: four minutes at the opening cutscene's
        // half a frame per second, while the title streams by the clock.
        // (About a second before the runtime told of the memory it maps:
        // the looks are now only for what that does not tell.)
        static PASSED: AtomicU64 = AtomicU64::new(0);
        let now = crate::host::now_ms();
        // (A run for a bake draw's own leaves the time between passes as
        // it is.)
        let took = if only { 100 } else { now.saturating_sub(PASSED.swap(now, Ordering::Relaxed)).max(1) };
        // The worker may run two passes in a frame: no shorter than that.
        let took = if matches!(into, Into::Ahead(_)) { took.max(100) } else { took };
        let most = (4000 / took).clamp(16, 256);
        // What a table entry of a texture a shader samples says.
        let sampled = UsedResource {
            kind: ResourceType::Srv,
            root: RootType::Table,
            table_offset_dwords: 0,
            lower_bound: 0,
            stride: 0,
            dwords: 8,
            user_data: 0,
            slot: 0,
        };
        // Also within a time per pass: the pass runs on a thread that
        // records the frame, and making and staging 192 MB of textures took
        // 1.6-2.2 s of each of the world's first twenty frames (run 20).
        // The rest waits for the next frames, as it does past the bytes.
        static SPAN_MS: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
        let span = std::time::Duration::from_millis(*SPAN_MS.get_or_init(|| {
            std::env::var("VICTORY_TEXTURE_STAGE_MS").ok().and_then(|v| v.trim().parse().ok()).unwrap_or(250)
        }));
        // A backlog (a scene's first minutes: 22,000 slots due, 2-3 minutes of black
        // surfaces at 192 MB a frame) gets three times the bytes and the time until it
        // drains; `VICTORY_TEXTURE_CATCHUP=0` keeps the plain budget.
        static CATCHUP: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
        let catchup = *CATCHUP.get_or_init(|| std::env::var_os("VICTORY_TEXTURE_CATCHUP").is_none_or(|v| v != "0"));
        let (per_frame, span) = if catchup && due.len() >= 1024 { (per_frame * 3, span * 3) } else { (per_frame, span) };
        // A run made early (`State::early_textures`) has a budget of its
        // own: its bytes on top of what the frame has spent so far.
        // (Not `most`: that is the longest wait between looks, used below.)
        let (span, bytes_at_most) = match PASS_BUDGET.get() {
            Some((ms, bytes)) => {
                (std::time::Duration::from_millis(ms), SPENT.load(Ordering::Relaxed).saturating_add(bytes))
            }
            None => (span, per_frame),
        };
        let began = std::time::Instant::now();
        for (index, write) in due {
            // A run for a bake draw's own ends at its own bytes, not at the
            // frame's budget or time.
            if only {
                if sourced_bytes >= sourced_at_most {
                    once!("a skin-bake draw samples more than VICTORY_BAKE_SOURCES_MB of textures not on the card yet: the rest take their turn");
                    break;
                }
            } else if SPENT.load(Ordering::Relaxed) >= bytes_at_most || began.elapsed() > span {
                break;
            }
            let ahead = matches!(into, Into::Ahead(_));
            if ahead && crate::heap_worker::make_way() {
                break;
            }
            // The record the slot has now, and the texture it names held
            // while a thread of ours works on it. (Released meanwhile, or
            // not a texture: nothing to upload.)
            let record = heap.handle(index).and_then(crate::descriptor::view_and_key);
            let _held = match &record {
                Some((View::Shader(object, _), _)) if ahead => {
                    let held = crate::resource::hold(*object as This);
                    if held.is_none() {
                        host.texture_slot_done(index, write, None);
                        continue;
                    }
                    held
                }
                _ => None,
            };
            // A placed texture whose memory is not mapped (not streamed in
            // yet, or streamed out again: the heap keeps tens of thousands
            // of such views) gets no host texture; looked at again soon (a
            // title writes a texture's view before its pages are mapped,
            // and the pass runs early in a frame). One of which some is
            // mapped (a texture streaming in: its smaller mips lie at the
            // start of its memory and arrive first) is uploaded with what
            // is there, and again when more is. (It was uploaded again
            // every 120 frames, changed or not; and asked by its first
            // page alone, one whose first page is the one missing counted
            // as not there.)
            // A slot uploaded whole is looked at again a few times
            // (`RECHECK_FRAMES`): the title writes a texture's view, and
            // may fill the texture's memory itself only afterwards, with
            // no descriptor write and nothing from the runtime to tell of
            // it. The host copy then kept what the memory held at the
            // first look for the rest of the run. The look reads 48 KiB;
            // only a changed texture is uploaded again.
            // `VICTORY_HEAP_RECHECK=0`: uploaded once, as before.
            // One found changed by such a look is looked at again in the
            // next frame with `VICTORY_RECHECK_SOON=1` ([`recheck_soon`]).
            let mut again_in = RECHECK_FRAMES[0];
            if let Some((at, len, print, looks, uploaded)) = host.texture_slot_checked(index) {
                if crate::texture::fingerprint(at, len) == print {
                    HEAP_TEXTURES_RECHECKED[0].fetch_add(1, Ordering::Relaxed);
                    match RECHECK_FRAMES.get(usize::from(looks) + 1) {
                        Some(wait) => {
                            host.texture_slot_recheck(index, write, frame + wait, (at, len), print, looks + 1, uploaded)
                        }
                        // The schedule is over: looked at again every so many
                        // frames for as long as the slot shows it
                        // (`recheck_every`), or never again.
                        None => match recheck_every() {
                            0 => host.texture_slot_done(index, write, None),
                            every => host.texture_slot_recheck(index, write, frame + every, (at, len), print, looks, uploaded),
                        },
                    }
                    continue;
                }
                if recheck_soon() {
                    again_in = 1;
                }
                let n = HEAP_TEXTURES_RECHECKED[1].fetch_add(1, Ordering::Relaxed);
                if n < 200 || n.is_multiple_of(1000) {
                    tracing::warn!(
                        slot = index,
                        address = format_args!("{at:#x}"),
                        bytes = len,
                        frames_after_upload = frame.saturating_sub(uploaded),
                        looks,
                        changed = n + 1,
                        looked_at_next_frame = HEAP_TEXTURES_SOON.load(Ordering::Relaxed),
                        "a texture picked by index changed after its upload with nothing to tell of it: uploaded again"
                    );
                }
            }
            let mut partly = None;
            if let Some((View::Placed(false, resource, desc), _)) = &record {
                let address = u64::from(desc[10]) | u64::from(desc[11]) << 32;
                let len = crate::resource::placed_bytes(resource).unwrap_or(1);
                let mapped = crate::texture::mapped_bytes(address as *const u8, len as usize);
                if mapped == 0 {
                    if only {
                        let [uploaded, bytes, unmapped, as_before] = PASS_SOURCED.get();
                        PASS_SOURCED.set([uploaded, bytes, unmapped + 1, as_before]);
                    }
                    HEAP_TEXTURES_AWAY.fetch_add(1, Ordering::Relaxed);
                    // Streamed out: what the slot showed is let go.
                    host.texture_slot_gone(index);
                    host.texture_slot_later(index, write, frame, most);
                    continue;
                }
                if mapped < len {
                    // (One of a bake draw's own is left so only if its host
                    // slot shows what was uploaded then.)
                    if mapped <= host.texture_slot_mapped(index)
                        && !(only
                            && !heap.handle(index).is_some_and(|handle| host.texture_slot_made_from(index, handle)))
                    {
                        if only {
                            let [uploaded, bytes, unmapped, as_before] = PASS_SOURCED.get();
                            PASS_SOURCED.set([uploaded, bytes, unmapped, as_before + 1]);
                        }
                        HEAP_TEXTURES_PARTLY[0].fetch_add(1, Ordering::Relaxed);
                        host.texture_slot_later(index, write, frame, most);
                        continue;
                    }
                    if !crate::texture::committed(address as *const u8, 1) {
                        HEAP_TEXTURES_PARTLY[2].fetch_add(1, Ordering::Relaxed);
                    }
                    partly = Some(mapped);
                }
            }
            // A texture object the host renders to or writes holds the
            // host's own contents: nothing to bring up to date, and this
            // pass is no use of it (its host copy is made when something
            // uses it, and its slots then: `descriptor::make_waiting`).
            if let Some((View::Shader(object, _), _)) = &record
                && crate::resource::buffer_bytes(*object as This).is_none()
                && !crate::resource::title_fills(*object as This)
            {
                host.texture_slot_done(index, write, None);
                continue;
            }
            // From here on a host texture is made: not without room. The
            // slot stays due.
            // Not so a slot already made from this very view: its host
            // texture exists, and uploading it again takes none of the
            // card's memory. It was refused all the same, so a texture the
            // title filled after its first upload kept its empty host copy
            // from the moment the card reached the line, which is the first
            // world frame in eight launches of nine: the same slots were
            // found changed again at every pass and never uploaded (run
            // 179: 33,000 such findings), and the main character's face
            // had its colour only in the launches that entered the world
            // under the line. `VICTORY_REUPLOAD_WHEN_FULL=0`: refused, as
            // before.
            if !room {
                let has_its_texture = reupload_when_full()
                    && heap.handle(index).is_some_and(|handle| host.texture_slot_made_from(index, handle));
                if !has_its_texture {
                    refused += 1;
                    if full_card_pass() {
                        // Held back for the frames of about four seconds
                        // (`most`), or until the card has room; the pass
                        // goes on, within a count that bounds one pass.
                        host.texture_slot_held(index, write, frame + most);
                        if refused > 1024 {
                            break;
                        }
                    } else if refused > 256 {
                        break;
                    }
                    continue;
                }
                let n = HEAP_TEXTURES_REUPLOADED_FULL.fetch_add(1, Ordering::Relaxed) + 1;
                if n == 1 || n.is_multiple_of(500) {
                    tracing::info!(
                        so_far = n,
                        slot = index,
                        "a texture already on the card is brought up to date although the card has no room for new ones"
                    );
                }
            }
            // On the worker the slot shows the view when the upload is in
            // the queue ([`crate::heap_worker::Batch::publish`]).
            let view = match &into {
                Into::List(_) => heap.handle(index).and_then(|handle| host.mirror(index, handle, true)),
                Into::Ahead(_) => record.as_ref().map(|(view, _)| view.clone()),
            };
            let placed = view.as_ref().and_then(crate::descriptor::host::placed_texture);
            let resource = match placed.as_ref().or(view.as_ref()) {
                Some(&View::Shader(resource, _)) => resource as This,
                _ => {
                    host.texture_slot_done(index, write, None);
                    continue;
                }
            };
            let title_fills = crate::resource::host_resource(resource).is_some_and(crate::texture::title_written);
            if !title_fills {
                // a render target, a depth or unordered access texture: the host's own
                host.texture_slot_done(index, write, None);
                continue;
            }
            // A texture that has a staging buffer now is one draws read
            // through a table: they upload it, and check it at every
            // submission that samples it.
            let tabled = crate::texture::is_staged(resource);
            if let Some(view) = &view {
                match &mut into {
                    Into::List(list) => readable(list, sampled.kind, view),
                    Into::Ahead(batch) => {
                        let Some(list) = batch.list() else {
                            return;
                        };
                        readable(list, sampled.kind, view);
                        if let (Some(handle), Some((_, key))) = (heap.handle(index), &record) {
                            batch.publish(index, handle, *key);
                        }
                    }
                }
            }
            // A texture view the host refuses removes the device: name it.
            static GONE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
            if !GONE.load(Ordering::Relaxed)
                && let Some(why) = crate::descriptor::host::device_removed()
            {
                GONE.store(true, Ordering::Relaxed);
                let text: String = format!("{view:?}").chars().take(300).collect();
                tracing::error!(why, slot = index, "the host device went while textures picked by index were made ready; last: {text}");
                crate::host::log_removal_context();
                return;
            }
            let (bytes, whole) = crate::texture::staged(resource).unwrap_or((0, true));
            // Not all of it there: looked at again next frame, and from
            // then on as a slot with nothing new (a placed texture: until
            // more of it is mapped; a texture object at an address the
            // title mapped: again in 120 frames, as before).
            let again = match partly {
                Some(mapped) => Some((frame + 1, mapped)),
                None => (!whole).then_some((frame + 120, 0)),
            };
            if again.is_some() {
                HEAP_TEXTURES_PARTLY[1].fetch_add(1, Ordering::Relaxed);
            }
            if only {
                let [uploaded, was, unmapped, as_before] = PASS_SOURCED.get();
                let staged = if tabled { 0 } else { bytes };
                PASS_SOURCED.set([uploaded + 1, was + staged, unmapped, as_before]);
                sourced_bytes += staged;
            }
            if tabled {
                // Nothing was staged here, so nothing is counted or spent,
                // and the staging buffer stays: taking it away made the
                // next draw that reads the texture through a table stage,
                // detile and upload all of it once more.
                HEAP_TEXTURES_TABLED.fetch_add(1, Ordering::Relaxed);
                host.texture_slot_done(index, write, again);
                continue;
            }
            // Uploaded: the list holds the staging buffer until it has run;
            // a table read, or the next look at a texture not all mapped
            // yet, makes it anew. (Kept for those until now: every partly
            // streamed texture of the heap held a staging buffer of its full
            // size for good.)
            let memory = if again.is_none() && heap_recheck() {
                crate::texture::staged_memory(resource)
            } else {
                None
            };
            crate::texture::unstage(resource);
            SPENT.fetch_add(bytes, Ordering::Relaxed);
            if let Into::Ahead(batch) = &mut into {
                batch.uploaded(bytes);
            }
            since_asked += bytes;
            if room && since_asked >= 32 << 20 {
                since_asked = 0;
                room = forced || crate::resource::room();
            }
            HEAP_TEXTURE_BYTES.fetch_add(bytes, Ordering::Relaxed);
            HEAP_TEXTURES.fetch_add(1, Ordering::Relaxed);
            match memory {
                // Whole and uploaded: looked at again (see above).
                Some((at, len)) => {
                    let print = crate::texture::fingerprint(at, len);
                    if again_in != RECHECK_FRAMES[0] {
                        HEAP_TEXTURES_SOON.fetch_add(1, Ordering::Relaxed);
                    }
                    host.texture_slot_recheck(index, write, frame + again_in, (at, len), print, 0, frame);
                }
                None => host.texture_slot_done(index, write, again),
            }
        }
    }

    /// For the frame listing, after a dropped dispatch's line: its group
    /// counts and the compute root arguments it was recorded with (root
    /// constants as hex dwords, root views with the title buffer they
    /// point into). A program that does not translate has no bindings to
    /// list; these say which memory it would have read and written (the
    /// ray-tracing structure builder passes its buffers this way).
    #[cfg(feature = "diagnostics")]
    fn list_dropped_roots(&self, groups: [u32; 3]) {
        crate::capture::op(0, || {
            use std::fmt::Write;
            let [x, y, z] = groups;
            let mut text = format!("  its roots ({x}x{y}x{z}):");
            for (slot, root) in self.compute.iter().enumerate() {
                match root {
                    Root::None => {}
                    Root::Table(a) => {
                        let _ = write!(text, " {slot}=table {a:#x}");
                    }
                    Root::Constants(v) => {
                        let _ = write!(text, " {slot}=constants {:x?}", &v[..v.len().min(48)]);
                    }
                    Root::View(a) => {
                        let _ = write!(text, " {slot}=view {a:#x}");
                        match crate::shadow::extent(*a) {
                            Some((start, len)) => {
                                let _ = write!(text, " in buffer {start:#x}+{len:#x}");
                            }
                            None => text.push_str(" in no buffer"),
                        }
                    }
                }
            }
            text
        });
    }

    /// Whether the shaders of the pipeline bound now can store to buffers
    /// (or the pipeline is not known): a draw with them may write the
    /// arguments of draws after it, which ends a list's gathering
    /// (`crate::indirect::gather`). The same test keeps such a draw out of
    /// a window ([`windows`]).
    fn may_store(&self) -> bool {
        crate::pipeline::host_pipeline(self.pipeline as This).is_none_or(|p| p.stages.iter().any(|stage| stage.translated().stores))
    }

    /// What is about to be recorded may write arguments (`stores`: a
    /// dispatch, or shaders that can store to buffers): the list's
    /// gathering ends, whole, or with `crate::indirect::gather_by_copy`
    /// only for the copies the shaders are declared to write as they are
    /// prepared (`HostList::gather_wrote`), and whole only for a pipeline
    /// the layer does not know.
    fn gather_ends(&self, list: &HostList, stores: bool) {
        if !stores {
            return;
        }
        // The open segment of gathered records ends whatever the prelude's gathering does
        // (`crate::indirect::gather_segments`).
        list.segments_end();
        // Whole, as before: without the switch, for a pipeline the layer
        // does not know, and for shaders that can store through a view
        // that is NOT declared where they are prepared: a buffer picked
        // from the view heap by an index the shader computes
        // (`State::heap_buffers`: what such a shader writes through a
        // picked unordered access view is declared nowhere), and title
        // memory by address (`crate::memory::ready` declares the numbered
        // copies once per list; a copy another list numbers after that is
        // reached through the directory and not declared in this list).
        let whole = !crate::indirect::gather_by_copy()
            || crate::pipeline::host_pipeline(self.pipeline as This).is_none_or(|p| {
                p.stages.iter().any(|stage| {
                    let t = stage.translated();
                    t.stores
                        && (t.host.root_srvs.contains(&victory_shader::emit::SLOT_STRIDES)
                            || t.host.root_srvs.contains(&victory_shader::emit::SLOT_MEMORY))
                })
            });
        if whole {
            list.gather_close();
        }
    }

    /// Runs a dispatch on the host.
    pub(crate) fn dispatch(&mut self, list: &HostList, x: u32, y: u32, z: u32) {
        self.dispatch_x(list, x, y, z, 0);
    }

    /// Runs a console `DispatchX(x, y, z, flags)` on the host: with
    /// `USE_THREAD_DIMENSIONS` (flags bit 5) the counts are threads, made
    /// groups here, and the program leaves out the threads past them
    /// ([`dispatch_dimensions`]; fork pull request #646).
    pub(crate) fn dispatch_x(&mut self, list: &HostList, x: u32, y: u32, z: u32, flags: u32) {
        // It may write the arguments a window holds clamped.
        self.window = None;
        crate::indirect::end_window(list);
        self.gather_ends(list, true);
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            crate::pipeline::note_drop(self.pipeline as This);
            return;
        }
        // A program that walks ray-tracing structures nothing has built.
        if crate::pipeline::walk_held(self.pipeline as This) {
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_why("a walk of ray-tracing structures that are not built (VICTORY_BVH_HOLD=1)".to_owned());
            return;
        }
        // The ray-tracing structure builder's stages run all or none.
        if crate::pipeline::is_builder(self.pipeline as This) && !crate::pipeline::builder_runs() {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            {
                crate::pipeline::note_why("a stage of the ray-tracing structure builder, which is off".to_owned());
                self.list_dropped_roots([x, y, z]);
            }
            return;
        }
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            once!(
                pipeline = format_args!("{:#x}", self.pipeline),
                "a dispatch without a host pipeline dropped"
            );
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            {
                crate::pipeline::note_drop(self.pipeline as This);
                self.list_dropped_roots([x, y, z]);
            }
            return;
        };
        if !p.compute {
            once!("a dispatch with a graphics pipeline");
            return;
        }
        // Counts of threads: the groups that hold them, and the counts the
        // program needs where the last group is cut (fork pull request #646:
        // taken as groups, the structure builder's box stages ran up to 64
        // times the threads asked and wrote past their structures).
        let translated = p.stages[0].translated();
        let Some(([x, y, z], partial)) = dispatch_dimensions([x, y, z], translated.threads, flags) else {
            return;
        };
        if partial != [0; 3]
            && !translated
                .host
                .user_data
                .iter()
                .any(|u| u.root == RootType::Other(victory_shader::emit::USER_DISPATCH_THREADS))
        {
            once!(
                pipeline = %p.name,
                threads = ?partial,
                group = ?translated.threads,
                "a DispatchX counting threads cuts its last group, and its program does not take the count: \
                 the threads past it run"
            );
        }
        // More than 65,535 groups in x: spread across z for a program that
        // takes the spread words (`USER_DISPATCH_SPREAD`).
        let (row, asked, gx, gy, gz) = victory_shader::emit::spread_groups(x, y, z, spreads(p).is_some());
        if row != 0 {
            SPREAD[0].fetch_add(1, Ordering::Relaxed);
        } else if gx > 65_535 || gy > 65_535 || gz > 65_535 {
            once!(gx, gy, gz, "a dispatch asks for more than 65,535 groups in a dimension and is not spread");
        }
        self.dispatch_spread = [row, asked];
        self.dispatch_threads = partial;
        let bound = self.bind(list, p, true);
        self.dispatch_spread = [0, 0];
        self.dispatch_threads = [0; 3];
        if !bound {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            // A stage of the structure builder that cannot be bound is a
            // dropped stage: the others stop too.
            if crate::pipeline::is_builder(self.pipeline as This) {
                crate::pipeline::builder_off("one of its dispatches could not be bound");
            }
            return;
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| self.op_text(format!("dispatch {x}x{y}x{z}"), true));
        // SAFETY: an open list with everything set.
        unsafe { list.list.Dispatch(gx, gy, gz) };
        DISPATCHED.fetch_add(1, Ordering::Relaxed);
    }

    /// What a draw or dispatch runs, for the device-removed report
    /// (diagnostics): the pipeline, then per stage the user data it gets
    /// (with the view at that heap index, if one is), the host addresses
    /// of its root views and the views its tables name.
    #[cfg(feature = "diagnostics")]
    fn op_text(&self, kind: String, compute: bool) -> String {
        use std::fmt::Write;
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return kind;
        };
        let mut text = format!("{kind} {}", p.name);
        // What it draws into (a dispatch writes through its views below).
        if !compute {
            let described = |t: usize| {
                crate::resource::host_made(t as This).map(|r| {
                    // SAFETY: a live resource.
                    let d = unsafe { r.GetDesc() };
                    format!("{t:#x} {}x{} f{}", d.Width, d.Height, d.Format.0)
                })
            };
            let targets: Vec<String> = self.targets.iter().filter_map(|&t| described(t)).collect();
            let _ = write!(text, " targets {targets:?} depth {:?} state [{}]", described(self.depth), p.state);
            // The title's scissor and viewport at this draw (the first of
            // each): the title draws what it sees through a
            // portal (a window, from inside a room) under the portal's
            // rectangle, and the listing did not say which draws those are.
            if let Some(r) = self.scissors.first() {
                let _ = write!(text, " scissor [{} {} {} {}]", r.left, r.top, r.right, r.bottom);
            }
            if let Some(v) = self.viewports.first() {
                let _ = write!(
                    text,
                    " viewport [{} {} {} {} depth {}..{}]",
                    v.TopLeftX, v.TopLeftY, v.Width, v.Height, v.MinDepth, v.MaxDepth
                );
            }
        }
        let views = (self.heaps[0] != 0)
            .then(|| crate::descriptor::heap(self.heaps[0] as This).host())
            .flatten();
        for stage in &p.stages {
            let t = stage.translated();
            let _ = write!(text, " | {}:", stage.shader.hash);
            for u in &t.host.user_data {
                let v = match u.root {
                    RootType::Other(kind) => self.host_word(compute, kind, u.slot, u.dword, None),
                    _ => self.user_value(compute, u.slot, u.root, u.dword),
                };
                let _ = write!(text, " u{}={v:#x}", u.index);
                let index = if u.root == RootType::Table { v / 32 } else { v };
                if let Some(slot) = views.and_then(|h| h.describe_slot(index)) {
                    let _ = write!(text, " [heap {index}: {slot}]");
                }
            }
            for &(slot, _) in stage.root_cbvs.iter().chain(&stage.root_srvs).chain(&stage.root_uavs) {
                if let Some(Root::View(address)) = self.roots_ref(compute).get(usize::from(slot)) {
                    let host = crate::shadow::find(*address, 1)
                        .map(|s| (s.gpu(*address), s.left(*address)));
                    let _ = write!(text, " root{slot}={address:#x} host {host:#x?}");
                }
            }
        }
        let _ = write!(text, " tables {:?}", self.reads(compute));
        if !compute && !list_slots().is_empty() {
            text.push_str(&self.slots_text());
        }
        text
    }

    /// For the frame listing, after a draw whose shader `VICTORY_LIST_SLOTS`
    /// names: the heap slots it will pick textures from by index, and what
    /// the title and the host hold there (one `~slot` line each), with the
    /// words they were read from (`~raw` lines, to decode offline what is
    /// not decoded here).
    ///
    /// Where the slots are, for the title's character shaders
    /// (`PS_DeferredPedTexturedNC`, from their native code): in the small
    /// buffers and constants of the draw's own table (every word of those
    /// that names a texture view gets a line); in the material record, at
    /// the byte offset the draw's constants (table 1, dword 56) hold at
    /// 0x1c, of the buffer at table 0, dword 160; and in the record at the
    /// byte offset the second root constant holds of the buffer at table 0,
    /// dword 688. A word that is a small number by chance and names a
    /// texture gets a line too.
    #[cfg(feature = "diagnostics")]
    fn slots_text(&self) -> String {
        use std::fmt::Write;
        const MATERIALS: u32 = 160;
        const GLOBAL: u32 = 688;
        const DRAW_CONSTANTS: u32 = 56;
        let wanted = list_slots();
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return String::new();
        };
        let Some(stage) = p.stages.iter().find(|s| wanted.iter().any(|w| s.shader.hash.starts_with(w.as_str())))
        else {
            return String::new();
        };
        let Some(heap) = self
            .heaps
            .iter()
            .filter(|&&h| h != 0)
            .map(|&h| crate::descriptor::heap(h as This))
            .find(|h| !h.samplers())
        else {
            return String::new();
        };
        let t = stage.translated();
        let mut text = String::new();
        let entry = |slot: u8, dwords: u32| -> Option<View> {
            let Some(Root::Table(table)) = self.roots_ref(false).get(usize::from(slot)) else {
                return None;
            };
            crate::descriptor::view(table + u64::from(dwords) * 4)
        };
        let mut material = None;
        for u in t.used.iter().filter(|u| u.root == RootType::Table && u.kind != ResourceType::Sampler) {
            let label = format!("s{}+{}", u.slot, u.table_offset_dwords);
            let Some(view) = entry(u.slot, u.table_offset_dwords) else {
                let _ = write!(text, "\n    ~table {label} no view");
                continue;
            };
            let range = match &view {
                View::Constant(address, size) => Some((*address, u64::from(*size))),
                other => crate::descriptor::host::buffer_range(other),
            };
            match range {
                // A small buffer or constants: its words.
                Some((address, len)) if len <= 2048 => {
                    Self::slot_words(&mut text, heap, &label, address, len as usize);
                    if u.slot == 1 && u.table_offset_dwords == DRAW_CONSTANTS && len >= 0x20 {
                        let mut word = [0u8; 4];
                        if crate::host::read_into((address + 0x1c) as *const u8, &mut word) {
                            material = Some(u32::from_le_bytes(word));
                        }
                    }
                }
                Some(_) => {}
                // A texture read through the table: the title slot it sits in.
                None => {
                    let Some(Root::Table(table)) = self.roots_ref(false).get(usize::from(u.slot)) else {
                        continue;
                    };
                    match heap.index(table + u64::from(u.table_offset_dwords) * 4) {
                        Some(index) => text.push_str(&Self::slot_line(heap, &format!("table:{label}"), 0, index)),
                        None => {
                            let _ = write!(text, "\n    ~table {label} title=[{}]", view_state(Some(&view)));
                        }
                    }
                }
            }
        }
        let uses = |slot: u8, dwords: u32| {
            t.used.iter().any(|u| u.root == RootType::Table && u.slot == slot && u.table_offset_dwords == dwords)
        };
        if let Some(at) = material
            && uses(0, MATERIALS)
            && let Some((base, len)) = entry(0, MATERIALS).as_ref().and_then(crate::descriptor::host::buffer_range)
        {
            let at = u64::from(at);
            let _ = write!(text, "\n    ~material record at {at:#x} of the buffer at {base:#x} bytes {len}");
            if at + 64 <= len {
                Self::slot_words(&mut text, heap, "M", base + at, 64);
            }
            if at + 1936 + 512 <= len {
                Self::slot_words(&mut text, heap, "M+1936", base + at + 1936, 512);
            }
        }
        if uses(0, GLOBAL)
            && let Some(slot) = t.used.iter().find(|u| u.root == RootType::Constants).map(|u| u.slot)
            && let Some(Root::Constants(words)) = self.roots_ref(false).get(usize::from(slot))
            && let Some(&at) = words.get(1)
            && let Some((base, len)) = entry(0, GLOBAL).as_ref().and_then(crate::descriptor::host::buffer_range)
        {
            let at = u64::from(at);
            let _ = write!(text, "\n    ~root constants {words:08x?}: record at {at:#x} of the buffer at {base:#x} bytes {len}");
            if at + 64 <= len {
                Self::slot_words(&mut text, heap, "G", base + at, 64);
            }
        }
        text
    }

    /// Up to 512 bytes of title memory at `address` as hex words (a `~raw`
    /// line), and a `~slot` line for every word of them that names a
    /// texture view in `heap`.
    #[cfg(feature = "diagnostics")]
    fn slot_words(text: &mut String, heap: &crate::descriptor::Heap, label: &str, address: u64, len: usize) {
        use std::fmt::Write;
        let mut bytes = vec![0u8; len.min(512) & !3];
        if bytes.is_empty() || !crate::host::read_into(address as *const u8, &mut bytes) {
            let _ = write!(text, "\n    ~raw {label} at {address:#x} bytes {len}: unreadable");
            return;
        }
        let words: Vec<u32> =
            bytes.chunks_exact(4).map(|w| u32::from_le_bytes([w[0], w[1], w[2], w[3]])).collect();
        let _ = write!(text, "\n    ~raw {label} at {address:#x} bytes {len}:");
        for w in &words {
            let _ = write!(text, " {w:08x}");
        }
        for (i, &index) in words.iter().enumerate() {
            let names = index != 0
                && heap
                    .handle(index)
                    .and_then(crate::descriptor::view)
                    .as_ref()
                    .is_some_and(crate::descriptor::host::is_texture_view);
            if names {
                text.push_str(&Self::slot_line(heap, label, i * 4, index));
            }
        }
    }

    /// One `~slot` line: where the slot number was read (`from`, byte
    /// `at`), what the title's heap holds in slot `index` with the state
    /// of that texture ([`view_state`]), what the host slot was made from,
    /// and what the queue of textures picked by index knows of it.
    #[cfg(feature = "diagnostics")]
    fn slot_line(heap: &crate::descriptor::Heap, from: &str, at: usize, index: u32) -> String {
        let title = heap.handle(index).and_then(crate::descriptor::view);
        let host: String = heap
            .host()
            .and_then(|h| h.describe_slot(index))
            .unwrap_or_else(|| "none".into())
            .chars()
            .take(90)
            .collect();
        let queue = match heap.host() {
            Some(h) => h.texture_slot_state(index, crate::frame::current()),
            None => "no host heap".into(),
        };
        format!(
            "\n    ~slot from={from}+{at:#x} slot={index} title=[{}] host=[{host}] queue=[{queue}]",
            view_state(title.as_ref())
        )
    }

    /// A mesh draw under `VICTORY_MESH_TEST=scissor` rasterises one pixel:
    /// the scissors before it (`on`), the title's after it.
    fn mesh_scissor(&self, list: &HostList, on: bool) {
        if mesh_test() != 2 {
            return;
        }
        if on {
            let one = RECT { left: 0, top: 0, right: 1, bottom: 1 };
            crate::draw::scissors(list, &vec![one; self.scissors.len().max(1)]);
        } else if !self.scissors.is_empty() {
            crate::draw::scissors(list, &self.scissors);
        }
    }

    /// Runs a mesh dispatch of `x` groups on the host (the title's mesh
    /// programs are one-dimensional).
    pub(crate) fn dispatch_mesh(&mut self, list: &HostList, x: u32) {
        self.window = None;
        crate::indirect::end_window(list);
        self.gather_ends(list, self.may_store());
        #[cfg(feature = "diagnostics")]
        self.trace("mesh", x, 0);
        let launch = crate::pipeline::host_pipeline(self.pipeline as This).and_then(|p| p.launch.as_ref());
        let ran = mesh_test() != 1
            && mesh_kept(self.pipeline)
            && self.prepare_for(list, false, true)
            && match list.list.cast::<ID3D12GraphicsCommandList6>() {
                Ok(l6) => {
                    #[cfg(feature = "diagnostics")]
                    list.note_op(|| self.op_text(format!("mesh {x}"), false));
                    self.mesh_scissor(list, true);
                    // With the launch stage the count goes to it as a root
                    // constant (`crate::launch`).
                    match launch {
                        Some(launch) => {
                            crate::launch::record(list, launch.roots, crate::launch::Launch::Direct(x));
                        }
                        // SAFETY: an open list with everything set.
                        None => unsafe { l6.DispatchMesh(x, 1, 1) },
                    }
                    self.mesh_scissor(list, false);
                    DRAWN.fetch_add(1, Ordering::Relaxed);
                    true
                }
                Err(_) => false,
            };
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    /// The walls trace (`crate::host::walls_trace`): one log line per
    /// frame with the draws of its main pass (four or more colour targets)
    /// by scissor rectangle, and how often each of the title's culling
    /// programs was dispatched. The title clips what it draws
    /// through a window or door to that opening's rectangle while it holds
    /// the camera to be in a room (run 182, frame 1529: 1,386 draws under
    /// `531 0 1789 1337`); in a launch without the room's walls the same
    /// shot has no such rectangle (run 187, frame 1414). The line shows
    /// the frame in which that changes.
    fn walls_census(&self, p: &HostPipeline, compute: bool) {
        const CULL: [(&str, &str); 9] = [
            ("55679a0547649d0b", "tree"),
            ("89d08e561897d4e5", "tree_imposter"),
            ("9209777e8face9dd", "grass"),
            ("17af15c4015cb4b6", "prop"),
            ("fdd56d97a806bcca", "prop_nocull"),
            ("42f717f1b4d18e94", "simple"),
            ("1c4b3c0547e3e6fa", "slod"),
            ("9146291b38ea9ce1", "meshlet_args"),
            ("42683eebec081520", "depth_pyramid_cpu_occlusion"),
        ];
        type Census = (u64, std::collections::BTreeMap<[i32; 4], u32>, [u32; 9]);
        static CENSUS: std::sync::Mutex<Census> =
            std::sync::Mutex::new((u64::MAX, std::collections::BTreeMap::new(), [0; 9]));
        let frame = crate::frame::current();
        let mut census = CENSUS.lock().unwrap_or_else(|e| e.into_inner());
        if census.0 != frame {
            let draws: u32 = census.1.values().sum();
            if draws > 0 {
                let mut rects: Vec<([i32; 4], u32)> = census.1.iter().map(|(r, n)| (*r, *n)).collect();
                rects.sort_unstable_by_key(|r| std::cmp::Reverse(r.1));
                let scissors: Vec<String> = rects
                    .iter()
                    .take(5)
                    .map(|(r, n)| format!("{} {} {} {} x{n}", r[0], r[1], r[2], r[3]))
                    .collect();
                let counts = census.2;
                let culling: Vec<String> = CULL
                    .iter()
                    .zip(counts)
                    .filter(|(_, n)| *n > 0)
                    .map(|(c, n)| format!("{}={n}", c.1))
                    .collect();
                tracing::info!(
                    frame = census.0,
                    main_pass_draws = draws,
                    rectangles = rects.len(),
                    scissors = ?scissors,
                    culling = ?culling,
                    "walls trace: the frame's main pass by scissor rectangle, and its culling dispatches"
                );
            }
            census.0 = frame;
            census.1.clear();
            census.2 = [0; 9];
        }
        if compute {
            for (i, c) in CULL.iter().enumerate() {
                if p.stages.iter().any(|s| s.shader.hash.as_str() == c.0) {
                    census.2[i] += 1;
                }
            }
        } else if self.targets.iter().filter(|&&t| t != 0).count() >= 4 {
            let r = self.scissors.first().map_or([0; 4], |r| [r.left, r.top, r.right, r.bottom]);
            *census.1.entry(r).or_insert(0) += 1;
        }
    }

    fn roots_ref(&self, compute: bool) -> &Vec<Root> {
        if compute {
            &self.compute
        } else {
            &self.graphics
        }
    }

    fn table_entry(&self, list: &HostList, u: &UsedResource, compute: bool) {
        let Some(Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
            return;
        };
        let handle = table + u64::from(u.table_offset_dwords) * 4;
        let Some(heap) = self.heap_of(handle) else {
            return;
        };
        let (Some(host), Some(index)) = (heap.host(), heap.index(handle)) else {
            return;
        };
        #[cfg(feature = "diagnostics")]
        tally(19, 1);
        if let Some(view) = host.mirror(index, handle, true) {
            readable(list, u.kind, &view);
            // Up to date by this draw: not the texture pass's to make.
            if crate::descriptor::host::is_texture_view(&view) {
                host.texture_slot_tabled(index, true);
            }
        }
    }

    /// The descriptors a shader reads through tables. Tables holding a
    /// constant buffer are made afresh for the draw in dynamic slots, with
    /// the constant data copied to upload memory when the list is
    /// submitted: the host wants constant buffers 256-byte aligned, the
    /// console only 16. Returns those tables' new byte offsets by root
    /// parameter; the other tables' slots are made from the title's.
    ///
    /// Within a list, the same table holding the same descriptors is made
    /// once: its constant data is copied at submission either way, and the
    /// big tables the world binds would otherwise take hundreds of slots per
    /// draw, wrapping the dynamic slots onto lists not yet submitted.
    ///
    /// `half`: the table descriptors the shader moves sixteen bits a
    /// channel as unsigned integers (`HostShader::half_views`). A table
    /// with a 16-bit float or signed buffer view at one of them is made
    /// afresh as well, with that view in the unsigned format of the same
    /// layout (`descriptor::host::half_view`): the title's own slot keeps
    /// the title's format for the shaders that read it as floats.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn tables(
        &mut self,
        list: &HostList,
        used: &[UsedResource],
        indexed: &[RootIndexedView],
        half: &[(u8, u32)],
        compute: bool,
        spans: &[(u8, u32, u32)],
        staged: &mut Vec<(u64, u32)>,
    ) -> Vec<(u8, u32)> {
        let tables = |u: &&UsedResource| u.root == RootType::Table;
        let mut fresh: Vec<u8> = used
            .iter()
            .filter(tables)
            .filter(|u| u.kind == ResourceType::Cbv)
            .map(|u| u.slot)
            .collect();
        // The view shown in place of the title's at dword `at` of the table
        // in root parameter `slot`, if the shader moves halves there and
        // the title's view is a 16-bit float or signed buffer view.
        let halved = |state: &Self, slot: u8, at: u32| -> Option<View> {
            if !half.contains(&(slot, at)) {
                return None;
            }
            let Some(&Root::Table(table)) = state.roots_ref(compute).get(usize::from(slot)) else {
                return None;
            };
            crate::descriptor::host::half_view(&crate::descriptor::view(table + u64::from(at) * 4)?)
        };
        for &(slot, at) in half {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let Some(view) = crate::descriptor::view(table + u64::from(at) * 4) else {
                continue;
            };
            if crate::descriptor::host::half_view(&view).is_some() {
                if !fresh.contains(&slot) {
                    fresh.push(slot);
                }
            } else if let Some(format) = crate::descriptor::host::half_inexact(&view) {
                once!(
                    format,
                    "a shader moves a buffer's 16-bit channels as integers through a view whose values the console converts: they are read and written unconverted"
                );
            }
        }
        // Tables reaching past a host heap smaller than the title's (the
        // host's limit, 1,000,000 views on NVIDIA) are made afresh too: the
        // title's slots there have no host slot of their own.
        for u in used.iter().filter(tables) {
            if fresh.contains(&u.slot) {
                continue;
            }
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(u.slot)) else {
                continue;
            };
            let end = table + u64::from(u.table_offset_dwords) * 4;
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            let (Some(host), Some(index)) = (heap.host(), heap.index(end)) else {
                continue;
            };
            if index >= host.count() {
                fresh.push(u.slot);
            }
        }
        fresh.sort_unstable();
        fresh.dedup();
        for u in used.iter().filter(tables) {
            if !fresh.contains(&u.slot) {
                self.table_entry(list, u, compute);
            }
        }
        let mut made = Vec::new();
        for slot in fresh {
            let Some(&Root::Table(table)) = self.roots_ref(compute).get(usize::from(slot)) else {
                continue;
            };
            let entries: Vec<&UsedResource> = used
                .iter()
                .filter(tables)
                .filter(|u| u.slot == slot)
                .collect();
            let Some(heap) = self.heap_of(table) else {
                continue;
            };
            if heap.samplers() || (table - heap.start) % 32 != 0 {
                continue;
            }
            let Some(host) = heap.host() else {
                continue;
            };
            let mut selected: Vec<u32> = indexed
                .iter()
                .filter(|index| index.table_slot == slot)
                .map(|index| {
                    let constant = self.user_value(
                        compute,
                        index.constant_slot,
                        RootType::Constants,
                        index.constant_dword,
                    );
                    // The shader adds i32 byte offsets and shifts by 5.
                    index.byte_offset.wrapping_add(constant << index.shift) / 32
                })
                .collect();
            // The descriptors the shader moves halves through are made too,
            // and the table reaches them: they may lie past the range starts
            // the entries give (a range of several views names only its
            // first; review on #927).
            selected.extend(
                half.iter()
                    .filter(|&&(s, _)| s == slot)
                    .map(|&(_, at)| at / 8),
            );
            selected.sort_unstable();
            selected.dedup();
            let n = entries
                .iter()
                .map(|u| u.table_offset_dwords / 8)
                .chain(selected.last().copied())
                .max()
                .unwrap_or(0)
                + 1;
            // With `table_stages`: as many slots as the stage of this draw
            // that reads furthest into the table needs among those that
            // make it afresh too: the ones reading a constant view in it,
            // or every stage where the table reaches past the host heap's
            // slots (`spans` is empty with the switch off).
            let n = match spans.iter().find(|s| s.0 == slot) {
                Some(&(_, with_constants, all)) => {
                    let past = heap
                        .index(table)
                        .is_some_and(|at| u64::from(at) + u64::from(all) > u64::from(host.count()));
                    n.max(if past { all } else { with_constants })
                }
                None => n,
            };
            let last = table.checked_add(u64::from(n - 1) * 32);
            if n > crate::descriptor::host::DYNAMIC
                || last.and_then(|handle| heap.index(handle)).is_none()
            {
                once!(slot, n, "fresh descriptor table exceeds the available span");
                continue;
            }
            // The table and what its entries hold (record words).
            let key = {
                use std::hash::{Hash, Hasher};
                let mut h = std::collections::hash_map::DefaultHasher::new();
                for k in 0..n {
                    let at = table + u64::from(k) * 32;
                    // SAFETY: a descriptor slot in title memory.
                    unsafe { (at as *const [u32; 3]).read_unaligned() }.hash(&mut h);
                    // (Another table than the same slots made for a shader
                    // that reads them in the title's formats.)
                    half.contains(&(slot, k * 8)).hash(&mut h);
                }
                (table, n, h.finish())
            };
            #[cfg(feature = "diagnostics")]
            tally(0, 1);
            if let Some(&first) = self.fresh_tables.get(&key) {
                #[cfg(feature = "diagnostics")]
                tally(1, 1);
                // Found in the table an earlier stage of this draw made
                // ([`table_stages`]): the title's own slots of the texture
                // views this stage reads are stamped and made as the
                // making of a table of its own did it (below).
                let staged_title = staged
                    .contains(&(table, n))
                    .then(|| heap.index(table).filter(|at| at + n <= host.count()))
                    .flatten();
                for u in entries.iter().filter(|u| u.kind != ResourceType::Cbv) {
                    let handle = table + u64::from(u.table_offset_dwords) * 4;
                    if let Some(view) = crate::descriptor::view(handle) {
                        readable(list, u.kind, &view);
                        if let Some(title) = staged_title
                            && crate::descriptor::host::is_texture_view(&view)
                        {
                            let k = u.table_offset_dwords / 8;
                            host.texture_slot_tabled(title + k, true);
                            host.mirror(title + k, handle, true);
                        }
                        #[cfg(feature = "diagnostics")]
                        tally(18, 1);
                    }
                }
                for k in selected {
                    if entries.iter().any(|u| u.table_offset_dwords / 8 == k) {
                        continue;
                    }
                    let handle = table + u64::from(k) * 32;
                    if let Some(view) = crate::descriptor::view(handle)
                        && let Some(kind) = view_kind(&view)
                    {
                        readable(list, kind, &view);
                        #[cfg(feature = "diagnostics")]
                        tally(18, 1);
                    }
                }
                made.push((slot, first * 32));
                continue;
            }
            #[cfg(feature = "diagnostics")]
            let mut clock = table_asked(&mut self.recording, table, n);
            let Some(first) = host.dynamic(n) else {
                // No room: the draw must not run with the title's own
                // offsets (slots never made for it); `bind` drops it.
                made.push((slot, u32::MAX));
                #[cfg(feature = "diagnostics")]
                tally(25, 1);
                continue;
            };
            #[cfg(feature = "diagnostics")]
            clock.lap(21);
            self.fresh_tables.insert(key, first);
            if !spans.is_empty() {
                staged.push((table, n));
            }
            self.dynamic
                .push((std::ptr::from_ref(heap) as usize, host.id(), first, n));
            // The title's own slots of this table: what the draw reads of
            // them it brings up to date here, in the dynamic slots.
            let title_first = heap.index(table).filter(|first| first + n <= host.count());
            // The whole range: a shader may index past the entries its
            // used-resource list names (descriptor arrays).
            for k in 0..n {
                let index = first + k;
                let used = entries.iter().find(|u| u.table_offset_dwords / 8 == k);
                match crate::descriptor::view(table + u64::from(k) * 32) {
                    Some(View::Constant(address, size)) => {
                        let Some((gpu, to)) = list.scratch(u64::from(size)) else {
                            host.make_null(index);
                            continue;
                        };
                        list.defer_fill(to, address as *const u8, size as usize);
                        host.constant_at(index, gpu, size);
                        #[cfg(feature = "diagnostics")]
                        {
                            tally(13, 1);
                            clock.lap(22);
                        }
                    }
                    Some(view) => {
                        match halved(self, slot, k * 8) {
                            Some(shown) => host.make_at(index, &shown),
                            None => host.make_at(index, &view),
                        }
                        #[cfg(feature = "diagnostics")]
                        {
                            tally(if crate::descriptor::host::is_texture_view(&view) { 14 } else { 15 }, 1);
                            clock.lap(22);
                        }
                        let kind = used.map(|u| u.kind).or_else(|| {
                            selected.binary_search(&k).ok().and_then(|_| view_kind(&view))
                        });
                        if let Some(kind) = kind {
                            readable(list, kind, &view);
                            #[cfg(feature = "diagnostics")]
                            {
                                tally(17, 1);
                                clock.lap(23);
                            }
                        }
                        // The title's slot is a table's: out of the queue
                        // of textures picked by index if the draw read it
                        // (its texture is up to date now), and made in
                        // place as the texture pass would have made it,
                        // for a shader that picks it by index all the same.
                        if let Some(title) = title_first
                            && crate::descriptor::host::is_texture_view(&view)
                        {
                            host.texture_slot_tabled(title + k, kind.is_some());
                            if kind.is_some() {
                                host.mirror(title + k, table + u64::from(k) * 32, true);
                            }
                            #[cfg(feature = "diagnostics")]
                            clock.lap(24);
                        }
                    }
                    None => {
                        host.make_null(index);
                        #[cfg(feature = "diagnostics")]
                        {
                            tally(16, 1);
                            clock.lap(22);
                        }
                        if used.is_some() {
                            once!(slot, "a table entry without a view");
                        }
                    }
                }
            }
            made.push((slot, first * 32));
        }
        #[cfg(feature = "diagnostics")]
        tally_done();
        made
    }

    /// The first draws, at debug level: what they bind and read
    /// (diagnostics builds).
    #[cfg(feature = "diagnostics")]
    fn trace(&self, kind: &str, count: u32, first: u32) {
        static TRACED: AtomicU64 = AtomicU64::new(0);
        if !tracing::enabled!(tracing::Level::DEBUG) {
            return;
        }
        let listed = crate::capture::listed(crate::queue::presented() + 1);
        let logged = listed || TRACED.fetch_add(1, Ordering::Relaxed) < 64;
        let targets: Vec<String> = self
            .targets
            .iter()
            .filter_map(|&t| crate::resource::host_made(t as This))
            .map(|t| {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                format!("{}x{} f{}", d.Width, d.Height, d.Format.0)
            })
            .collect();
        let v = self.vertex[0];
        let line = format!(
            "{kind} {count} from {first} topology {} pipeline {:#x} targets {targets:?} vb {:#x}+{} stride {} reads {:?}",
            self.topology,
            self.pipeline,
            v.address,
            v.size,
            v.stride,
            self.reads(false)
        );
        if logged {
            tracing::debug!("draw {line}");
        }
        let mut frame = FRAME.lock().unwrap_or_else(|e| e.into_inner());
        if frame.len() < 256 {
            frame.push(line);
        }
    }

    /// What a draw's shaders read, one line per used resource (diagnostics
    /// builds).
    #[cfg(feature = "diagnostics")]
    fn reads(&self, compute: bool) -> Vec<String> {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return vec!["no host pipeline".into()];
        };
        let mut out = Vec::new();
        for stage in &p.stages {
            let t = stage.translated();
            for u in &t.used {
                let what = match (u.root, self.roots_ref(compute).get(usize::from(u.slot))) {
                    (RootType::Table, Some(Root::Table(table))) => {
                        let handle = table + u64::from(u.table_offset_dwords) * 4;
                        describe(crate::descriptor::view(handle).as_ref())
                    }
                    (root, _) => format!("{root:?}"),
                };
                out.push(format!(
                    "{} {:?} s{}+{}: {what}",
                    stage.shader.hash, u.kind, u.slot, u.table_offset_dwords
                ));
            }
        }
        out
    }

    pub(crate) fn draw(
        &mut self,
        list: &HostList,
        vertices: u32,
        instances: u32,
        first: u32,
        first_instance: u32,
    ) {
        self.window = None;
        crate::indirect::end_window(list);
        self.gather_ends(list, self.may_store());
        #[cfg(feature = "diagnostics")]
        self.trace("draw", vertices, first);
        let ready = self.prepare(list, false);
        // A pipeline with the rectangle geometry stage draws the title's
        // vertices as they are; any other gets its rectangles expanded.
        let rects = self.topology == RECTANGLES && !self.completes_rects();
        let ran = ready && (!rects || self.rectangles(list, vertices, first));
        if ran {
            // Rectangles were expanded into their own vertex buffers.
            let (count, from) = if rects {
                (vertices / 3 * 6, 0)
            } else {
                (vertices, first)
            };
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("draw {count}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawInstanced(count, instances, from, first_instance)
            };
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }

    /// Whether the bound pipeline completes rectangles itself, in the
    /// geometry stage rectangle pipelines get (`HostPipeline::rects`).
    fn completes_rects(&self) -> bool {
        crate::pipeline::host_pipeline(self.pipeline as This).is_some_and(|p| p.rects)
    }

    /// Binds vertex buffers holding the rectangles of vertices `first..first
    /// + count` as triangles, made when the list is submitted.
    fn rectangles(&self, list: &HostList, count: u32, first: u32) -> bool {
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            return false;
        };
        let rects = u64::from(count / 3);
        for slot in 0..32 {
            let elements: Vec<(u32, u32)> = p
                .inputs
                .iter()
                .filter(|e| e.slot == slot && !e.per_instance)
                .map(|e| (e.offset, e.format))
                .collect();
            if elements.is_empty() {
                continue;
            }
            let v = self.vertex[slot as usize];
            let stride = u64::from(v.stride);
            let from = v.address + u64::from(first) * stride;
            if v.address == 0
                || stride == 0
                || u64::from(first + count) * stride > u64::from(v.size)
                || crate::shadow::find(from, rects * 3 * stride).is_none()
            {
                once!(slot, "rectangle vertices outside their buffer");
                return false;
            }
            let Some((gpu, to)) = list.scratch(rects * 6 * stride) else {
                return false;
            };
            let (from, to, stride) = (from as usize, to as usize, stride as usize);
            let n = rects as usize;
            list.defer_host(move || {
                // The vertices are read in place, at submission: no change
                // of mapping under the read (see
                // `crate::watch::title_memory_reading`).
                let _mapped = crate::watch::title_memory_reading_range(from, n * 3 * stride);
                // SAFETY: the vertices lie in a title buffer (checked when
                // recorded) and the scratch memory is the list's.
                let (src, dst) = unsafe {
                    (
                        std::slice::from_raw_parts(from as *const u8, n * 3 * stride),
                        std::slice::from_raw_parts_mut(to as *mut u8, n * 6 * stride),
                    )
                };
                crate::primitive::expand_rects(src, dst, stride, &elements);
            });
            let view = D3D12_VERTEX_BUFFER_VIEW {
                BufferLocation: gpu,
                SizeInBytes: (rects * 6 * stride as u64) as u32,
                StrideInBytes: v.stride,
            };
            // SAFETY: an open list; the view names live scratch memory.
            unsafe { list.list.IASetVertexBuffers(slot, Some(&[view])) };
        }
        true
    }

    pub(crate) fn draw_indexed(
        &mut self,
        list: &HostList,
        indices: u32,
        instances: u32,
        first: u32,
        base: i32,
        first_instance: u32,
    ) {
        self.window = None;
        crate::indirect::end_window(list);
        self.gather_ends(list, self.may_store());
        #[cfg(feature = "diagnostics")]
        self.trace("indexed", indices, first);
        // Indexed rectangles are drawn by the rectangle geometry stage
        // alone (nothing expands them on the CPU).
        let lost = self.topology == RECTANGLES && !self.completes_rects();
        if lost {
            once!("indexed rectangles not drawn");
        }
        let ran = !lost && self.prepare(list, true);
        if ran {
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indexed {indices}x{instances}"), false));
            // SAFETY: an open list with everything set.
            unsafe {
                list.list
                    .DrawIndexedInstanced(indices, instances, first, base, first_instance)
            };
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        if !ran {
            DROPPED.fetch_add(1, Ordering::Relaxed);
            #[cfg(feature = "diagnostics")]
            dropped(self.pipeline);
        }
    }
}

impl State {
    /// Indirect execution of a one-argument signature of `kind` (PC
    /// numbering: 0 draw, 1 indexed draw, 2 dispatch, 10 mesh dispatch) with
    /// records `stride` bytes apart at title address `arguments`, and the
    /// record count at `count` if the title gives one: the GPU reads them
    /// from the buffer's host copy, where earlier dispatches may have
    /// written them. False if it cannot run on the host.
    pub(crate) fn execute_indirect(
        &mut self,
        list: &HostList,
        (kind, stride, max): (u32, u32, u32),
        arguments: u64,
        count: Option<u64>,
    ) -> bool {
        // Skip before the argument clamp can mutate the indirect buffer.
        #[cfg(feature = "diagnostics")]
        if crate::pipeline::is_skipped(self.pipeline as This) {
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        }
        if crate::pipeline::is_builder(self.pipeline as This) && !crate::pipeline::builder_runs() {
            return false;
        }
        // A program that walks ray-tracing structures nothing has built:
        // dropped before the argument clamp, like a skipped one.
        if crate::pipeline::walk_held(self.pipeline as This) {
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_why("a walk of ray-tracing structures that are not built (VICTORY_BVH_HOLD=1)".to_owned());
            return false;
        }
        const DRAW: u32 = 0;
        const INDEXED: u32 = 1;
        const DISPATCH: u32 = 2;
        const MESH: u32 = 10;
        #[cfg(feature = "diagnostics")]
        let why = |what: &str| crate::pipeline::note_why(format!("indirect kind {kind}: {what}"));
        #[cfg(not(feature = "diagnostics"))]
        let why = |_: &str| {};
        let Some(p) = crate::pipeline::host_pipeline(self.pipeline as This) else {
            #[cfg(feature = "diagnostics")]
            crate::pipeline::note_drop(self.pipeline as This);
            return false;
        };
        // Indirect rectangles (the title's tiled lighting, 256
        // indirect draws a frame, makes their corners from the vertex
        // number) are completed by the pipeline's geometry stage: the
        // arguments stay the title's, three vertices a rectangle. They
        // cannot be expanded on the CPU, which does not know their count.
        // shortcut: in a pipeline without that stage, a rectangle with no
        // per-vertex inputs is drawn as its first triangle (three vertices,
        // half of it), and one with vertex buffers is not drawn.
        let made_rects = !p.rects
            && kind == DRAW
            && self.topology == RECTANGLES
            && !p.inputs.iter().any(|e| !e.per_instance);
        if made_rects {
            once!("indirect rectangles without vertex inputs drawn as their first triangle");
        }
        let matches_kind = match kind {
            DRAW | INDEXED => {
                (self.topology != RECTANGLES || made_rects || p.rects) && !p.compute && !p.mesh
            }
            MESH => !p.compute && p.mesh,
            DISPATCH => p.compute,
            _ => false,
        };
        if !matches_kind {
            why(&format!(
                "pipeline mismatch (compute {}, mesh {}, topology {})",
                p.compute, p.mesh, self.topology
            ));
            return false;
        }
        let len = u64::from(stride) * u64::from(max.max(1));
        let Some(args) = crate::shadow::find_any(arguments, len) else {
            why("arguments outside title buffers");
            return false;
        };
        let counter = match count {
            Some(at) => match crate::shadow::find_any(at, 4) {
                Some(s) => Some((s, at)),
                None => {
                    why("count outside title buffers");
                    return false;
                }
            },
            None => None,
        };
        // A dispatch of a program that takes the spread words: the clamp
        // writes [row, asked x, x, y, z] records, run through a signature
        // that sets the two words, then dispatches.
        let spread = (kind == DISPATCH && p.compute)
            .then(|| spreads(p))
            .flatten()
            .and_then(|(param, index)| host_spread_signature(&p.root, param, index));
        let is_spread = spread.is_some();
        let Some(signature) = spread.or_else(|| {
            matches!(kind, DRAW | INDEXED | DISPATCH | MESH)
                .then(|| host_signature(kind, stride))
                .flatten()
        }) else {
            why("no host signature");
            return false;
        };
        if kind == MESH && (mesh_test() == 1 || !mesh_kept(self.pipeline)) {
            why("mesh draws are switched off (VICTORY_MESH_TEST=skip)");
            return false;
        }
        // One mesh record of a pipeline with the launch stage: the draw
        // reads it itself (`crate::launch`), no clamp of ours before it.
        if kind == MESH
            && max == 1
            && count.is_none()
            && let Some(launch) = &p.launch
        {
            let offset = args.offset(arguments);
            if !offset.is_multiple_of(4) {
                why("a mesh record not 4-byte aligned");
                return false;
            }
            self.window = None;
            args.refresh(list, arguments, len);
            if !self.prepare_for(list, false, true) {
                why("prepare failed");
                return false;
            }
            #[cfg(feature = "diagnostics")]
            list.note_op(|| self.op_text(format!("indirect kind {kind} max {max} at {arguments:#x} launched"), false));
            // SAFETY: a live buffer.
            let record = unsafe { args.buffer.GetGPUVirtualAddress() } + offset;
            list.keep(&args.buffer);
            self.mesh_scissor(list, true);
            let launched = crate::launch::record(list, launch.roots, crate::launch::Launch::Record(record));
            self.mesh_scissor(list, false);
            if !launched {
                why("no budget for the launch");
                return false;
            }
            DRAWN.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        // A window of records clamped in one pass (`windows`): for single
        // records without a count buffer, drawn by shaders that store to no
        // buffer. Anything else ends the window: it uses the scratch buffer
        // or may write the arguments.
        const WINDOW_USE: usize = usize::MAX - 0x77;
        let offset = args.offset(arguments);
        let windowed = windows()
            && kind != DISPATCH
            && max == 1
            && count.is_none()
            && !p.stages.iter().any(|stage| stage.translated().stores);
        // Another host list (a new one after a cut, or this one after its
        // reset) has another scratch buffer, or this one's with old bytes.
        let first_on_list = list.first_use(WINDOW_USE);
        if !windowed || first_on_list {
            self.window = None;
        }
        // An indirect dispatch's own record may take a gathered slot as a draw's does
        // (`crate::indirect::gather_dispatches`); the dispatch ends the gathering right after that, below.
        let joins = kind == DISPATCH
            && max == 1
            && count.is_none()
            && !is_spread
            && gathers()
            && crate::indirect::gather_dispatches();
        if !windowed && !joins {
            // A dispatch, or shaders that store to buffers: arguments may
            // be written from here on.
            self.gather_ends(list, true);
        }
        // Diagnostics: why this command gets a pass of ours, if it does.
        #[cfg(feature = "diagnostics")]
        let why_not = if !windows() {
            usize::MAX
        } else if kind == DISPATCH {
            1
        } else if max != 1 {
            2
        } else if count.is_some() {
            3
        } else if !windowed {
            4
        } else if first_on_list {
            5
        } else {
            match &self.window {
                None => 6,
                Some(w) if !(w.copy == args.id() && w.kind == kind && w.stride == stride) => 6,
                Some(w) if offset >= w.base && (offset - w.base) % u64::from(w.step) == 0 => 8,
                Some(_) => 7,
            }
        };
        let same = |w: &Window| w.copy == args.id() && w.kind == kind && w.stride == stride;
        let hit = self
            .window
            .as_ref()
            .filter(|w| same(w) && offset >= w.base)
            .filter(|w| (offset - w.base) % u64::from(w.step) == 0 && (offset - w.base) / u64::from(w.step) < u64::from(w.records))
            .map(|w| (w.scratch.clone(), offset - w.base, w.step));
        // A record that misses its window: a slot the list's prelude fills,
        // if the list still gathers ([`gathers`]).
        let gathered = if hit.is_none() && (windowed || joins) && gathers() {
            let slot = crate::indirect::gather(list, (&args.buffer, args.id()), offset, (kind, stride));
            #[cfg(feature = "diagnostics")]
            WINDOW_WHY[if slot.is_some() { 12 } else { 13 }].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            slot
        } else {
            None
        };
        if joins {
            // The dispatch's record is taken (or clamped alone below): what the dispatch writes, from here on.
            self.gather_ends(list, true);
        }
        let (bounded, at) = match (hit, gathered) {
            (Some((scratch, at, step)), _) => {
                // Diagnostics of windows: the record clamped again and
                // compared (`crate::indirect::window_check`).
                if crate::indirect::window_check()
                    && crate::indirect::check_hit(
                        list,
                        (&args.buffer, offset),
                        (kind, step),
                        &scratch,
                        at,
                    )
                {
                    self.host_pipeline = 0;
                    self.host_roots[1] = 0;
                }
                #[cfg(feature = "diagnostics")]
                WINDOW_WHY[0].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                if !self.window_read {
                    self.window_read = true;
                    if let Some(w) = self.window.as_ref().filter(|w| w.records > 1) {
                        self.walk_step = Some((w.copy, w.kind, w.stride, w.step));
                    }
                }
                (scratch, at)
            }
            (None, Some((scratch, at, recorded))) => {
                // A segment's pass recorded here bound our pipeline (`crate::indirect::gather_segments`).
                if recorded {
                    self.host_pipeline = 0;
                    self.host_roots[1] = 0;
                }
                // The record's bytes are up to date when the list is
                // submitted, as for a record clamped in the list.
                args.refresh(list, arguments, len);
                // Diagnostics of windows, for gathered records too: the
                // record clamped again here and compared.
                if crate::indirect::window_check() && crate::indirect::check_hit(list, (&args.buffer, offset), (kind, stride), &scratch, at) {
                    self.host_pipeline = 0;
                    self.host_roots[1] = 0;
                }
                (scratch, at)
            }
            (None, None) => {
                // A window opened at the remembered step that no draw read
                // from: the title does not walk here by that step.
                if self.window_jumped && !self.window_read {
                    self.walk_step = None;
                    #[cfg(feature = "diagnostics")]
                    WINDOW_WHY[11].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                }
                self.window_jumped = false;
                // A record a short step after the last window's last one
                // (the same step as that window's, if it had one): the
                // title walks the buffer, so the ones that step apart
                // after it are clamped with it.
                let (step, records) = match &self.window {
                    Some(w) if windowed && same(w) => {
                        let last = w.base + u64::from(w.records - 1) * u64::from(w.step);
                        let step = offset.saturating_sub(last);
                        let walks = step >= u64::from(stride)
                            && step <= 256
                            && step % 4 == 0
                            && (w.records == 1 || step == u64::from(w.step));
                        if walks {
                            let fit = (args.left(arguments) / step).min(WINDOW_MOST) as u32;
                            (step as u32, (if w.records == 1 { 16 } else { w.records * 2 }).min(fit).max(1))
                        } else {
                            (stride, max)
                        }
                    }
                    _ => (stride, max),
                };
                // A record that would be clamped alone, on a buffer whose
                // step is remembered: the window the next record would
                // open, opened here ([`jump_windows`]).
                let (step, records) = match (windowed && records == 1 && jump_windows())
                    .then(|| jump_window(self.walk_step, (args.id(), kind, stride), args.left(arguments)))
                    .flatten()
                {
                    Some(jumped) => {
                        self.window_jumped = true;
                        #[cfg(feature = "diagnostics")]
                        WINDOW_WHY[10].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        jumped
                    }
                    None => (step, records),
                };
                self.window_read = false;
                args.refresh(list, arguments, len.max(u64::from(step) * u64::from(records)));
                if let Some((s, at)) = &counter {
                    s.refresh(list, *at, 4);
                }
                // Bounded first: the clamp binds a compute pipeline of its
                // own and copies clamped active records into a scratch
                // buffer at offset 0.
                let count_arg = counter.as_ref().map(|(s, at)| (&s.buffer, s.offset(*at)));
                // More than the record asked for: a window clamped ahead,
                // whose unfilled slots must not spend the list's budget.
                // With `VICTORY_CLAMP_WINDOW`: the record clamped in the list's window of indirect draws (a draw with no
                // count buffer whose shaders cannot store), at `window_at` in the window's buffer.
                let mut window_at = None;
                let bounded = if windowed && records > 1 {
                    crate::indirect::bound_ahead(list, (&args.buffer, offset), (kind, step, records))
                } else if let Some((dst, at, opened)) = (crate::indirect::clamp_window()
                    && count_arg.is_none()
                    && !is_spread
                    && kind != DISPATCH
                    && !self.may_store())
                .then(|| crate::indirect::windowed(list, (&args.buffer, offset), (kind, step, records)))
                .flatten()
                {
                    window_at = Some((at, opened));
                    Some(dst)
                } else {
                    // (A spread dispatch is never windowed: `windowed` is
                    // false for dispatches.)
                    crate::indirect::bound(list, (&args.buffer, offset), count_arg, (kind, step, records), is_spread)
                };
                let Some(bounded) = bounded else {
                    self.window = None;
                    why("not bounded (no clamp, or arguments not 4-byte aligned)");
                    return false;
                };
                #[cfg(feature = "diagnostics")]
                if why_not < WINDOW_WHY.len() {
                    WINDOW_WHY[why_not].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    WINDOW_WHY[9].fetch_add(u64::from(records), std::sync::atomic::Ordering::Relaxed);
                    // One in 256 of the draws that miss their window, said:
                    // where the record lies against the window and against
                    // the record drawn before it.
                    if (6..=8).contains(&why_not) {
                        static N: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
                        if N.fetch_add(1, std::sync::atomic::Ordering::Relaxed).is_multiple_of(256) {
                            let (last_copy, last_offset) = self.last_indirect;
                            tracing::info!(
                                "a window miss (one in 256): cause {why_not}, kind {kind}, stride {stride}, buffer {} offset {offset:#x} ({} left); the command before it on the list: buffer {last_copy} offset \
                                 {last_offset:#x} (this one is {} bytes on); the window it missed: {}; clamped now: {records} records at step {step}",
                                args.id(),
                                args.left(arguments),
                                offset as i64 - last_offset as i64,
                                match &self.window {
                                    Some(w) => format!("buffer {} kind {} stride {} base {:#x} step {} records {} (this record is {} bytes from its base)", w.copy, w.kind, w.stride, w.base, w.step, w.records, offset as i64 - w.base as i64),
                                    None => "none".to_string(),
                                }
                            );
                        }
                    }
                }
                if let Some((at, opened)) = window_at {
                    // A draw that joined a window recorded no pass: the graphics state stands.
                    if opened {
                        self.host_pipeline = 0;
                        self.host_roots[1] = 0;
                    }
                    // The records sit at `at` of the window's buffer: no ahead-window state is kept over them.
                    self.window = None;
                    (bounded, at)
                } else {
                    self.host_pipeline = 0;
                    self.host_roots[1] = 0;
                    self.window = windowed.then(|| Window {
                        copy: args.id(),
                        kind,
                        stride,
                        step,
                        base: offset,
                        records,
                        scratch: bounded.clone(),
                    });
                    (bounded, 0)
                }
            }
        };
        #[cfg(feature = "diagnostics")]
        {
            self.last_indirect = (args.id(), offset);
        }
        let ran = match kind {
            DRAW | INDEXED => self.prepare(list, kind == INDEXED),
            MESH => self.prepare_for(list, false, true),
            DISPATCH => self.bind(list, p, true),
            _ => false,
        };
        if !ran {
            why("prepare failed");
            return false;
        }
        // A mesh call the launch stage does not take (a count buffer, more
        // than one record): the same pipeline without the stage.
        if kind == MESH
            && let Some(launch) = &p.launch
        {
            let Some(plain) = launch.plain() else {
                why("no host pipeline without the launch stage");
                return false;
            };
            // SAFETY: an open list and a live pipeline of the same root
            // signature.
            unsafe { list.list.SetPipelineState(plain) };
            self.host_pipeline = windows::core::Interface::as_raw(plain) as usize;
            list.keep(plain);
        }
        #[cfg(feature = "diagnostics")]
        list.note_op(|| {
            self.op_text(format!("indirect kind {kind} max {max} at {arguments:#x}"), kind == DISPATCH)
        });
        if kind == MESH {
            self.mesh_scissor(list, true);
        }
        // SAFETY: an open list with everything set; the scratch buffer and
        // host copies are kept alive by the list, and the signature is cached.
        unsafe {
            list.list.ExecuteIndirect(
                &signature,
                max,
                &bounded,
                at,
                counter.as_ref().map(|(s, _)| &s.buffer),
                counter.as_ref().map_or(0, |(s, at)| s.offset(*at)),
            )
        };
        if kind == MESH {
            self.mesh_scissor(list, false);
        }
        // The scratch buffer stays readable as arguments: the next draw of
        // the window reads it with no barrier, and the next clamp moves it
        // back (`crate::indirect`).
        if kind == DISPATCH {
            DISPATCHED.fetch_add(1, Ordering::Relaxed);
            if is_spread {
                SPREAD[1].fetch_add(1, Ordering::Relaxed);
            }
        } else {
            DRAWN.fetch_add(1, Ordering::Relaxed);
        }
        true
    }
}

/// The host command signature for one argument of `kind` with records
/// `stride` bytes apart (no root arguments, so no root signature), made
/// once.
/// Dispatches spread across z since the frame line last took them: direct,
/// indirect ([`victory_shader::emit::USER_DISPATCH_SPREAD`]).
pub(crate) static SPREAD: [AtomicU64; 2] = [const { AtomicU64::new(0) }; 2];

/// `COMPUTE_DISPATCH_INITIATOR` bit 5, `USE_THREAD_DIMENSIONS`: a
/// `DispatchX` whose `x`, `y`, `z` count threads, not groups; in the last
/// group of an axis the threads at or past the count are not in EXEC.
const DISPATCH_THREAD_COUNTS: u32 = 0x20;

/// A console dispatch `(x, y, z)` with `DispatchX` flags `flags`, as host
/// groups and the thread counts the program is to keep to: per axis the
/// count where it is not a whole number of `group`-thread groups, else 0
/// (`USER_DISPATCH_THREADS`). Without [`DISPATCH_THREAD_COUNTS`] the counts
/// are groups and go as they are. `None`: no thread along some axis, so
/// nothing to run. (Fork pull request #646, mynzjerko.)
fn dispatch_dimensions(dims: [u32; 3], group: [u32; 3], flags: u32) -> Option<([u32; 3], [u32; 3])> {
    if flags & DISPATCH_THREAD_COUNTS == 0 {
        return Some((dims, [0; 3]));
    }
    if dims.contains(&0) {
        return None;
    }
    let groups = std::array::from_fn(|c| dims[c].div_ceil(group[c].max(1)));
    let partial = std::array::from_fn(|c| {
        let g = group[c].max(1);
        if g > 1 && !dims[c].is_multiple_of(g) { dims[c] } else { 0 }
    });
    Some((groups, partial))
}

/// The user-data root parameter of compute pipeline `p` whose program takes
/// the spread words, and the index of the first in it (the second follows:
/// one root-constant argument sets both); `None` for any other pipeline.
fn spreads(p: &crate::pipeline::host::HostPipeline) -> Option<(u32, u32)> {
    if !spread_on() {
        return None;
    }
    let stage = p.stages.first()?;
    let (param, _) = stage.user_data?;
    let kind = RootType::Other(victory_shader::emit::USER_DISPATCH_SPREAD);
    let words = &stage.translated().host.user_data;
    let first = words.iter().find(|u| u.root == kind && u.dword == 0)?.index;
    words
        .iter()
        .any(|u| u.root == kind && u.dword == 1 && u.index == first + 1)
        .then_some((param, first))
}

/// Whether dispatches past 65,535 groups in x are spread across z
/// (`VICTORY_DISPATCH_SPREAD`: on by default; `0` off, as before f93: clamped to
/// 65,535; `alt`: twenty seconds with, twenty without, in turn, each change
/// said in the log), to measure what the spread costs and shows within one
/// run. Off, a translated program's words stay 0: it runs as dispatched.
fn spread_on() -> bool {
    /// 0 off, 1 on, 2 in turn.
    static MODE: std::sync::OnceLock<u8> = std::sync::OnceLock::new();
    // (Not `VICTORY_SPREAD`: `crate::spread` reads that one, for the title's
    // pinned threads; run 326 alternated both under it.)
    match *MODE.get_or_init(|| match std::env::var("VICTORY_DISPATCH_SPREAD").as_deref() {
        Ok("0") => 0,
        Ok("alt") => 2,
        _ => 1,
    }) {
        0 => false,
        1 => true,
        _ => {
            static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
            let on = (EPOCH.get_or_init(std::time::Instant::now).elapsed().as_secs() / 20) % 2 == 1;
            static LAST: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(2);
            if LAST.swap(u8::from(on), Ordering::Relaxed) != u8::from(on) {
                tracing::info!(
                    "dispatches past 65,535 groups are spread across z from now (VICTORY_DISPATCH_SPREAD=alt): {}",
                    if on { "ON" } else { "off, clamped to 65,535 as before" }
                );
            }
            on
        }
    }
}

/// The command signature of a spread dispatch: two root constants (row,
/// asked x) at `index` of root parameter `param`, then the dispatch; 20
/// bytes a record (the clamp's spread layout). Made once per root
/// signature, parameter and index. (After upstream VIctory pull request
/// #2312.)
fn host_spread_signature(root: &ID3D12RootSignature, param: u32, index: u32) -> Option<ID3D12CommandSignature> {
    type Key = (usize, u32, u32);
    static MADE: std::sync::Mutex<Vec<(Key, ID3D12CommandSignature)>> = std::sync::Mutex::new(Vec::new());
    let key = (root.as_raw() as usize, param, index);
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == key) {
        return Some(s.clone());
    }
    let args = [
        D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_CONSTANT,
            Anonymous: D3D12_INDIRECT_ARGUMENT_DESC_0 {
                Constant: D3D12_INDIRECT_ARGUMENT_DESC_0_1 {
                    RootParameterIndex: param,
                    DestOffsetIn32BitValues: index,
                    Num32BitValuesToSet: 2,
                },
            },
        },
        D3D12_INDIRECT_ARGUMENT_DESC {
            Type: D3D12_INDIRECT_ARGUMENT_TYPE_DISPATCH,
            ..Default::default()
        },
    ];
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: 20,
        NumArgumentDescs: args.len() as u32,
        pArgumentDescs: args.as_ptr(),
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: two valid argument descriptions and a live root signature.
    let result = unsafe { crate::host::host()?.device.CreateCommandSignature(&desc, Some(root), &mut signature) };
    if let Err(e) = result {
        once!(error = %e, param, index, "spread command signature refused");
        return None;
    }
    let signature = signature?;
    made.push((key, signature.clone()));
    Some(signature)
}

fn host_signature(kind: u32, stride: u32) -> Option<ID3D12CommandSignature> {
    static MADE: std::sync::Mutex<Vec<((u32, u32), ID3D12CommandSignature)>> =
        std::sync::Mutex::new(Vec::new());
    let mut made = MADE.lock().unwrap_or_else(|e| e.into_inner());
    if let Some((_, s)) = made.iter().find(|(k, _)| *k == (kind, stride)) {
        return Some(s.clone());
    }
    let argument = D3D12_INDIRECT_ARGUMENT_DESC {
        Type: D3D12_INDIRECT_ARGUMENT_TYPE(kind as i32),
        ..Default::default()
    };
    let desc = D3D12_COMMAND_SIGNATURE_DESC {
        ByteStride: stride,
        NumArgumentDescs: 1,
        pArgumentDescs: &argument,
        NodeMask: 0,
    };
    let mut signature: Option<ID3D12CommandSignature> = None;
    // SAFETY: a description of one argument; no root arguments.
    let result = unsafe {
        crate::host::host()?.device.CreateCommandSignature(
            &desc,
            None::<&ID3D12RootSignature>,
            &mut signature,
        )
    };
    if let Err(e) = result {
        once!(error = %e, kind, stride, "host command signature refused");
        return None;
    }
    let signature = signature?;
    made.push(((kind, stride), signature.clone()));
    Some(signature)
}

/// Gets what a shader reads or writes through a view ready: buffer data
/// copied, textures moved to the state they need.
fn view_kind(view: &View) -> Option<ResourceType> {
    match view {
        View::Shader(..) | View::Placed(false, ..) => Some(ResourceType::Srv),
        View::Unordered(..) | View::Placed(true, ..) => Some(ResourceType::Uav),
        _ => None,
    }
}

/// The format of a host view of `texture` whose description names
/// `described` (none without a description): that one, or the texture's
/// own, which is what the host gives a view without a format.
fn bound_format(texture: &ID3D12Resource, described: Option<DXGI_FORMAT>) -> u32 {
    match described {
        Some(format) if format.0 != 0 => format.0 as u32,
        // SAFETY: a live texture.
        _ => unsafe { texture.GetDesc() }.Format.0 as u32,
    }
}

/// The host state a depth view needs of its texture: the read state if the
/// view writes no plane the format has (the title's read-only flags), else
/// the write state.
/// shortcut: whole textures have one state, so a view that writes one of
/// two planes counts as writing.
fn depth_state(texture: &ID3D12Resource, desc: Option<&D3D12_DEPTH_STENCIL_VIEW_DESC>) -> u32 {
    // D16_UNORM and D32_FLOAT: depth formats without a stencil plane.
    const ONE_PLANE: [i32; 2] = [55, 40];
    let Some(desc) = desc else {
        return DEPTH_WRITE;
    };
    let format = if desc.Format.0 == 0 {
        // SAFETY: a live texture; a view without a format has its own.
        unsafe { texture.GetDesc() }.Format.0
    } else {
        desc.Format.0
    };
    let mut needed = D3D12_DSV_FLAG_READ_ONLY_DEPTH.0;
    if !ONE_PLANE.contains(&format) {
        needed |= D3D12_DSV_FLAG_READ_ONLY_STENCIL.0;
    }
    if desc.Flags.0 & needed == needed {
        DEPTH_READ
    } else {
        DEPTH_WRITE
    }
}

fn readable(list: &HostList, kind: ResourceType, view: &View) {
    const UNORDERED_ACCESS: u32 = 0x8;
    let placed = crate::descriptor::host::placed_texture(view);
    let view = placed.as_ref().unwrap_or(view);
    // Buffer views read title memory: their range is copied at submission;
    // what shaders write through unordered access goes back to it after.
    if let Some((shadow, address, len)) = crate::descriptor::host::buffer_copy(view) {
        shadow.refresh(list, address, len);
        if kind == ResourceType::Uav {
            // A shader that cannot store to buffers leaves the range as
            // the title has it: nothing to find afterwards.
            if STORES.get() || !write_skip() {
                shadow.gpu_writes(list, address, len);
            } else {
                #[cfg(feature = "diagnostics")]
                {
                    crate::copystat::add(crate::copystat::C::GpuWritesLeftOut, 1);
                    crate::copystat::add(crate::copystat::C::GpuWritesLeftOutBytes, len);
                }
            }
        }
    }
    match (view, kind) {
        (View::Shader(resource, _), ResourceType::Srv) => {
            // A view the title placed, in another format, in the memory of
            // a texture the host writes: that texture's texels, copied on
            // the host (title memory does not have them).
            let aliased = crate::resource::alias_of(*resource as This)
                .is_some_and(|source| crate::texture::alias(list, *resource as This, source));
            if !aliased
                && let Some(t) = crate::resource::host_resource(*resource as This)
                && crate::texture::title_written(t)
            {
                crate::texture::upload(list, *resource as This, t);
            }
            // A depth texture shaders also write: what they wrote, first.
            list.twin_use(*resource as This, false, false);
            // The depth target the draw only tests is read in both ways at
            // once: both read states, which its preparation asks for too.
            let (tested, _) = TESTED_DEPTH.get();
            let state = if tested != 0 && tested == *resource {
                TESTED_DEPTH.set((tested, true));
                DEPTH_READ | SHADER_RESOURCE
            } else {
                SHADER_RESOURCE
            };
            crate::list::ensure_state(list, *resource as This, state);
            #[cfg(feature = "diagnostics")]
            if let Some(t) = crate::resource::host_resource(*resource as This) {
                // SAFETY: a live resource.
                let d = unsafe { t.GetDesc() };
                if d.Flags.0 & 7 == 0 {
                    tracing::debug!(
                        resource = format_args!("{resource:#x}"),
                        ?d,
                        "a shader reads a texture the title fills"
                    );
                }
            }
        }
        (View::Unordered(resource, _, _), ResourceType::Uav) => {
            #[cfg(feature = "diagnostics")]
            crate::capture::note_target(*resource as This);
            // Some compute-written textures are also read by the title CPU.
            list.texture_written(*resource as This);
            // Unordered access to a depth texture is its colour twin's.
            if !list.twin_use(*resource as This, true, true) {
                crate::list::ensure_state(list, *resource as This, UNORDERED_ACCESS);
            }
        }
        _ => {}
    }
}

/// The pixel shaders whose draws get their slots listed
/// ([`State::slots_text`]): `VICTORY_LIST_SLOTS`, hash prefixes separated
/// by commas. None by default.
#[cfg(feature = "diagnostics")]
fn list_slots() -> &'static [String] {
    static LIST: std::sync::OnceLock<Vec<String>> = std::sync::OnceLock::new();
    LIST.get_or_init(|| {
        std::env::var("VICTORY_LIST_SLOTS")
            .unwrap_or_default()
            .split(',')
            .map(|s| s.trim().to_ascii_lowercase())
            .filter(|s| !s.is_empty())
            .collect()
    })
}

/// A fingerprint of the 4 KiB of title memory at `at` (FNV-1a, 64 bits; of
/// 4 KiB of zeros it is always the same value, which `slots.py` knows), or
/// "unreadable".
#[cfg(feature = "diagnostics")]
fn print_4k(at: u64) -> String {
    let mut bytes = [0u8; 4096];
    if at == 0 || !crate::host::read_into(at as *const u8, &mut bytes) {
        return "unreadable".into();
    }
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for b in bytes {
        hash = (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// The `bytes` of title memory at `at`: how many are mapped, and the
/// fingerprints of the first and the last 4 KiB (a texture's smallest
/// levels lie first, its largest last).
#[cfg(feature = "diagnostics")]
fn memory_state(at: u64, bytes: u64) -> String {
    let mapped = crate::texture::mapped_bytes(at as *const u8, bytes as usize);
    let last = at + bytes.saturating_sub(4096);
    format!("memory={at:#x} mapped={mapped}/{bytes} first4k={} last4k={}", print_4k(at), print_4k(last))
}

/// Title texture `this` for the frame listing: its size, whether it has a
/// host copy (not made here), its staging buffer's state, whether the host
/// copied into it, and its memory ([`memory_state`]).
#[cfg(feature = "diagnostics")]
fn texture_state(this: This) -> String {
    let Some((at, desc)) = crate::resource::texture_memory(this) else {
        return "not a texture".into();
    };
    // SAFETY: 56 bytes in the description's layout.
    let words = unsafe { std::ptr::from_ref(&desc).cast::<[u32; 14]>().read_unaligned() };
    let bytes = crate::resource::placed_bytes(&words).unwrap_or(0);
    let staged = match crate::texture::staged_state(this) {
        Some((uploaded, dirty)) => format!("staged=yes uploaded={uploaded} dirty={dirty}"),
        None => "staged=no".into(),
    };
    format!(
        "{}x{} f{} levels={} host_copy={} {staged} host_written={} title_fills={} {}",
        u64::from(words[4]) | u64::from(words[5]) << 32,
        words[6],
        words[8],
        words[7] >> 16,
        crate::resource::host_made(this).is_some(),
        crate::texture::host_wrote(this as usize),
        crate::resource::title_fills(this),
        memory_state(at as u64, bytes)
    )
}

/// A view of the title's heap for the frame listing: a texture with its
/// state ([`texture_state`]; a placed one that has no resource yet with
/// its memory alone), anything else in a few words.
#[cfg(feature = "diagnostics")]
fn view_state(view: Option<&View>) -> String {
    match view {
        None => "no view".into(),
        Some(View::Placed(false, r, d)) if r[0] != 1 => {
            let address = u64::from(d[10]) | u64::from(d[11]) << 32;
            let state = match crate::resource::texture_at_made(address, r) {
                Some(this) => texture_state(this),
                None => format!(
                    "{}x{} f{} levels={} resource=none {}",
                    u64::from(r[4]) | u64::from(r[5]) << 32,
                    r[6],
                    r[8],
                    r[7] >> 16,
                    memory_state(address, crate::resource::placed_bytes(r).unwrap_or(0))
                ),
            };
            // (Last, so that readers of the older lines still find
            // theirs: the level the view starts at, and how many it shows.)
            format!("placed texture view_format={} {state} view_levels={}+{}", d[0], d[4], d[5] as i32)
        }
        Some(View::Shader(r, d)) if crate::resource::texture_memory(*r as This).is_some() => {
            format!(
                "texture object {r:#x} view_format={} {} view_levels={}",
                d.as_ref().map_or(0, |d| d[0]),
                texture_state(*r as This),
                d.as_ref().map_or("all".to_owned(), |d| format!("{}+{}", d[4], d[5] as i32))
            )
        }
        Some(v) => format!("{v:?}").chars().take(80).collect(),
    }
}

/// A view in a few words (diagnostics builds).
#[cfg(feature = "diagnostics")]
fn describe(view: Option<&View>) -> String {
    let texture = |r: This| {
        crate::resource::host_made(r).map(|t| {
            // SAFETY: a live resource.
            let d = unsafe { t.GetDesc() };
            format!("{}x{} f{} flags {:#x} of {r:?}", d.Width, d.Height, d.Format.0, d.Flags.0)
        })
    };
    match view {
        None => "no view".into(),
        Some(v @ View::Shader(r, _)) => texture(*r as This)
            .unwrap_or_else(|| format!("buffer {:?}", crate::descriptor::host::buffer_range(v))),
        Some(v @ View::Unordered(r, ..)) => match texture(*r as This) {
            Some(t) => format!("uav texture {t}"),
            None => format!(
                "uav resource {r:#x} buffer {:?}",
                crate::descriptor::host::buffer_range(v)
            ),
        },
        Some(View::Placed(u, r, d)) => format!(
            "placed uav={u} dim {} {}x{} f{} layout {:#x} at {:#x}",
            r[0],
            u64::from(r[4]) | u64::from(r[5]) << 32,
            r[6],
            r[8],
            r[11],
            u64::from(d[10]) | u64::from(d[11]) << 32
        ),
        // The first four values, as a colour rectangle's constants are.
        Some(View::Constant(address, size))
            if *size >= 16 && crate::texture::committed(*address as *const u8, 16) =>
        {
            // SAFETY: 16 committed bytes of title memory.
            let v = unsafe { (*address as *const [f32; 4]).read_unaligned() };
            format!("Constant({address:#x}, {size}) {v:?}")
        }
        Some(v) => format!("{v:?}").chars().take(60).collect(),
    }
}

/// The console's own primitive topologies.
const TRIANGLE_FAN: u32 = 6;
const RECTANGLES: u32 = 17;
const LINE_LOOP: u32 = 18;
const QUADS: u32 = 19;
const QUAD_STRIP: u32 = 20;

/// Fixed-function state, straight onto the host list.
pub(crate) fn viewports(list: &HostList, viewports: &[D3D12_VIEWPORT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetViewports(viewports) };
}

pub(crate) fn scissors(list: &HostList, rects: &[RECT]) {
    // SAFETY: an open list.
    unsafe { list.list.RSSetScissorRects(rects) };
}

pub(crate) fn blend_factor(list: &HostList, factor: Option<&[f32; 4]>) {
    // SAFETY: an open list.
    unsafe { list.list.OMSetBlendFactor(factor) };
}

pub(crate) fn stencil_ref(list: &HostList, reference: u32) {
    #[cfg(feature = "diagnostics")]
    crate::capture::op(std::ptr::from_ref(list) as usize, || format!("STENCIL REF {reference:#x}"));
    // SAFETY: an open list.
    unsafe { list.list.OMSetStencilRef(reference) };
}

pub(crate) fn depth_bounds(list: &HostList, min: f32, max: f32) {
    #[cfg(feature = "diagnostics")]
    crate::capture::op(std::ptr::from_ref(list) as usize, || {
        format!("DEPTH BOUNDS {min}..{max}")
    });
    if crate::host::host().is_some_and(|h| h.depth_bounds())
        && let Ok(l1) = list.list.cast::<ID3D12GraphicsCommandList1>()
    {
        // SAFETY: an open list on a device that supports depth bounds.
        unsafe { l1.OMSetDepthBounds(min, max) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_lone_record_opens_a_window_at_the_remembered_step() {
        // (a mesh dispatch is kind 10, a draw kind 0)
        let remembered = Some((7u64, 10u32, 12u32, 32u32));
        // the same buffer, kind and stride: sixteen records at the title's step
        assert_eq!(jump_window(remembered, (7, 10, 12), 1 << 20), Some((32, 16)));
        // never past the buffer: 100 bytes left hold three records 32 apart
        assert_eq!(jump_window(remembered, (7, 10, 12), 100), Some((32, 3)));
        // one record fits: nothing to open
        assert_eq!(jump_window(remembered, (7, 10, 12), 40), None);
        // another buffer, kind or stride, or nothing remembered: alone, as before
        assert_eq!(jump_window(remembered, (8, 10, 12), 1 << 20), None);
        assert_eq!(jump_window(remembered, (7, 0, 12), 1 << 20), None);
        assert_eq!(jump_window(remembered, (7, 10, 16), 1 << 20), None);
        assert_eq!(jump_window(None, (7, 10, 12), 1 << 20), None);
    }

    /// A bake draw names the indices in its constants, and those in the
    /// records its constants give and in the one after them; not those of
    /// records past the buffer, nor of one that cannot be read.
    #[test]
    fn a_bake_draw_names_the_indices_of_its_constants_and_records() {
        let mut constants = [0u8; 0x80];
        let put = |bytes: &mut [u8], at: usize, value: u32| bytes[at..at + 4].copy_from_slice(&value.to_le_bytes());
        for (n, at) in BAKE_CONSTANT_INDICES.into_iter().enumerate() {
            put(&mut constants, at, 100 + n as u32);
        }
        // Records 2 and 3, of 9 in all.
        put(&mut constants, 0, 9);
        put(&mut constants, 0x78, 2);
        put(&mut constants, 0x7c, 4);
        // Record `i` names 1000 + 10 i + n at its n-th place; record 3 the
        // index of the constants' first once more.
        let record = |i: u32| {
            let mut record = [0xffu8; 0x44];
            for (n, at) in BAKE_RECORD_INDICES.into_iter().enumerate() {
                put(&mut record, at, 1000 + 10 * i + n as u32);
            }
            if i == 3 {
                put(&mut record, 0, 100);
            }
            Some(record)
        };
        let (named, records) = bake_named(&constants, 9, record);
        assert_eq!(records, (2, 3), "two records walked, and the one after them looked at");
        let mut expected: Vec<u32> = (100..105).collect();
        expected.extend((1020..1025).chain(1031..1035).chain(1040..1045));
        assert_eq!(named, expected);
        // The buffer ends with record 3: the one after is not there.
        assert_eq!(bake_named(&constants, 4, record).1, (2, 2));
        // A record that cannot be read ends the walk.
        let (named, records) = bake_named(&constants, 9, |i| if i == 3 { None } else { record(i) });
        assert_eq!(records, (2, 1));
        assert!(named.contains(&1024) && !named.contains(&1040));
        // No records at all: the constants' own.
        put(&mut constants, 0x7c, 2);
        assert_eq!(bake_named(&constants, 0, record), ((100..105).collect(), (0, 0)));
    }

    /// Fork pull request #646: a `DispatchX` with bit 5 counts threads.
    #[test]
    fn a_dispatch_counting_threads_runs_the_groups_that_hold_them() {
        // Without bit 5 (plain Dispatch, or only the 32-lane bit 0x8000): groups.
        assert_eq!(dispatch_dimensions([48, 1, 1], [64, 1, 1], 0), Some(([48, 1, 1], [0; 3])));
        assert_eq!(dispatch_dimensions([48, 1, 1], [64, 1, 1], 0x8000), Some(([48, 1, 1], [0; 3])));
        // Threads: the groups that hold them, and the count where the last is cut.
        assert_eq!(dispatch_dimensions([48, 1, 1], [64, 1, 1], 0x20), Some(([1, 1, 1], [48, 0, 0])));
        assert_eq!(dispatch_dimensions([1, 1, 1], [64, 1, 1], 0x8020), Some(([1, 1, 1], [1, 0, 0])));
        assert_eq!(dispatch_dimensions([65, 1, 1], [64, 1, 1], 0x20), Some(([2, 1, 1], [65, 0, 0])));
        // The builder's box stage: 5,348 nodes of one thread each in groups of 64.
        assert_eq!(dispatch_dimensions([5348, 1, 1], [64, 1, 1], 0x20), Some(([84, 1, 1], [5348, 0, 0])));
        // A whole number of groups needs no count.
        assert_eq!(dispatch_dimensions([128, 1, 1], [64, 1, 1], 0x20), Some(([2, 1, 1], [0; 3])));
        assert_eq!(dispatch_dimensions([10, 16, 5], [8, 8, 4], 0x20), Some(([2, 2, 2], [10, 0, 5])));
        // No thread along an axis: nothing runs.
        assert_eq!(dispatch_dimensions([0, 1, 1], [64, 1, 1], 0x20), None);
    }
}
