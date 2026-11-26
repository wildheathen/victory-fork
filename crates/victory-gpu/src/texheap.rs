//! Host textures placed in heaps of ours instead of each made with memory
//! of its own.
//!
//! While the world streams in, the title has thousands of textures made in
//! a frame, and each host copy was a committed resource: an allocation of
//! the card's memory in the graphics kernel for every one. In the loading
//! window of run prof-burst (an RTX 5080) making them took about 38 ms of
//! every frame on the recording threads, most of it in frames of 1.3 to
//! 3.4 s with up to 5,000 uploads each. A placed texture takes a piece of a
//! heap made ahead; making it asks the kernel for nothing. With it the
//! world's mean frame went from 548-618 to 352 ms and its frames over a
//! second from 8-10 to 1 in a reading (run perf-placed against
//! ab-slab-1/2).
//!
//! Only textures shaders read and nothing renders to or writes (the
//! streamed ones): render targets, depth and unordered access keep their
//! own memory, as their clears and compression want. Pieces are powers of
//! two from 64 KiB to [`PIECE_MOST`], cut from heaps of [`HEAP_BYTES`], and
//! a piece goes back to a list by its size when its texture goes (when the
//! last list holding the texture has been reset, so the card is done with
//! it). The heaps stay; past [`heaps_most`] in all, textures are made as
//! before.
//!
//! A piece used again holds what the texture before it left until the new
//! one is uploaded, where a committed texture would be zeros; a texture is
//! uploaded before the first draw that samples it (`crate::draw`'s early
//! textures), so nothing should show it.
//!
//! `VICTORY_PLACED_TEXTURES=0`: every host texture with memory of its own,
//! as before.

use std::collections::{BTreeMap, HashMap, VecDeque};
use std::time::{Duration, Instant};
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use windows::Win32::Graphics::Direct3D12::{
    D3D12_HEAP_DESC, D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES, D3D12_HEAP_PROPERTIES,
    D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_DESC, D3D12_RESOURCE_STATES, ID3D12Device, ID3D12Heap,
    ID3D12Resource,
};
use windows::core::Interface;

/// A heap's size.
const HEAP_BYTES: u64 = 256 << 20;
/// The most bytes of heaps made: a quarter of the card's memory budget,
/// at most 4 GiB (3.7 GiB on a 16 GB card, about 1.8 on an 8 GB one).
fn heaps_most() -> u64 {
    static MOST: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *MOST.get_or_init(|| {
        let budget = crate::host::host().and_then(|h| h.video_memory()).map_or(0, |m| m[0]);
        let most = if budget == 0 { 2 << 30 } else { (budget / 4).min(4 << 30) };
        tracing::info!(mib = most >> 20, "the most memory for placed textures");
        most
    })
}
/// The largest piece: bigger textures have memory of their own.
const PIECE_MOST: u64 = 16 << 20;
/// The smallest piece (a texture's usual alignment).
const PIECE_LEAST: u64 = 64 << 10;
/// How long a piece rests before it is used again. A texture goes when the
/// last `Arc` of its host copy does, but a list or a slot may still hold
/// the host resource itself (a COM reference) a little longer; no list is
/// in flight for seconds.
const REST: Duration = Duration::from_secs(5);

struct Heaps {
    heaps: Vec<ID3D12Heap>,
    /// The heap pieces are cut from, and its bytes cut so far.
    cutting: Option<(usize, u64)>,
    /// Pieces given back, by size, oldest first: heap, offset, when.
    spare: BTreeMap<u64, VecDeque<(usize, u64, Instant)>>,
    /// Pieces held, by their texture (its interface pointer): size, heap,
    /// offset.
    held: HashMap<usize, (u64, usize, u64)>,
}

// SAFETY: D3D12 heaps may be used from any thread; the rest is plain data.
unsafe impl Send for Heaps {}

static HEAPS: Mutex<Option<Heaps>> = Mutex::new(None);

/// Textures placed and made as before because no piece could be had (too
/// big, or the heaps full), since the start; pieces used again.
pub(crate) static STATS: [AtomicU64; 3] = [const { AtomicU64::new(0) }; 3];

pub(crate) fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("VICTORY_PLACED_TEXTURES").is_none_or(|v| v != "0"))
}

/// A texture of `desc` placed in a heap of ours, in `state`; `None` when it
/// gets no piece (too big, heaps full, or the host refuses), and the caller
/// makes it with memory of its own.
pub(crate) fn place(device: &ID3D12Device, desc: &D3D12_RESOURCE_DESC, state: D3D12_RESOURCE_STATES) -> Option<ID3D12Resource> {
    // SAFETY: a plain query of a description.
    let info = unsafe { device.GetResourceAllocationInfo(0, std::slice::from_ref(desc)) };
    if info.SizeInBytes == u64::MAX || info.SizeInBytes == 0 || info.Alignment > PIECE_LEAST {
        return None;
    }
    let size = info.SizeInBytes.max(PIECE_LEAST).next_power_of_two();
    if size > PIECE_MOST {
        STATS[1].fetch_add(1, Relaxed);
        return None;
    }
    let mut all = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    let h = all.get_or_insert_with(|| Heaps {
        heaps: Vec::new(),
        cutting: None,
        spare: BTreeMap::new(),
        held: HashMap::new(),
    });
    let rested = h
        .spare
        .get_mut(&size)
        .and_then(|list| list.front().is_some_and(|&(_, _, at)| at.elapsed() >= REST).then(|| list.pop_front()).flatten());
    let (heap, offset) = match rested {
        Some((heap, offset, _)) => {
            STATS[2].fetch_add(1, Relaxed);
            (heap, offset)
        }
        None => {
            if h.cutting.is_none_or(|(_, cut)| cut + size > HEAP_BYTES) {
                if (h.heaps.len() as u64 + 1) * HEAP_BYTES > heaps_most() {
                    STATS[1].fetch_add(1, Relaxed);
                    return None;
                }
                let heap_desc = D3D12_HEAP_DESC {
                    SizeInBytes: HEAP_BYTES,
                    Properties: D3D12_HEAP_PROPERTIES {
                        Type: D3D12_HEAP_TYPE_DEFAULT,
                        ..Default::default()
                    },
                    Alignment: 0,
                    Flags: D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
                };
                let mut made: Option<ID3D12Heap> = None;
                // SAFETY: plain creation; the out pointer is ours.
                if let Err(e) = unsafe { device.CreateHeap::<ID3D12Heap>(&raw const heap_desc, &raw mut made) } {
                    static ONCE: std::sync::Once = std::sync::Once::new();
                    ONCE.call_once(|| tracing::warn!(error = %e, "no heap for placed textures: they are made as before"));
                    return None;
                }
                crate::host::name(made.as_ref()?, || "placed textures".to_string());
                h.heaps.push(made?);
                let n = h.heaps.len();
                if n.is_power_of_two() {
                    tracing::info!(heaps = n, "heaps for placed textures made so far ({} MiB each)", HEAP_BYTES >> 20);
                }
                h.cutting = Some((n - 1, 0));
            }
            let (heap, cut) = h.cutting.as_mut()?;
            // Pieces are powers of two: each lies on its own size.
            let at = *cut;
            *cut += size;
            (*heap, at)
        }
    };
    let mut texture: Option<ID3D12Resource> = None;
    // SAFETY: the piece is ours alone, large enough and aligned for the
    // description (asked above); the out pointer is ours.
    let made = unsafe { device.CreatePlacedResource(&h.heaps[heap], offset, desc, state, None, &mut texture) };
    match (made, texture) {
        (Ok(()), Some(texture)) => {
            h.held.insert(texture.as_raw() as usize, (size, heap, offset));
            STATS[0].fetch_add(1, Relaxed);
            Some(texture)
        }
        _ => {
            // Never used: no rest needed.
            h.spare.entry(size).or_default().push_front((heap, offset, Instant::now() - REST));
            None
        }
    }
}

/// `texture` goes: its piece, if it has one, is spare. Called when nothing
/// holds the texture any more (the card is done with it).
pub(crate) fn release(texture: &ID3D12Resource) {
    let mut all = HEAPS.lock().unwrap_or_else(|e| e.into_inner());
    let Some(h) = all.as_mut() else {
        return;
    };
    if let Some((size, heap, offset)) = h.held.remove(&(texture.as_raw() as usize)) {
        h.spare.entry(size).or_default().push_back((heap, offset, Instant::now()));
    }
}
