//! A cache for storing the results of layout computation

#![allow(clippy::unusual_byte_groupings)]

use crate::geometry::Size;
use crate::style::AvailableSpace;
use crate::tree::{LayoutInput, LayoutOutput, RunMode};
use crate::RequestedAxis;

/// The number of cache entries for each node in the tree
///
/// Nine before this was measured, which was one slot per category under the
/// old fixed-slot scheme. With slots as plain storage that number is a working
/// set instead, and nine is well below ours: intrinsic sizing asks a node many
/// more distinct questions than that in a single pass, so the cache spent its
/// time evicting entries it was about to want.
///
/// Measured against typing one character into a ~7,000 node tree, counting
/// `compute_child_layout` calls and the distinct nodes they touched:
///
/// |  size | recomputations | distinct nodes | layout phase |
/// | ----: | -------------: | -------------: | -----------: |
/// |     9 |          3,334 |            127 |      2.60 ms |
/// |    12 |          3,747 |            123 |      3.50 ms |
/// |    16 |            337 |             24 |      0.47 ms |
/// |    24 |            141 |             15 |      0.27 ms |
/// |    32 |            141 |             15 |      0.26 ms |
///
/// It is a cliff rather than a curve. Below the working set the round-robin
/// eviction thrashes and more slots do not help — 12 is no better than 9 — and
/// above it almost everything hits. 24 is where it saturates: 32 is identical.
///
/// The 15 distinct nodes are the number to read. Around that many are genuinely
/// dirty per keystroke, so the cache now recomputes what changed and nothing
/// else, which is what it was always for.
///
/// **This constant was first tuned to 16 against a workload that was itself
/// broken.** `element.style.x = y` was a silent no-op in the embedder at the
/// time, so an autosizing text field never actually resized and the layout it
/// provoked was smaller than the real one. With that fixed the working set grew
/// and 16 became the thrashing case. A constant is only as good as the workload
/// it was measured against, and a broken workload measures a smaller one.
const CACHE_SIZE: usize = 24;

// Manually written-out results of float to u32 bit casts because
// `f32::to_bits` is not yet const at our MSRV.

/// `f32::INFINITY` as a u32
const INFINITY_BITS: u32 = 0b_0_11111111_00000000000000000000000_u32;
/// `f32::NEG_INFINITY` as a u32
const NEG_INFINITY_BITS: u32 = 0b_1_11111111_00000000000000000000000_u32;

// The `CacheKey` encodes two f32s as a u64. We know that the f32s will always be
// non-negative, so we pack two extra bits encoding the `RequestedAxis` into the
// sign bits of the f32s. These constants help to encode and decode those bits.

/// The sign bit of the first f32
const SIGN_BIT_1: u64 = 1u64 << 63;
/// The sign bit of the second f32
const SIGN_BIT_2: u64 = 1u64 << 31;
/// Mask of both sign bits (used to compute NON_SIGN_BITS_MASK)
const BOTH_SIGN_BITS_MASK: u64 = SIGN_BIT_1 | SIGN_BIT_2;
/// Mask of excluding the sign bits (used when setting/getting the size excluding the packed bits)
const NON_SIGN_BITS_MASK: u64 = !BOTH_SIGN_BITS_MASK;

/// Mask which includes only the bits which encode the x-axis value that we can use to ignore the
/// y-axis value when comparing a cache key.
const X_AXIS_VALUE_MASK: u64 = (u32::MAX as u64) << 32;

/// Pack `Option<f32>` into `u32`
#[inline(always)]
fn option_cache_key(input: Option<f32>) -> u32 {
    match input {
        Some(value) => value.to_bits(),
        None => INFINITY_BITS,
    }
}

/// Pack `Size<Option<f32>>` into `u64`
#[inline(always)]
fn size_option_cache_key(input: Size<Option<f32>>) -> u64 {
    (option_cache_key(input.width) as u64) << 32 | option_cache_key(input.height) as u64
}

/// Pack `AvailableSpace` into `u32`
#[inline(always)]
fn available_space_cache_key(input: AvailableSpace) -> u32 {
    match input {
        AvailableSpace::Definite(value) => (-value).to_bits(),
        AvailableSpace::MinContent => NEG_INFINITY_BITS,
        AvailableSpace::MaxContent => INFINITY_BITS,
    }
}

/// Pack `Size<AvailableSpace>` into `u64`
#[inline(always)]
#[allow(dead_code)]
fn size_available_space_cache_key(input: Size<AvailableSpace>) -> u64 {
    (available_space_cache_key(input.width) as u64) << 32 | available_space_cache_key(input.height) as u64
}

/// Encodes combination of a `known_dimension` (Option<f32>) and `AvailableSpace` in
/// a single dimension into a cache key in a single dimension.
#[inline(always)]
fn mixed_cache_key(kd: Option<f32>, avs: AvailableSpace) -> u32 {
    kd.map(|kd| kd.to_bits()).unwrap_or_else(|| available_space_cache_key(avs))
}

/// Encodes combination of a `known_dimension` (Option<f32>) and `AvailableSpace` in
/// two dimensions into a cache key in a single dimension.
#[inline(always)]
fn size_mixed_cache_key(kd: Size<Option<f32>>, avs: Size<AvailableSpace>) -> u64 {
    (mixed_cache_key(kd.width, avs.width) as u64) << 32 | mixed_cache_key(kd.height, avs.height) as u64
}

/// Space-optimised cache key that packs bits into as small a size as possible
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
struct CacheKey {
    /// The initial cached size of the node itself
    kd_available_space: u64,
    /// The initial cached size of the parent's node
    parent_size: u64,
}

impl CacheKey {
    #[inline(always)]
    #[allow(dead_code)]
    /// Return the parent size with the extra bits that encode the requested axis masked out
    fn parent_size(&self) -> u64 {
        self.parent_size & NON_SIGN_BITS_MASK
    }

    /// Return the parent size with the extra bits that encode the requested axis masked out
    /// And the y-axis value masked out
    fn x_axis_parent_size(&self) -> u64 {
        self.parent_size & (X_AXIS_VALUE_MASK & NON_SIGN_BITS_MASK)
    }
}

impl From<&LayoutInput> for CacheKey {
    fn from(input: &LayoutInput) -> Self {
        // Pack axis enum into spare bits in the known_dimensions and available_space values
        let extra_bits = match input.axis {
            RequestedAxis::Horizontal => SIGN_BIT_1,
            RequestedAxis::Vertical => SIGN_BIT_2,
            RequestedAxis::Both => SIGN_BIT_1 | SIGN_BIT_2,
        };

        Self {
            kd_available_space: size_mixed_cache_key(input.known_dimensions, input.available_space),
            parent_size: (size_option_cache_key(input.parent_size) & NON_SIGN_BITS_MASK) | extra_bits,
        }
    }
}

/// Cached intermediate layout results
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub(crate) struct CacheEntry<T> {
    /// The key for the cache entry
    key: CacheKey,
    /// The cached size and baselines of the item
    content: T,
}

/// The sizing question a measure entry answered, kept unpacked.
///
/// [`CacheKey`] packs its inputs into two `u64`s, which makes equality cheap and
/// makes any other comparison impossible. Answering "is this stored result still
/// correct for a different question" needs the numbers themselves, so measure
/// entries carry them alongside the key.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub(crate) struct MeasureInputs {
    /// The dimensions that were already decided when this entry was computed.
    known_dimensions: Size<Option<f32>>,
    /// The space the node was offered when this entry was computed.
    available_space: Size<AvailableSpace>,
}

/// Whether a stored measurement still answers a new question, per axis.
///
/// Taffy's `get` accepts an entry only on exact key equality. That is sound but
/// pessimistic: intrinsic sizing re-descends the same subtree with a sequence of
/// different definite widths, and an answer measured under a wider offer is
/// still the right answer whenever the content fit inside the narrower one.
///
/// This is Yoga's `canUseCachedMeasurement`, whose three heuristics live in
/// `yoga/algorithm/Cache.cpp`. The one that matters here is
/// `newSizeIsStricterAndStillValid`: both offers definite, the new one smaller,
/// and the measured content already inside it.
///
/// Being wrong here is a wrong layout rather than a slow one, so each arm below
/// only returns true when the measured extent is provably unaffected by the
/// difference between the two offers.
#[inline]
fn axis_still_valid(
    stored_known: Option<f32>,
    new_known: Option<f32>,
    stored_space: AvailableSpace,
    new_space: AvailableSpace,
    measured: f32,
) -> bool {
    use AvailableSpace::{Definite, MaxContent, MinContent};

    // A known dimension is imposed on the result rather than discovered, so it
    // has to match exactly. Nothing about the offered space can rescue it.
    match (stored_known, new_known) {
        (Some(stored), Some(new)) => return stored == new,
        (None, None) => {}
        // One side had the dimension decided and the other did not: different
        // questions entirely.
        _ => return false,
    }

    match (stored_space, new_space) {
        // Identical offers: what the exact-match path already accepts.
        (MinContent, MinContent) | (MaxContent, MaxContent) => true,
        (Definite(stored), Definite(new)) => {
            // Equal is the plain hit. Otherwise the content must have fit inside
            // *both* offers: fitting the wider one alone does not show what a
            // narrower one would have wrapped.
            stored == new || (measured <= stored && measured <= new)
        }
        // Measured with no upper bound and it fit inside the new offer, so the
        // new offer never binds. Yoga's `oldSizeIsMaxContentAndStillFits`.
        (MaxContent, Definite(new)) => measured <= new,
        // The reverse is not safe: a max-content query can be wider than
        // whatever the stored definite offer allowed.
        (Definite(_), MaxContent) => false,
        // Min-content is a different measurement, not a looser or tighter one.
        (MinContent, _) | (_, MinContent) => false,
    }
}

/// A cache for caching the results of a sizing a Grid Item or Flexbox Item
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize))]
pub struct Cache {
    /// The cache entry for the node's final layout
    final_layout_entry: Option<CacheEntry<LayoutOutput>>,
    /// The cache entries for the node's preliminary size measurements
    measure_entries: [Option<CacheEntry<Size<f32>>>; CACHE_SIZE],
    /// The unpacked question each measure entry answered, for validity testing.
    measure_inputs: [Option<MeasureInputs>; CACHE_SIZE],
    /// Tracks if all cache entries are empty
    is_empty: bool,
    /// Round-robin cursor for evicting when every slot is taken.
    next_eviction: u8,
}

impl Default for Cache {
    fn default() -> Self {
        Self::new()
    }
}

impl Cache {
    /// Create a new empty cache
    pub const fn new() -> Self {
        Self {
            final_layout_entry: None,
            measure_entries: [None; CACHE_SIZE],
            measure_inputs: [None; CACHE_SIZE],
            is_empty: true,
            next_eviction: 0,
        }
    }

    /// Choose which slot to write a measurement into.
    ///
    /// The slot function this replaces mapped each question to one fixed slot,
    /// and documented the assumption that made it safe: "definite available
    /// space shares a cache slot with max-content because a node will generally
    /// be sized under one or the other but not both."
    ///
    /// Intrinsic sizing breaks that assumption. It re-descends the same subtree
    /// offering a sequence of different definite widths, and with neither
    /// dimension known every one of those lands in the same slot, so each
    /// measurement destroys the one before it. The cache was evicting the
    /// entries it was about to be asked for.
    ///
    /// Slots are only storage: `get` already scans all of them. So the rule is
    /// simply to not clobber a different question — reuse the entry for this
    /// exact key, else take an empty slot, else evict the oldest write.
    ///
    /// Eviction is insertion-ordered rather than least-recently-used, which is
    /// what Chromium's `LayoutResult` cache does. `get` takes `&self` and so
    /// cannot record a touch, and adding interior mutability to a lookup on the
    /// hottest path in layout costs more than the better eviction is worth at
    /// nine entries.
    #[inline]
    fn slot_for(&mut self, key: &CacheKey) -> usize {
        for (index, entry) in self.measure_entries.iter().enumerate() {
            match entry {
                Some(entry) if entry.key == *key => return index,
                _ => {}
            }
        }
        if let Some(index) = self.measure_entries.iter().position(Option::is_none) {
            return index;
        }
        let index = self.next_eviction as usize;
        self.next_eviction = (self.next_eviction + 1) % CACHE_SIZE as u8;
        index
    }

    /// Try to retrieve a cached result from the cache
    #[inline]
    pub fn get(&self, input: &LayoutInput) -> Option<LayoutOutput> {
        let key = CacheKey::from(input);
        match input.run_mode {
            RunMode::PerformLayout => self.final_layout_entry.filter(|entry| entry.key == key).map(|e| e.content),
            RunMode::ComputeSize => {
                for (entry, stored) in self.measure_entries.iter().zip(self.measure_inputs.iter()) {
                    let Some(entry) = entry else { continue };

                    // Percentages resolve against the parent, so an entry
                    // measured under a different parent size answers a different
                    // question no matter how the offers compare.
                    if entry.key.x_axis_parent_size() != key.x_axis_parent_size() {
                        continue;
                    }

                    if entry.key.kd_available_space == key.kd_available_space {
                        return Some(LayoutOutput::from_outer_size(entry.content));
                    }

                    // Not the same question. Ask whether the answer still holds.
                    let Some(stored) = stored else { continue };
                    let width_ok = axis_still_valid(
                        stored.known_dimensions.width,
                        input.known_dimensions.width,
                        stored.available_space.width,
                        input.available_space.width,
                        entry.content.width,
                    );
                    let height_ok = axis_still_valid(
                        stored.known_dimensions.height,
                        input.known_dimensions.height,
                        stored.available_space.height,
                        input.available_space.height,
                        entry.content.height,
                    );
                    if width_ok && height_ok {
                        return Some(LayoutOutput::from_outer_size(entry.content));
                    }
                }

                None
            }
            RunMode::PerformHiddenLayout => None,
        }
    }

    /// Store a computed size in the cache
    pub fn store(&mut self, input: &LayoutInput, layout_output: LayoutOutput) {
        let key = CacheKey::from(input);
        match input.run_mode {
            RunMode::PerformLayout => {
                self.is_empty = false;
                self.final_layout_entry = Some(CacheEntry { key, content: layout_output })
            }
            RunMode::ComputeSize => {
                self.is_empty = false;
                let slot = self.slot_for(&key);
                self.measure_entries[slot] = Some(CacheEntry { key, content: layout_output.size });
                self.measure_inputs[slot] = Some(MeasureInputs {
                    known_dimensions: input.known_dimensions,
                    available_space: input.available_space,
                });
            }
            RunMode::PerformHiddenLayout => {}
        }
    }

    /// Clear all cache entries and reports clear operation outcome ([`ClearState`])
    pub fn clear(&mut self) -> ClearState {
        if self.is_empty {
            return ClearState::AlreadyEmpty;
        }
        self.is_empty = true;
        self.final_layout_entry = None;
        self.measure_entries = [None; CACHE_SIZE];
        self.measure_inputs = [None; CACHE_SIZE];
        ClearState::Cleared
    }

    /// Returns true if all cache entries are None, else false
    pub fn is_empty(&self) -> bool {
        self.final_layout_entry.is_none() && !self.measure_entries.iter().any(|entry| entry.is_some())
    }
}

/// Clear operation outcome. See [`Cache::clear`]
pub enum ClearState {
    /// Cleared some values
    Cleared,
    /// Everything was already cleared
    AlreadyEmpty,
}
