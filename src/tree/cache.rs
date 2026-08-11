//! A cache for storing the results of layout computation

#![allow(clippy::unusual_byte_groupings)]

use crate::geometry::Size;
use crate::style::AvailableSpace;
use crate::tree::{LayoutInput, LayoutOutput, RunMode};
use crate::RequestedAxis;

/// The number of cache entries for each node in the tree
const CACHE_SIZE: usize = 9;

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
        }
    }

    /// Return the cache slot to cache the current computed result in
    ///
    /// ## Caching Strategy
    ///
    /// We need multiple cache slots, because a node's size is often queried by it's parent multiple times in the course of the layout
    /// process, and we don't want later results to clobber earlier ones.
    ///
    /// The two variables that we care about when determining cache slot are:
    ///
    ///   - How many "known_dimensions" are set. In the worst case, a node may be called first with neither dimension known, then with one
    ///     dimension known (either width of height - which doesn't matter for our purposes here), and then with both dimensions known.
    ///   - Whether unknown dimensions are being sized under a min-content or a max-content available space constraint (definite available space
    ///     shares a cache slot with max-content because a node will generally be sized under one or the other but not both).
    ///
    /// ## Cache slots:
    ///
    /// - Slot 0: Both known_dimensions were set
    /// - Slots 1-4: 1 of 2 known_dimensions were set and:
    ///   - Slot 1: width but not height known_dimension was set and the other dimension was either a MaxContent or Definite available space constraintraint
    ///   - Slot 2: width but not height known_dimension was set and the other dimension was a MinContent constraint
    ///   - Slot 3: height but not width known_dimension was set and the other dimension was either a MaxContent or Definite available space constraintable space constraint
    ///   - Slot 4: height but not width known_dimension was set and the other dimension was a MinContent constraint
    /// - Slots 5-8: Neither known_dimensions were set and:
    ///   - Slot 5: x-axis available space is MaxContent or Definite and y-axis available space is MaxContent or Definite
    ///   - Slot 6: x-axis available space is MaxContent or Definite and y-axis available space is MinContent
    ///   - Slot 7: x-axis available space is MinContent and y-axis available space is MaxContent or Definite
    ///   - Slot 8: x-axis available space is MinContent and y-axis available space is MinContent
    #[inline]
    fn compute_cache_slot(known_dimensions: Size<Option<f32>>, available_space: Size<AvailableSpace>) -> usize {
        use AvailableSpace::{Definite, MaxContent, MinContent};

        let has_known_width = known_dimensions.width.is_some();
        let has_known_height = known_dimensions.height.is_some();

        // Slot 0: Both known_dimensions were set
        if has_known_width && has_known_height {
            return 0;
        }

        // Slot 1: width but not height known_dimension was set and the other dimension was either a MaxContent or Definite available space constraint
        // Slot 2: width but not height known_dimension was set and the other dimension was a MinContent constraint
        if has_known_width && !has_known_height {
            return 1 + (available_space.height == MinContent) as usize;
        }

        // Slot 3: height but not width known_dimension was set and the other dimension was either a MaxContent or Definite available space constraint
        // Slot 4: height but not width known_dimension was set and the other dimension was a MinContent constraint
        if has_known_height && !has_known_width {
            return 3 + (available_space.width == MinContent) as usize;
        }

        // Slots 5-8: Neither known_dimensions were set and:
        match (available_space.width, available_space.height) {
            // Slot 5: x-axis available space is MaxContent or Definite and y-axis available space is MaxContent or Definite
            (MaxContent | Definite(_), MaxContent | Definite(_)) => 5,
            // Slot 6: x-axis available space is MaxContent or Definite and y-axis available space is MinContent
            (MaxContent | Definite(_), MinContent) => 6,
            // Slot 7: x-axis available space is MinContent and y-axis available space is MaxContent or Definite
            (MinContent, MaxContent | Definite(_)) => 7,
            // Slot 8: x-axis available space is MinContent and y-axis available space is MinContent
            (MinContent, MinContent) => 8,
        }
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
                let cache_slot = Self::compute_cache_slot(input.known_dimensions, input.available_space);
                self.measure_entries[cache_slot] = Some(CacheEntry { key, content: layout_output.size });
                self.measure_inputs[cache_slot] = Some(MeasureInputs {
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
