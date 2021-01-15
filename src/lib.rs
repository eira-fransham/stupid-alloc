#![deny(missing_docs)]

//! # `stupid-alloc`
//!
//! A very simple, serializable allocator, designed for external memory such as long-lived files
//! or GPU buffers. It has approximately O(log(N)) allocation and deallocation where N is the
//! fragmentation - i.e., the number of free spaces of any size. If you want to avoid
//! fragmentation, you can align allocations to 8-word boundaries or run a compaction step.
//!
//! Since it is serializable when using the `serde` feature, if it is being used to allocate space
//! in a file the allocation metadata can, too, can be stored directly in the file itself.
//!
//! Originally used to allocate and reuse space on the runtime stack for code compiled by
//! [`lightbeam`][lightbeam].
//!
//! [lightbeam]: https://github.com/Vurich/wasmtime/blob/master/crates/lightbeam

use std::{
    collections::{BTreeMap, HashSet},
    ops,
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// A pointer into the allocated space. Does not carry its size with it,
/// the size is implicit.
#[derive(Default, PartialOrd, Ord, PartialEq, Eq, Hash, Debug, Copy, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Ptr(pub usize);

impl Ptr {
    fn offset(self, size: Size) -> Ptr {
        Ptr(self.0 + size.0)
    }
}

/// The size of the whole allocatable area, or of a specific allocation.
#[derive(Default, PartialOrd, Ord, PartialEq, Eq, Hash, Debug, Copy, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Size(pub usize);

impl ops::Add for Size {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Size(self.0 + other.0)
    }
}

impl ops::Sub for Size {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Size(self.0 - other.0)
    }
}

/// The allocator itself. See documentation for individual functions for more information.
/// The implementation is 100% safe Rust.
#[derive(Debug, Default, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct Alloc {
    size: Size,
    blocks_by_address: BTreeMap<Ptr, Size>,
    blocks_by_size: BTreeMap<Size, HashSet<Ptr>>,
}

impl Alloc {
    /// Create a new allocator of the specified size.
    pub fn new(bytes: Size) -> Self {
        let mut out = Self::default();
        out.set_size(bytes);
        out.free(Ptr(0), bytes);
        out
    }

    /// Resize the maximum size of the allocatable space. This does _not_ free the newly-available space if
    /// the new size is larger than the existing size. If you want to mark the space between the previous size
    /// and the new size as free, you must manually call `free`.
    pub fn set_size(&mut self, size: Size) {
        use std::cmp::Ordering;

        match size.cmp(&self.size) {
            Ordering::Less => self.truncate(size),
            Ordering::Equal | Ordering::Greater => {}
        }

        self.size = size;
    }

    /// Return the size of the current maximum allocatable space.
    pub fn size(&self) -> Size {
        self.size
    }

    fn truncate(&mut self, size: Size) {
        self.mark_allocated(Ptr(size.0), self.size - size);
    }

    /// Explicitly mark an area as allocated. This is useful if, for example, part of the allocatable area
    /// must be reserved for specific uses by the host application.
    ///
    /// This cannot cause undefined behavior in the `unsafe` sense, but it assumes that the space to be
    /// marked allocated is currently free in its entirely, and it may panic or leave the allocator in an
    /// unexpected state if this is not upheld.
    pub fn mark_allocated(&mut self, ptr: Ptr, size: Size) {
        use std::cmp::Ordering;

        let end = ptr.offset(size);

        if let Some((&existing_ptr, &existing_size)) = self
            .blocks_by_address
            .range(..end)
            .last()
            .filter(|(p, s)| **p <= ptr && p.offset(**s) >= end)
        {
            match ptr.cmp(&existing_ptr) {
                Ordering::Less => unreachable!(),
                Ordering::Equal => {
                    self.remove_block(existing_ptr);
                    self.add_block(end, existing_size - size);
                }
                Ordering::Greater => {
                    let existing_end = existing_ptr.offset(existing_size);

                    self.modify_block(existing_ptr, Size(ptr.0 - existing_ptr.0));
                    self.add_block(end, Size(existing_end.0 - end.0));
                }
            }
        }
    }

    /// Return a pointer to a block of memory of the given size, or `None` if no such block exists.
    pub fn malloc(&mut self, size: Size) -> Option<Ptr> {
        use std::cmp::Ordering;

        let (&existing_size, ptrs) = self.blocks_by_size.range_mut(size..).next()?;

        let ptr = *ptrs.iter().next().expect("Allocator metadata corrupted");

        self.remove_block(ptr);

        match existing_size.cmp(&size) {
            Ordering::Less => unreachable!(),
            Ordering::Equal => {}
            Ordering::Greater => self.add_block(ptr.offset(size), existing_size - size),
        }

        Some(ptr)
    }

    /// Mark this range as free. This assumes that the area is currently marked as allocated, but `ptr` does _not_
    /// have to be returned by [Alloc::malloc]. It is entirely valid to allocate a larger area and then only free
    /// a small section thereof.
    pub fn free(&mut self, ptr: Ptr, size: Size) {
        let prev_block = self
            .blocks_by_address
            .range(..=ptr)
            .last()
            .filter(|(p, s)| p.offset(**s) == ptr)
            .map(|(p, s)| (*p, *s));
        let next_block = self
            .blocks_by_address
            .get_key_value(&ptr.offset(size))
            .map(|(p, s)| (*p, *s));

        match (prev_block, next_block) {
            (Some((prev_ptr, prev_size)), Some((next_ptr, next_size))) => {
                self.remove_block(next_ptr);
                self.modify_block(prev_ptr, prev_size + size + next_size);
            }
            (None, Some((next_ptr, next_size))) => {
                self.remove_block(next_ptr);
                self.add_block(ptr, size + next_size);
            }
            (Some((prev_ptr, prev_size)), None) => {
                self.modify_block(prev_ptr, size + prev_size);
            }
            (None, None) => {
                self.add_block(ptr, size);
            }
        }
    }

    /// Checks whether the entirety of the supplied range is currently free.
    pub fn is_free(&self, ptr: Ptr, size: Size) -> bool {
        self.blocks_by_address
            .range(..=ptr)
            .last()
            .map(|(p, s)| p.offset(*s) >= ptr.offset(size))
            .unwrap_or(false)
    }

    /// Returns the amount of space (not necessarily contiguous) between the start of the allocatable
    /// area and the end of the last allocated block.
    pub fn used_size(&self) -> Size {
        if let Some((p, _)) = self
            .blocks_by_address
            .iter()
            .last()
            .filter(|(p, s)| p.offset(**s).0 == self.size.0)
        {
            Size(p.0)
        } else {
            self.size
        }
    }

    fn modify_block(&mut self, ptr: Ptr, new_size: Size) {
        self.remove_block(ptr);
        self.add_block(ptr, new_size);
    }

    fn add_block(&mut self, ptr: Ptr, size: Size) {
        if size.0 > 0 {
            let existing = self.blocks_by_address.insert(ptr, size);
            debug_assert!(existing.is_none());
            let existing = self.blocks_by_size.entry(size).or_default().insert(ptr);
            debug_assert!(existing);
        }
    }

    fn remove_block(&mut self, ptr: Ptr) {
        use std::collections::btree_map::Entry;

        let size = self
            .blocks_by_address
            .remove(&ptr)
            .expect("Double-free'd block");
        match self.blocks_by_size.entry(size) {
            Entry::Occupied(mut entry) => {
                let existing = entry.get_mut().remove(&ptr);
                debug_assert!(existing, "Allocator metadata corrupted");
                if entry.get().is_empty() {
                    entry.remove_entry();
                }
            }
            Entry::Vacant(_) => panic!("Allocator metadata corrupted"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{Alloc, Ptr, Size};

    #[test]
    fn alloc_free() {
        let small = Size(8);
        let mid = Size(16);
        let large = Size(32);
        let mut alloc = Alloc::new(small + mid + large);

        let a = alloc.malloc(small).unwrap();
        let b = alloc.malloc(mid).unwrap();
        let c = alloc.malloc(large).unwrap();
        assert_eq!(a, Ptr(0));
        assert_eq!(b, Ptr(8));
        assert_eq!(c, Ptr(24));
        assert!(alloc.malloc(small).is_none());

        alloc.free(a, small);
        alloc.free(b, mid);

        let a = alloc.malloc(small).unwrap();
        let b1 = alloc.malloc(small).unwrap();
        let b2 = alloc.malloc(small).unwrap();

        assert!(alloc.malloc(small).is_none());
        assert_eq!(a, Ptr(0));
        assert_eq!(b1, Ptr(8));
        assert_eq!(b2, Ptr(16));
    }

    #[test]
    fn mark_allocated() {
        let small = Size(8);
        let mid = Size(16);
        let large = Size(32);
        let mut alloc = Alloc::new(small + mid + large);

        let a = Ptr(0);
        let b = Ptr(8);
        alloc.mark_allocated(b, mid);
        alloc.mark_allocated(a, small);

        alloc.free(a, small);
        alloc.free(b, mid);
        alloc.mark_allocated(Ptr(24), large);

        let a = alloc.malloc(small).unwrap();
        let b1 = alloc.malloc(small).unwrap();
        let b2 = alloc.malloc(small).unwrap();

        assert_eq!(a, Ptr(0));
        assert_eq!(b1, Ptr(8));
        assert_eq!(b2, Ptr(16));
    }

    #[cfg(feature = "serde")]
    mod serde {
        use super::{Alloc, Ptr, Size};

        macro_rules! round_trip {
            ($name:expr) => {
                serde_json::from_str::<Alloc>(&serde_json::to_string($name).unwrap()).unwrap()
            };
        }

        #[test]
        fn alloc_free() {
            let small = Size(8);
            let mid = Size(16);
            let large = Size(32);
            let mut alloc = Alloc::new(small + mid + large);

            let a = alloc.malloc(small).unwrap();
            let mut alloc = round_trip!(&alloc);
            let b = alloc.malloc(mid).unwrap();
            let mut alloc = round_trip!(&alloc);
            let c = alloc.malloc(large).unwrap();
            let mut alloc = round_trip!(&alloc);
            assert_eq!(a, Ptr(0));
            assert_eq!(b, Ptr(8));
            assert_eq!(c, Ptr(24));
            assert!(alloc.malloc(small).is_none());

            let mut alloc = round_trip!(&alloc);

            alloc.free(a, small);
            let mut alloc = round_trip!(&alloc);
            alloc.free(b, mid);
            let mut alloc = round_trip!(&alloc);

            let a = alloc.malloc(small).unwrap();
            let mut alloc = round_trip!(&alloc);
            let b1 = alloc.malloc(small).unwrap();
            let mut alloc = round_trip!(&alloc);
            let b2 = alloc.malloc(small).unwrap();
            let mut alloc = round_trip!(&alloc);

            assert!(alloc.malloc(small).is_none());
            assert_eq!(a, Ptr(0));
            assert_eq!(b1, Ptr(8));
            assert_eq!(b2, Ptr(16));
        }

        #[test]
        fn mark_allocated() {
            let small = Size(8);
            let mid = Size(16);
            let large = Size(32);
            let mut alloc = Alloc::new(small + mid + large);

            let a = Ptr(0);
            let b = Ptr(8);
            alloc.mark_allocated(b, mid);
            let mut alloc = round_trip!(&alloc);
            alloc.mark_allocated(a, small);

            let mut alloc = round_trip!(&alloc);

            alloc.free(a, small);
            let mut alloc = round_trip!(&alloc);
            alloc.free(b, mid);
            let mut alloc = round_trip!(&alloc);
            alloc.mark_allocated(Ptr(24), large);
            let mut alloc = round_trip!(&alloc);

            let a = alloc.malloc(small).unwrap();
            let mut alloc = round_trip!(&alloc);
            let b1 = alloc.malloc(small).unwrap();
            let mut alloc = round_trip!(&alloc);
            let b2 = alloc.malloc(small).unwrap();

            assert_eq!(a, Ptr(0));
            assert_eq!(b1, Ptr(8));
            assert_eq!(b2, Ptr(16));
        }
    }
}
