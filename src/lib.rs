//! [Bounding Volume Hierarchy (BVH)][bvh] construction for ray tracing, and construction *only*.
//!
//! Beevage constructs high-quality binary BVHs in three-dimensional space.
//! It works with any kind of shape that supports bounding box calculations.
//! Only construction is implemented, this crate doesn't come with any traversal operations.
//! Consequently, it is agnostic both w.r.t. what kinds of geometric primitives you put into it
//! and what exactly you later do with the BVH.
//! The construction method of choice produces trees that work best for ray tracing, but otherwise
//! it is pretty general-purpose.
//!
//!
//! [bvh]: https://en.wikipedia.org/wiki/Bounding_volume_hierarchy

extern crate ordered_float;
extern crate rayon;
pub extern crate cgmath;
pub extern crate beebox;

use std::f32;
use std::mem;
use std::marker::PhantomData;
use std::ops::{Index, Range};
use std::cmp::min;
use std::sync::atomic::{AtomicUsize, Ordering};
use cgmath::Vector3;
use beebox::Aabb;
use ordered_float::NotNaN;
use rayon::prelude::*;

/// A completed BVH.
pub struct Bvh<'a> {
    pub root: Node,
    pub primitives: Vec<PrimRef<'a>>,
    pub node_count: usize,
}

/// This type represents a geometric primitive in a leaf node of the BVH.
///
/// Conceptually this is just a pointer (with lifetime `'p`) to an element of the geometric
/// primitives slice passed into the construction routine. Its representation, however, is the
/// *index* of that primitive in that slice. The reasons for that are mostly implementation
/// convenience and the possibility for a space optimization.
#[derive(Copy, Clone)]
pub struct PrimRef<'p>(usize, PhantomData<&'p ()>); // FIXME should be u32

impl<'p> PrimRef<'p> {
    fn new(index: usize) -> Self {
        PrimRef(index, PhantomData)
    }

    /// Return the index of the corresponding primitive.
    pub fn index(&self) -> usize {
        self.0
    }
}

/// This split axis of an interior node.
pub enum Axis {
    X,
    Y,
    Z,
}

// TODO create a builder for this
/// Configuration for the binned SAH construction algorithm.
#[derive(Copy, Clone, Debug)]
pub struct Config {
    /// The number of buckets to use for split plane selection.
    pub bucket_count: usize, // FIXME this should be u32
    /// The SAH cost of a traversal step, relative to one primitive intersection test.
    /// (It is assumed that all primitives are equally expensive to test for test.)
    pub traversal_cost: f32, // FIXME don't assume that all primitives are equally expensive
    /// The maximum depth of the tree. If this depth is reached, a leaf node is forced regardless
    /// of SAH cost and number of primitives.
    pub max_depth: usize,
}

impl Axis {
    fn new(axis_id: usize) -> Self {
        match axis_id {
            0 => Axis::X,
            1 => Axis::Y,
            2 => Axis::Z,
            i => panic!("non-existent axis {}", i),
        }
    }
}

// FIXME arena allocation for the Nodes

/// A node in the BVH. All nodes, whether interior or leaves, have a bounding box.
pub enum Node {
    /// A leaf node encompassing a variable number of primitives.
    Leaf {
        /// Bounding box of the leaf. Note that this is not necessarily encompass all primitives'
        /// bounding boxes, since some construction algorithms use "spatial splits": assingning
        /// one primitive to *several* nodes with smaller bounding boxes which encompass only
        /// parts of the primitive.
        bb: Aabb,
        /// The primitives of this leaf. This range refers to `Bvh::primitives`, i.e.,
        /// the leaf's primitives are `bvh.primitives[primitive_range]`.
        primitive_range: Range<usize>, // FIXME it should be Range<u32>
    },
    /// A leaf node with two children.
    Inner {
        /// The union of the child nodes' bounding boxes.
        bb: Aabb,
        /// The child nodes.
        children: Box<(Node, Node)>,
        /// The axis used for the split plane.
        axis: Axis,
    },
}

/// Data shared by all subtree builders
#[derive(Debug)]
struct BuilderCommon {
    node_count: AtomicUsize,
    prim_bbs: Box<[Aabb]>,
    _root_bb: Aabb, // TODO use it for SBVH
    config: Config,
}

#[derive(Clone, Debug)]
struct Bucket {
    bb: Aabb,
    count: usize,
}

impl Bucket {
    fn empty() -> Self {
        Bucket {
            count: 0,
            bb: Aabb::empty(),
        }
    }
}

#[derive(Debug)]
struct Buckets<'c> {
    common: &'c BuilderCommon,
    centroid_bb: Aabb,
    axis: usize,
    buckets: Vec<Bucket>,
}

impl<'c> Buckets<'c> {
    fn new<I>(common: &'c BuilderCommon, centroids: I) -> Result<Self, ()>
        where I: Iterator<Item = Vector3<f32>>
    {
        let centroid_bb = Aabb::new(centroids);
        let axis = centroid_bb.largest_axis();
        if centroid_bb.min()[axis] == centroid_bb.max()[axis] {
            return Err(());
        }
        let bucket_count = common.config.bucket_count;
        Ok(Buckets {
            common: common,
            centroid_bb: centroid_bb,
            axis: axis,
            buckets: vec![Bucket::empty(); bucket_count],
        })
    }

    fn index(&self, x: Vector3<f32>) -> usize {
        let (left_border, right_border) = (self.centroid_bb.min()[self.axis],
                                           self.centroid_bb.max()[self.axis]);
        let relative_pos = (x[self.axis] - left_border) / (right_border - left_border);
        let bucket_count = self.common.config.bucket_count;
        min((bucket_count as f32 * relative_pos) as usize,
            bucket_count - 1)
    }

    fn add_primitive(&mut self, bb: Aabb) {
        let b = self.index(bb.centroid());
        self.buckets[b].bb.add_box(bb);
        self.buckets[b].count += 1;
    }

    fn child<R>(&self, range: R) -> (Aabb, usize)
        where Vec<Bucket>: Index<R, Output = [Bucket]>
    {
        let mut bb = Aabb::empty();
        let mut count = 0;
        for bucket in &self.buckets[range] {
            bb.add_box(bucket.bb);
            count += bucket.count;
        }
        (bb, count)
    }
}

/// Data for the construction process of a particular subtree
struct SubtreeBuilder<'c, 'p: 'c> {
    common: &'c BuilderCommon,
    prims: &'c mut [PrimRef<'p>],
    bb: Aabb,
    prim_offset: usize,
    depth: usize,
}


impl<'c, 'p> SubtreeBuilder<'c, 'p> {
    fn new(common: &'c BuilderCommon,
           prims: &'c mut [PrimRef<'p>],
           bb: Aabb,
           prim_offset: usize,
           depth: usize)
           -> Self {
        assert!(depth < common.config.max_depth,
                "BVH is becoming unreasonably deep --- infinite loop?");
        common.node_count.fetch_add(1, Ordering::SeqCst);
        SubtreeBuilder {
            common: common,
            prims: prims,
            bb: bb,
            prim_offset: prim_offset,
            depth: depth + 1,
        }
    }

    fn make_inner(mut self, buckets: &Buckets, split: usize) -> Node {
        let bb = self.bb;
        let mid = self.partition(buckets, split);
        let (prims_l, prims_r) = self.prims.split_at_mut(mid);
        let offset_l = self.prim_offset;
        let offset_r = offset_l + mid;
        let (bb_l, count_l) = buckets.child(..split);
        let (bb_r, count_r) = buckets.child(split..);
        assert_eq!(mid, count_l);
        assert_eq!(count_l, prims_l.len());
        assert_eq!(count_r, prims_r.len());
        let (l, r) = (SubtreeBuilder::new(self.common, prims_l, bb_l, offset_l, self.depth),
                      SubtreeBuilder::new(self.common, prims_r, bb_r, offset_r, self.depth));
        let children = rayon::join(move || l.build(), move || r.build());
        Node::Inner {
            bb: bb,
            children: Box::new(children),
            axis: Axis::new(buckets.axis),
        }
    }

    fn make_leaf(self) -> Node {
        let start = self.prim_offset;
        let end = start + self.prims.len();
        Node::Leaf {
            bb: self.bb,
            primitive_range: start..end,
        }
    }

    fn build(self) -> Node {
        if self.prims.len() == 1 {
            return self.make_leaf();
        }
        if let Ok(buckets) = self.buckets() {
            let (split_cost, split) = self.best_split(&buckets);
            let leaf_cost = self.prims.len() as f32;
            if leaf_cost <= split_cost {
                self.make_leaf()
            } else {
                self.make_inner(&buckets, split)
            }
        } else {
            // Couldn't construct buckets, perhaps because the centroids are all clumped together.
            // Give up and put all remaining primitives in a big leaf
            self.make_leaf()
        }
    }

    fn buckets(&self) -> Result<Buckets<'c>, ()> {
        let prim_bbs = &self.common.prim_bbs;
        let centroids = self.prims.iter().map(|p| prim_bbs[p.index()].centroid());
        let mut buckets = try!(Buckets::new(self.common, centroids));
        for prim in self.prims.iter() {
            buckets.add_primitive(self.common.prim_bbs[prim.index()]);
        }
        Ok(buckets)
    }

    fn best_split(&self, buckets: &Buckets) -> (f32, usize) {
        let cfg = &self.common.config;
        let possible_splits = 1..cfg.bucket_count;
        let mut costs = Vec::with_capacity(possible_splits.len());
        for split in possible_splits.clone() {
            let (bb0, count0) = buckets.child(..split);
            let (bb1, count1) = buckets.child(split..);
            let cost0 = count0 as f32 * bb0.surface_area() / self.bb.surface_area();
            let cost1 = count1 as f32 * bb1.surface_area() / self.bb.surface_area();
            let cost = cfg.traversal_cost + cost0 + cost1;
            costs.push(NotNaN::new(cost).unwrap());
        }
        let (cost_not_nan, split) = costs.into_iter().zip(possible_splits).min().unwrap();
        (cost_not_nan.into_inner(), split)
    }

    fn partition(&mut self, buckets: &Buckets, split_bucket: usize) -> usize {
        // The primitives slice is composed of three sub-slices (in this order):
        // 1. Those known to be left of the split plane,
        // 2. The still-unclassified ones
        // 3. Those known to be right of the split plane
        // We start with everything uncategorized and grow the left and right slices in the loop.
        // The slices are represented by integers (left, remaining) s.t. [0..left] is the left
        // slice, [left..left+remaining] is the uncategorized slice, and [left+remaining..]
        // is the right slice.
        // FIXME this is a pretty naive partitioning algorithm, there are better ones
        let mut left = 0;
        let mut remaining = self.prims.len();
        let prim_bbs = &self.common.prim_bbs;
        let is_left = |p: &PrimRef| buckets.index(prim_bbs[p.index()].centroid()) < split_bucket;
        while remaining > 0 {
            let uncategorized = &mut self.prims[left..left + remaining];
            // Split off the first element of uncategorized, to be able to swap it if necessary
            let (uncat_start, uncat_rest) = uncategorized.split_at_mut(1);
            let prim = &mut uncat_start[0];
            remaining -= 1;
            if is_left(prim) {
                left += 1;
            } else if let Some(last_uncat) = uncat_rest.last_mut() {
                mem::swap(prim, last_uncat);
            }
        }
        left
    }
}

/// The interface between `beevage` and geometric primitives.
pub trait Primitive: Send + Sync {
    /// Return the bounding box of the primitive.
    fn bounding_box(&self) -> Aabb;
}

/// Construct a BVH using the surface area heuristic approximated via binning. See:
///
/// > Wald, Ingo. "On fast construction of SAH-based bounding volume hierarchies."
/// > 2007 IEEE Symposium on Interactive Ray Tracing. IEEE, 2007.
///
/// # Returns
///
/// The root node of the BVH, and the number of nodes in the BVH.
pub fn binned_sah<P: Primitive>(config: Config, prims: &[P], root_bb: Aabb) -> Bvh {
    let prim_bbs: Vec<_> = prims.par_iter().map(Primitive::bounding_box).collect();
    assert!(prim_bbs.len() == prim_bbs.capacity());
    let common = BuilderCommon {
        config: config,
        prim_bbs: prim_bbs.into_boxed_slice(),
        node_count: AtomicUsize::new(0),
        _root_bb: root_bb,
    };
    let mut prim_refs: Vec<_> = (0..prims.len()).map(PrimRef::new).collect();
    let root;
    {
        let builder = SubtreeBuilder::new(&common, &mut prim_refs, root_bb, 0, 0);
        root = builder.build();
    }
    Bvh {
        root: root,
        primitives: prim_refs,
        node_count: common.node_count.load(Ordering::SeqCst),
    }
}
