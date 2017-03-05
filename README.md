# Beevage

Bounding volume hierarchy (BVH) construction for ray tracing, and
construction *only*. No traversal code is included, so this crate is
almost entirely agnostic w.r.t. the primitive shapes, memory layout,
or traversal strategy. On the other hand, its tree representation
also isn't optimized for traversal, so it's recommended to transfer
the tree structure built with this crate to a more specialized
representation.

## License

Licensed under either of

 * Apache License, Version 2.0, ([LICENSE-APACHE](LICENSE-APACHE) or
 * http://www.apache.org/licenses/LICENSE-2.0) MIT license
   ([LICENSE-MIT](LICENSE-MIT) or http://opensource.org/licenses/MIT)

at your option.
