# Rust Ownership

Ownership is Rust's most distinctive feature and underpins its memory-safety guarantees without a garbage collector. Every value in Rust has a single owner.

## The borrow checker

The borrow checker enforces that references never outlive the data they point to. At any time you can have either one mutable reference or any number of immutable references, but not both, which prevents data races at compile time.

## Move semantics

When a value is assigned to another variable or passed to a function, ownership is moved. The original binding can no longer be used. Types that implement the Copy trait, such as integers, are duplicated instead of moved.
