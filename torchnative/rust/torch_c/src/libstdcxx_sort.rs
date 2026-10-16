//! A transcription of libstdc++'s `std::sort`, `std::partial_sort` and
//! `std::nth_element`, taken from GCC 13's `bits/stl_algo.h` and
//! `bits/stl_heap.h` (`__introsort_loop` with `_S_threshold = 16`,
//! `__final_insertion_sort`, `__move_median_to_first`,
//! `__unguarded_partition_pivot`, `__heap_select` + `__sort_heap`,
//! `__introselect`, `__adjust_heap`, `__push_heap`).
//!
//! **Why this exists (issue #81).** Upstream's manylinux torch 2.13.0 wheel
//! (aarch64 and x86_64) is built by GCC 13.3.1 (`readelf -p .comment
//! libtorch_cpu.so`: "GCC: (GNU) 13.3.1 20240611 (Red Hat 13.3.1-2)"), so its
//! cpu `sort(stable=False)` and `topk` order ties the way these header
//! templates leave them. `libcxx_sort.rs` is the same thing for libc++ (Apple,
//! Android). Upstream calls both runtimes identically: a comparator over
//! `(value, index)` pairs that reads the value only, NaN greatest; only the
//! algorithm differs. Measured against `tests/ops/sortties_oracle_linux.json`.
//!
//! `less` is the strict-weak "comes before" comparator. Positions are slice
//! indices.

type Cmp<'a, T> = &'a mut dyn FnMut(&T, &T) -> bool;

const S_THRESHOLD: usize = 16;

/// `std::__lg`: floor(log2(n)), n > 0.
fn lg(n: usize) -> usize {
    (usize::BITS - 1 - n.leading_zeros()) as usize
}

/// `__move_median_to_first(result, a, b, c)`
fn move_median_to_first<T: Copy>(v: &mut [T], r: usize, a: usize, b: usize, c: usize, cmp: Cmp<T>) {
    if cmp(&v[a], &v[b]) {
        if cmp(&v[b], &v[c]) {
            v.swap(r, b);
        } else if cmp(&v[a], &v[c]) {
            v.swap(r, c);
        } else {
            v.swap(r, a);
        }
    } else if cmp(&v[a], &v[c]) {
        v.swap(r, a);
    } else if cmp(&v[b], &v[c]) {
        v.swap(r, c);
    } else {
        v.swap(r, b);
    }
}

/// `__unguarded_partition(first, last, pivot)`
fn unguarded_partition<T: Copy>(
    v: &mut [T],
    mut first: usize,
    mut last: usize,
    pivot: usize,
    cmp: Cmp<T>,
) -> usize {
    loop {
        while cmp(&v[first], &v[pivot]) {
            first += 1;
        }
        last -= 1;
        while cmp(&v[pivot], &v[last]) {
            last -= 1;
        }
        if first >= last {
            return first;
        }
        v.swap(first, last);
        first += 1;
    }
}

/// `__unguarded_partition_pivot`
fn unguarded_partition_pivot<T: Copy>(v: &mut [T], first: usize, last: usize, cmp: Cmp<T>) -> usize {
    let mid = first + (last - first) / 2;
    move_median_to_first(v, first, first + 1, mid, last - 1, cmp);
    unguarded_partition(v, first + 1, last, first, cmp)
}

/// `__push_heap(first, hole, top, value)`
fn push_heap<T: Copy>(v: &mut [T], first: usize, mut hole: usize, top: usize, value: T, cmp: Cmp<T>) {
    // `(hole - 1) / 2` on a signed distance; hole == 0 never enters the loop
    // because then hole <= top.
    let mut parent = hole.wrapping_sub(1) / 2;
    while hole > top && cmp(&v[first + parent], &value) {
        v[first + hole] = v[first + parent];
        hole = parent;
        parent = hole.wrapping_sub(1) / 2;
    }
    v[first + hole] = value;
}

/// `__adjust_heap(first, hole, len, value)`
fn adjust_heap<T: Copy>(v: &mut [T], first: usize, mut hole: usize, len: usize, value: T, cmp: Cmp<T>) {
    let top = hole;
    let mut second = hole;
    // `len` >= 1 here, so `(len - 1) / 2` is the signed expression's value.
    while second < (len - 1) / 2 {
        second = 2 * (second + 1);
        if cmp(&v[first + second], &v[first + second - 1]) {
            second -= 1;
        }
        v[first + hole] = v[first + second];
        hole = second;
    }
    // `(len - 2) / 2` with len == 1 is 0 in C++ (truncation toward zero of -1/2).
    if len & 1 == 0 && second == (len - 2) / 2 {
        second = 2 * (second + 1);
        v[first + hole] = v[first + second - 1];
        hole = second - 1;
    }
    push_heap(v, first, hole, top, value, cmp);
}

/// `__pop_heap(first, last, result)`
fn pop_heap<T: Copy>(v: &mut [T], first: usize, last: usize, result: usize, cmp: Cmp<T>) {
    let value = v[result];
    v[result] = v[first];
    adjust_heap(v, first, 0, last - first, value, cmp);
}

/// `__make_heap`
fn make_heap<T: Copy>(v: &mut [T], first: usize, last: usize, cmp: Cmp<T>) {
    if last - first < 2 {
        return;
    }
    let len = last - first;
    let mut parent = (len - 2) / 2;
    loop {
        let value = v[first + parent];
        adjust_heap(v, first, parent, len, value, cmp);
        if parent == 0 {
            return;
        }
        parent -= 1;
    }
}

/// `__sort_heap`
fn sort_heap<T: Copy>(v: &mut [T], first: usize, mut last: usize, cmp: Cmp<T>) {
    while last - first > 1 {
        last -= 1;
        pop_heap(v, first, last, last, cmp);
    }
}

/// `__heap_select(first, middle, last)`
fn heap_select<T: Copy>(v: &mut [T], first: usize, middle: usize, last: usize, cmp: Cmp<T>) {
    make_heap(v, first, middle, cmp);
    let mut i = middle;
    while i < last {
        if cmp(&v[i], &v[first]) {
            pop_heap(v, first, middle, i, cmp);
        }
        i += 1;
    }
}

/// `__partial_sort(first, middle, last)`
fn partial_sort_range<T: Copy>(v: &mut [T], first: usize, middle: usize, last: usize, cmp: Cmp<T>) {
    heap_select(v, first, middle, last, cmp);
    sort_heap(v, first, middle, cmp);
}

/// `__unguarded_linear_insert(last)`
fn unguarded_linear_insert<T: Copy>(v: &mut [T], mut last: usize, cmp: Cmp<T>) {
    let val = v[last];
    let mut next = last - 1;
    while cmp(&val, &v[next]) {
        v[last] = v[next];
        last = next;
        next = next.wrapping_sub(1);
    }
    v[last] = val;
}

/// `__insertion_sort(first, last)`
fn insertion_sort<T: Copy>(v: &mut [T], first: usize, last: usize, cmp: Cmp<T>) {
    if first == last {
        return;
    }
    let mut i = first + 1;
    while i != last {
        if cmp(&v[i], &v[first]) {
            let val = v[i];
            v.copy_within(first..i, first + 1);
            v[first] = val;
        } else {
            unguarded_linear_insert(v, i, cmp);
        }
        i += 1;
    }
}

/// `__introsort_loop(first, last, depth_limit)`
fn introsort_loop<T: Copy>(v: &mut [T], first: usize, mut last: usize, mut depth: usize, cmp: Cmp<T>) {
    while last - first > S_THRESHOLD {
        if depth == 0 {
            partial_sort_range(v, first, last, last, cmp);
            return;
        }
        depth -= 1;
        let cut = unguarded_partition_pivot(v, first, last, cmp);
        introsort_loop(v, cut, last, depth, cmp);
        last = cut;
    }
}

/// `__final_insertion_sort(first, last)`
fn final_insertion_sort<T: Copy>(v: &mut [T], first: usize, last: usize, cmp: Cmp<T>) {
    if last - first > S_THRESHOLD {
        insertion_sort(v, first, first + S_THRESHOLD, cmp);
        for i in first + S_THRESHOLD..last {
            unguarded_linear_insert(v, i, cmp);
        }
    } else {
        insertion_sort(v, first, last, cmp);
    }
}

/// `std::sort(v.begin(), v.end(), less)`.
pub fn sort<T: Copy>(v: &mut [T], less: &mut dyn FnMut(&T, &T) -> bool) {
    let n = v.len();
    if n == 0 {
        return;
    }
    introsort_loop(v, 0, n, lg(n) * 2, less);
    final_insertion_sort(v, 0, n, less);
}

/// `std::partial_sort(v.begin(), v.begin() + middle, v.end(), less)`.
pub fn partial_sort<T: Copy>(v: &mut [T], middle: usize, less: &mut dyn FnMut(&T, &T) -> bool) {
    let n = v.len();
    partial_sort_range(v, 0, middle, n, less);
}

/// `std::nth_element(v.begin(), v.begin() + nth, v.end(), less)` (`__introselect`).
pub fn nth_element<T: Copy>(v: &mut [T], nth: usize, less: &mut dyn FnMut(&T, &T) -> bool) {
    let (mut first, mut last) = (0usize, v.len());
    if first == last || nth == last {
        return;
    }
    let mut depth = lg(last - first) * 2;
    while last - first > 3 {
        if depth == 0 {
            heap_select(v, first, nth + 1, last, less);
            v.swap(first, nth);
            return;
        }
        depth -= 1;
        let cut = unguarded_partition_pivot(v, first, last, less);
        if cut <= nth {
            first = cut;
        } else {
            last = cut;
        }
    }
    insertion_sort(v, first, last, less);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn idx(v: &[(i32, usize)]) -> Vec<usize> {
        v.iter().map(|p| p.1).collect()
    }

    fn pairs(vals: &[i32]) -> Vec<(i32, usize)> {
        vals.iter().copied().zip(0..).collect()
    }

    fn lt(a: &(i32, usize), b: &(i32, usize)) -> bool {
        a.0 < b.0
    }

    /// 17 equal keys: one partition round, then the final insertion sort.
    /// Hand-traced: move_median_to_first swaps v[0] with v[8] (all compares
    /// false, so the `else` arm picks `b` = mid), the partition then swaps
    /// symmetric pairs from both ends around the pivot at 0.
    #[test]
    fn all_equal_17_is_libstdcxx_order() {
        let mut v = pairs(&[1; 17]);
        sort(&mut v, &mut lt);
        assert_eq!(idx(&v), vec![8, 16, 15, 14, 13, 12, 11, 10, 9, 0, 7, 6, 5, 4, 3, 2, 1]);
    }

    /// 16 elements or fewer is a plain insertion sort: stable.
    #[test]
    fn short_rows_are_stable() {
        let mut v = pairs(&[1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]);
        sort(&mut v, &mut lt);
        assert_eq!(idx(&v), vec![1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14]);
    }

    /// partial_sort through the heap: 4 equal keys, keep 2.
    /// make_heap over [0,1] moves 1 to the root (adjust_heap's even-length
    /// branch) and 0 below it; nothing from 2.. compares less; sort_heap pops
    /// the root 1 to the back: 0, 1.
    #[test]
    fn partial_sort_heap_order() {
        let mut v = pairs(&[5, 5, 5, 5]);
        partial_sort(&mut v, 2, &mut lt);
        assert_eq!(idx(&v[..2]), vec![0, 1]);
    }

    /// Short nth_element (len <= 3) is an insertion sort.
    #[test]
    fn nth_element_short_is_insertion() {
        let mut v = pairs(&[2, 1, 1]);
        nth_element(&mut v, 1, &mut lt);
        assert_eq!(idx(&v), vec![1, 2, 0]);
    }

    /// Values are always sorted, whatever the tie order.
    #[test]
    fn sorts_values() {
        let vals: Vec<i32> = (0..300).map(|i| (i * 7919) % 13).collect();
        let mut v = pairs(&vals);
        sort(&mut v, &mut lt);
        assert!(v.windows(2).all(|w| w[0].0 <= w[1].0));
        let mut u = pairs(&vals);
        nth_element(&mut u, 150, &mut lt);
        let mut s = vals.clone();
        s.sort();
        assert_eq!(u[150].0, s[150]);
    }
}
