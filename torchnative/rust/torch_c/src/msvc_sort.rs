//! A transcription of MSVC STL's `std::sort`, `std::partial_sort` and
//! `std::nth_element`, taken from microsoft/STL at commit
//! `b6f01e438b68ee497f9c08313e49d12bb2f0d8e0` (main, fetched 2026-10-25):
//!
//! * `stl/inc/algorithm`: `_ISORT_MAX = 32` (l.596), `_Insertion_sort_unchecked`
//!   (l.8946), `_Med3_unchecked` (l.8970), `_Guess_median_unchecked` (Tukey's
//!   ninther above 40 elements, l.8986), `_Partition_by_pivot_unchecked`
//!   (l.9003, a three-way partition that returns the equal range),
//!   `_Partition_by_median_guess_unchecked` (l.9072), `_Sort_unchecked`
//!   (introsort with an `_Ideal` budget, l.9081), `partial_sort` (l.9822),
//!   `_Swap_med5_unchecked` (l.10091), `_Nth_element_median_of_medians`
//!   (l.10125), `_Nth_element_unchecked` (l.10174), `_Sort_heap_unchecked`
//!   (l.7843).
//! * `stl/inc/__msvc_heap_algorithms.hpp`: `_Push_heap_by_index`,
//!   `_Pop_heap_hole_by_index`, `_Pop_heap_hole_unchecked`,
//!   `_Pop_heap_unchecked`, `_Make_heap_unchecked` (Floyd bottom-up, l.121).
//!
//! **Why this exists (issue #81, MSVC half).** Upstream's Windows torch wheel
//! is built with MSVC, so its cpu `sort(stable=False)` and `topk` order ties
//! the way MSVC's STL leaves them. `libcxx_sort.rs` and `libstdcxx_sort.rs`
//! are the same thing for the other runtimes, with the same API. Upstream
//! calls all three identically: a comparator over `(value, index)` pairs that
//! reads the value only, NaN greatest; only the algorithm differs.
//!
//! **Not yet measured against upstream.** There is no Windows oracle in the
//! repository: `tests/ops/sortties_oracle_windows.json` is produced by the
//! `sortties-oracle-windows` workflow on a Windows runner. Which MSVC toolset
//! built the torch 2.13.0 win_amd64 wheel is not recorded in this repository
//! and could not be determined offline. The part of the STL that changed
//! recently is `nth_element`: microsoft/STL #5100 (2025-04-23) added the
//! worst-case linear fallback (`MEDIAN_OF_MEDIANS` below). A toolset built
//! before it never takes that branch. If the oracle disagrees on topk cells
//! only, flip that constant.
//!
//! `less` is the strict-weak "comes before" comparator. Positions are slice
//! indices; differences are `isize` where the C++ uses a signed distance.

type Cmp<'a, T> = &'a mut dyn FnMut(&T, &T) -> bool;

const ISORT_MAX: isize = 32;

/// Whether `nth_element` falls back to median-of-medians once its work budget
/// is spent (STL #5100, 2025-04-23). See the module note.
const MEDIAN_OF_MEDIANS: bool = true;

/// `_Push_heap_by_index(first, hole, top, val)`
fn push_heap_by_index<T: Copy>(
    v: &mut [T],
    first: usize,
    mut hole: isize,
    top: isize,
    val: T,
    cmp: Cmp<T>,
) {
    let mut idx = (hole - 1) >> 1;
    while top < hole && cmp(&v[first + idx as usize], &val) {
        v[first + hole as usize] = v[first + idx as usize];
        hole = idx;
        idx = (hole - 1) >> 1;
    }
    v[first + hole as usize] = val;
}

/// `_Pop_heap_hole_by_index(first, hole, bottom, val)`
fn pop_heap_hole_by_index<T: Copy>(
    v: &mut [T],
    first: usize,
    mut hole: isize,
    bottom: isize,
    val: T,
    cmp: Cmp<T>,
) {
    let top = hole;
    let mut idx = hole;
    let max_non_leaf = (bottom - 1) >> 1;
    while idx < max_non_leaf {
        idx = 2 * idx + 2;
        if cmp(&v[first + idx as usize], &v[first + (idx - 1) as usize]) {
            idx -= 1;
        }
        v[first + hole as usize] = v[first + idx as usize];
        hole = idx;
    }
    if idx == max_non_leaf && bottom % 2 == 0 {
        v[first + hole as usize] = v[first + (bottom - 1) as usize];
        hole = bottom - 1;
    }
    push_heap_by_index(v, first, hole, top, val, cmp);
}

/// `_Pop_heap_hole_unchecked(first, last, dest, val)`
fn pop_heap_hole_unchecked<T: Copy>(
    v: &mut [T],
    first: usize,
    last: usize,
    dest: usize,
    val: T,
    cmp: Cmp<T>,
) {
    v[dest] = v[first];
    pop_heap_hole_by_index(v, first, 0, (last - first) as isize, val, cmp);
}

/// `_Pop_heap_unchecked(first, last)`
fn pop_heap<T: Copy>(v: &mut [T], first: usize, mut last: usize, cmp: Cmp<T>) {
    if last - first >= 2 {
        last -= 1;
        let val = v[last];
        pop_heap_hole_unchecked(v, first, last, last, val, cmp);
    }
}

/// `_Make_heap_unchecked(first, last)`
fn make_heap<T: Copy>(v: &mut [T], first: usize, last: usize, cmp: Cmp<T>) {
    let bottom = (last - first) as isize;
    let mut hole = bottom >> 1;
    while hole > 0 {
        hole -= 1;
        let val = v[first + hole as usize];
        pop_heap_hole_by_index(v, first, hole, bottom, val, cmp);
    }
}

/// `_Sort_heap_unchecked(first, last)`
fn sort_heap<T: Copy>(v: &mut [T], first: usize, mut last: usize, cmp: Cmp<T>) {
    while last - first >= 2 {
        pop_heap(v, first, last, cmp);
        last -= 1;
    }
}

/// `_Insertion_sort_unchecked(first, last)`
fn insertion_sort<T: Copy>(v: &mut [T], first: usize, last: usize, cmp: Cmp<T>) {
    if first == last {
        return;
    }
    let mut mid = first + 1;
    while mid != last {
        let mut hole = mid;
        let val = v[mid];
        if cmp(&val, &v[first]) {
            v.copy_within(first..mid, first + 1);
            v[first] = val;
        } else {
            let mut prev = hole;
            loop {
                prev -= 1;
                if !cmp(&val, &v[prev]) {
                    break;
                }
                v[hole] = v[prev];
                hole = prev;
            }
            v[hole] = val;
        }
        mid += 1;
    }
}

/// `_Med3_unchecked(first, mid, last)`
fn med3<T: Copy>(v: &mut [T], first: usize, mid: usize, last: usize, cmp: Cmp<T>) {
    if cmp(&v[mid], &v[first]) {
        v.swap(mid, first);
    }
    if cmp(&v[last], &v[mid]) {
        v.swap(last, mid);
        if cmp(&v[mid], &v[first]) {
            v.swap(mid, first);
        }
    }
}

/// `_Guess_median_unchecked(first, mid, last)`; `last` is the last element.
fn guess_median<T: Copy>(v: &mut [T], first: usize, mid: usize, last: usize, cmp: Cmp<T>) {
    let count = last - first;
    if 40 < count {
        let step = (count + 1) >> 3;
        let two_step = step << 1;
        med3(v, first, first + step, first + two_step, cmp);
        med3(v, mid - step, mid, mid + step, cmp);
        med3(v, last - two_step, last - step, last, cmp);
        med3(v, first + step, mid, last - step, cmp);
    } else {
        med3(v, first, mid, last, cmp);
    }
}

/// `_Partition_by_pivot_unchecked(first, pfirst, last)`: returns the range of
/// elements equal to the pivot.
fn partition_by_pivot<T: Copy>(
    v: &mut [T],
    first: usize,
    mut pfirst: usize,
    last: usize,
    cmp: Cmp<T>,
) -> (usize, usize) {
    let mut plast = pfirst + 1;
    while first < pfirst && !cmp(&v[pfirst - 1], &v[pfirst]) && !cmp(&v[pfirst], &v[pfirst - 1]) {
        pfirst -= 1;
    }
    while plast < last && !cmp(&v[plast], &v[pfirst]) && !cmp(&v[pfirst], &v[plast]) {
        plast += 1;
    }
    let mut gfirst = plast;
    let mut glast = pfirst;
    loop {
        while gfirst < last {
            if cmp(&v[pfirst], &v[gfirst]) {
                gfirst += 1;
                continue;
            } else if cmp(&v[gfirst], &v[pfirst]) {
                break;
            } else if plast != gfirst {
                v.swap(plast, gfirst);
                plast += 1;
            } else {
                plast += 1;
            }
            gfirst += 1;
        }
        while first < glast {
            let glast_prev = glast - 1;
            if cmp(&v[glast_prev], &v[pfirst]) {
                glast -= 1;
                continue;
            } else if cmp(&v[pfirst], &v[glast_prev]) {
                break;
            } else {
                pfirst -= 1;
                if pfirst != glast_prev {
                    v.swap(pfirst, glast_prev);
                }
            }
            glast -= 1;
        }
        if glast == first && gfirst == last {
            return (pfirst, plast);
        }
        if glast == first {
            if plast != gfirst {
                v.swap(pfirst, plast);
            }
            plast += 1;
            v.swap(pfirst, gfirst);
            pfirst += 1;
            gfirst += 1;
        } else if gfirst == last {
            glast -= 1;
            pfirst -= 1;
            if glast != pfirst {
                v.swap(glast, pfirst);
            }
            plast -= 1;
            v.swap(pfirst, plast);
        } else {
            glast -= 1;
            v.swap(gfirst, glast);
            gfirst += 1;
        }
    }
}

/// `_Partition_by_median_guess_unchecked(first, last)`
fn partition_by_median_guess<T: Copy>(
    v: &mut [T],
    first: usize,
    last: usize,
    cmp: Cmp<T>,
) -> (usize, usize) {
    let mid = first + ((last - first) >> 1);
    guess_median(v, first, mid, last - 1, cmp);
    partition_by_pivot(v, first, mid, last, cmp)
}

/// `_Sort_unchecked(first, last, ideal)`
fn sort_unchecked<T: Copy>(v: &mut [T], mut first: usize, mut last: usize, mut ideal: isize, cmp: Cmp<T>) {
    loop {
        if (last - first) as isize <= ISORT_MAX {
            insertion_sort(v, first, last, cmp);
            return;
        }
        if ideal <= 0 {
            make_heap(v, first, last, cmp);
            sort_heap(v, first, last, cmp);
            return;
        }
        let mid = partition_by_median_guess(v, first, last, cmp);
        ideal = (ideal >> 1) + (ideal >> 2);
        if mid.0 - first < last - mid.1 {
            sort_unchecked(v, first, mid.0, ideal, cmp);
            first = mid.1;
        } else {
            sort_unchecked(v, mid.1, last, ideal, cmp);
            last = mid.0;
        }
    }
}

/// `_Swap_med5_unchecked(first, dest)`
fn swap_med5<T: Copy>(v: &mut [T], first: usize, dest: usize, cmp: Cmp<T>) {
    let mut guess = first + 1;
    let third = guess + 1;
    med3(v, first, guess, third, cmp);
    let fourth = third + 1;
    let fifth = fourth + 1;
    if cmp(&v[fifth], &v[fourth]) {
        v.swap(fourth, fifth);
    }
    if cmp(&v[fifth], &v[guess]) {
        if cmp(&v[first], &v[fifth]) {
            guess = fifth;
        } else {
            guess = first;
        }
    } else if cmp(&v[guess], &v[fourth]) {
        if cmp(&v[third], &v[fourth]) {
            guess = third;
        } else {
            guess = fourth;
        }
    }
    v.swap(dest, guess);
}

/// `_Nth_element_median_of_medians(first, nth, last)`
fn nth_element_median_of_medians<T: Copy>(
    v: &mut [T],
    mut first: usize,
    nth: usize,
    mut last: usize,
    cmp: Cmp<T>,
) {
    loop {
        let length = (last - first) as isize;
        if length <= ISORT_MAX {
            insertion_sort(v, first, last, cmp);
            return;
        }
        let median_count = (length / 5) as usize;
        let median_last = first + median_count;
        let mut current = first;
        let mut median_dest = first;
        while median_dest != median_last {
            swap_med5(v, current, median_dest, cmp);
            current += 5;
            median_dest += 1;
        }
        let median_mid = first + (median_count >> 1);
        nth_element_median_of_medians(v, first, median_mid, median_last, cmp);
        let median_greater_count = (median_last - median_mid) - 1;
        let median_greater_first = last - median_greater_count;
        // `_Swap_ranges_unchecked(median_mid + 1, median_last, median_greater_first)`
        for k in 0..median_greater_count {
            v.swap(median_mid + 1 + k, median_greater_first + k);
        }
        let middle = first + ((length >> 1) as usize);
        v.swap(median_mid, middle);
        let mid = partition_by_pivot(v, median_mid, middle, median_greater_first, cmp);
        if mid.1 <= nth {
            first = mid.1;
        } else if mid.0 <= nth {
            return;
        } else {
            last = mid.0;
        }
    }
}

/// `std::sort(v.begin(), v.end(), less)`.
pub fn sort<T: Copy>(v: &mut [T], less: &mut dyn FnMut(&T, &T) -> bool) {
    let n = v.len();
    sort_unchecked(v, 0, n, n as isize, less);
}

/// `std::partial_sort(v.begin(), v.begin() + middle, v.end(), less)`.
pub fn partial_sort<T: Copy>(v: &mut [T], middle: usize, less: &mut dyn FnMut(&T, &T) -> bool) {
    let n = v.len();
    if middle == 0 {
        return;
    }
    make_heap(v, 0, middle, less);
    for next in middle..n {
        if less(&v[next], &v[0]) {
            let val = v[next];
            pop_heap_hole_unchecked(v, 0, middle, next, val, less);
        }
    }
    sort_heap(v, 0, middle, less);
}

/// `std::nth_element(v.begin(), v.begin() + nth, v.end(), less)`
/// (`_Nth_element_unchecked`).
pub fn nth_element<T: Copy>(v: &mut [T], nth: usize, less: &mut dyn FnMut(&T, &T) -> bool) {
    let (mut first, mut last) = (0usize, v.len());
    if nth == last {
        return;
    }
    let mut ideal = (last - first) as isize;
    loop {
        let length = (last - first) as isize;
        if length <= ISORT_MAX {
            insertion_sort(v, first, last, less);
            return;
        }
        if MEDIAN_OF_MEDIANS && ideal < (length >> 1) {
            nth_element_median_of_medians(v, first, nth, last, less);
            return;
        }
        let mid = partition_by_median_guess(v, first, last, less);
        if mid.1 <= nth {
            first = mid.1;
        } else if mid.0 <= nth {
            return;
        } else {
            last = mid.0;
        }
        ideal = (ideal >> 1) + (ideal >> 2);
    }
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

    /// 33 equal keys: just past `_ISORT_MAX`. Hand-traced: the guess-median
    /// compares are all false, then `partition_by_pivot` walks `pfirst` down
    /// to 0 and `plast` up to 33 because every element equals the pivot, so
    /// the equal range is the whole row and nothing moves. (libstdc++ scrambles
    /// this row; MSVC's three-way partition leaves it.)
    #[test]
    fn all_equal_33_is_untouched() {
        let mut v = pairs(&[1; 33]);
        sort(&mut v, &mut lt);
        assert_eq!(idx(&v), (0..33).collect::<Vec<_>>());
    }

    /// 32 elements or fewer is a plain insertion sort: stable.
    #[test]
    fn short_rows_are_stable() {
        let vals: Vec<i32> = (0..32).map(|i| i % 2).collect();
        let mut v = pairs(&vals);
        sort(&mut v, &mut lt);
        let mut want: Vec<usize> = (0..32).filter(|i| i % 2 == 0).collect();
        want.extend((0..32).filter(|i| i % 2 == 1));
        assert_eq!(idx(&v), want);
    }

    /// partial_sort through the heap: 4 equal keys, keep 2. make_heap over
    /// [0,1] takes the only-child branch of pop_heap_hole_by_index (element 1
    /// moves to the root, 0 is pushed below it); nothing from 2.. compares
    /// less; sort_heap pops the root to the back: 0, 1.
    #[test]
    fn partial_sort_heap_order() {
        let mut v = pairs(&[5, 5, 5, 5]);
        partial_sort(&mut v, 2, &mut lt);
        assert_eq!(idx(&v[..2]), vec![0, 1]);
    }

    /// Short nth_element (len <= 32) is an insertion sort.
    #[test]
    fn nth_element_short_is_insertion() {
        let mut v = pairs(&[2, 1, 1]);
        nth_element(&mut v, 1, &mut lt);
        assert_eq!(idx(&v), vec![1, 2, 0]);
    }

    /// Hand-traced guess_median on 3 distinct keys [3,1,2]: mid<first swaps
    /// (1,3,2); last<mid swaps (1,2,3); mid<first is false.
    #[test]
    fn med3_orders_three() {
        let mut v = pairs(&[3, 1, 2]);
        med3(&mut v, 0, 1, 2, &mut lt);
        assert_eq!(v.iter().map(|p| p.0).collect::<Vec<_>>(), vec![1, 2, 3]);
    }

    /// Values are always sorted, whatever the tie order, across every path:
    /// insertion, quicksort with ninther, and the heap fallback.
    #[test]
    fn sorts_values() {
        for n in [0usize, 1, 2, 31, 32, 33, 40, 41, 64, 300, 1000, 5000] {
            for m in [1i32, 2, 4, 13, 100000] {
                let vals: Vec<i32> = (0..n as i64).map(|i| ((i * 7919) % 104729) as i32 % m).collect();
                let mut v = pairs(&vals);
                sort(&mut v, &mut lt);
                assert!(v.windows(2).all(|w| w[0].0 <= w[1].0), "n={n} m={m}");
                let mut seen: Vec<usize> = idx(&v);
                seen.sort();
                assert_eq!(seen, (0..n).collect::<Vec<_>>(), "n={n} m={m}");
            }
        }
    }

    /// The heap fallback of `sort_unchecked` (ideal exhausted) sorts too.
    #[test]
    fn heap_fallback_sorts() {
        let vals: Vec<i32> = (0..200).map(|i| (i * 37) % 11).collect();
        let mut v = pairs(&vals);
        sort_unchecked(&mut v, 0, 200, 0, &mut lt);
        assert!(v.windows(2).all(|w| w[0].0 <= w[1].0));
    }

    #[test]
    fn partial_sort_and_nth_element_select() {
        for n in [5usize, 33, 100, 700, 4000] {
            let vals: Vec<i32> = (0..n as i64).map(|i| ((i * 7919) % 104729) as i32 % 13).collect();
            let mut s = vals.clone();
            s.sort();
            for k in [1usize, 2, n / 2, n - 1] {
                let mut u = pairs(&vals);
                nth_element(&mut u, k, &mut lt);
                assert_eq!(u[k].0, s[k], "nth n={n} k={k}");
                assert!(u[..k].iter().all(|p| p.0 <= u[k].0));
                assert!(u[k..].iter().all(|p| p.0 >= u[k].0));
                let mut w = pairs(&vals);
                partial_sort(&mut w, k, &mut lt);
                assert_eq!(w[..k].iter().map(|p| p.0).collect::<Vec<_>>(), s[..k].to_vec(), "ps n={n} k={k}");
            }
        }
    }

    /// The median-of-medians path on its own selects correctly.
    #[test]
    fn median_of_medians_selects() {
        let vals: Vec<i32> = (0..1000i64).map(|i| ((i * 7919) % 104729) as i32 % 50).collect();
        let mut s = vals.clone();
        s.sort();
        for k in [0usize, 10, 500, 999] {
            let mut u = pairs(&vals);
            let n = u.len();
            nth_element_median_of_medians(&mut u, 0, k, n, &mut lt);
            assert_eq!(u[k].0, s[k]);
            assert!(u[..k].iter().all(|p| p.0 <= u[k].0));
            assert!(u[k + 1..].iter().all(|p| p.0 >= u[k].0));
        }
    }
}
