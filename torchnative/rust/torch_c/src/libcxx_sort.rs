//! A transcription of libc++'s `std::sort`, `std::partial_sort` and
//! `std::nth_element` (the non-branchless path, which is the one a tuple or
//! pair element type takes), taken from the LLVM 19 era headers
//! (`__algorithm/sort.h`, `partial_sort.h`, `nth_element.h`, `make_heap.h`,
//! `sift_down.h`, `pop_heap.h`, `push_heap.h`, `sort_heap.h`).
//!
//! **Why this exists (issue #33).** Upstream's CPU `sort(stable=False)` and
//! `topk` call those three algorithms on a `(value, index)` pair, so the order
//! of indices among equal values is whatever libc++'s introsort leaves. It is
//! not stable (for 24 or more elements) and it is not a promise upstream makes,
//! but it is what `TopPLogitsWarper` and `multinomial` see, so the cpu path
//! reproduces it instead of answering with a stable order. The tie order is the
//! one libc++ gives; a libstdc++ build of upstream (Linux) orders ties
//! differently, which this module does not model.
//!
//! `less` is the strict-weak "comes before" comparator, exactly as the C++
//! algorithms take it. Positions are slice indices; no iterator proxies.

#![allow(clippy::many_single_char_names)]

type Cmp<'a, T> = &'a mut dyn FnMut(&T, &T) -> bool;

/// `__sort3`: returns whether it swapped anything.
fn sort3<T: Copy>(v: &mut [T], x: usize, y: usize, z: usize, c: Cmp<T>) -> bool {
    if !c(&v[y], &v[x]) {
        if !c(&v[z], &v[y]) {
            return false;
        }
        v.swap(y, z);
        if c(&v[y], &v[x]) {
            v.swap(x, y);
        }
        return true;
    }
    if c(&v[z], &v[y]) {
        v.swap(x, z);
        return true;
    }
    v.swap(x, y);
    if c(&v[z], &v[y]) {
        v.swap(y, z);
    }
    true
}

fn sort4<T: Copy>(v: &mut [T], x1: usize, x2: usize, x3: usize, x4: usize, c: Cmp<T>) {
    sort3(v, x1, x2, x3, c);
    if c(&v[x4], &v[x3]) {
        v.swap(x3, x4);
        if c(&v[x3], &v[x2]) {
            v.swap(x2, x3);
            if c(&v[x2], &v[x1]) {
                v.swap(x1, x2);
            }
        }
    }
}

fn sort5<T: Copy>(
    v: &mut [T],
    x1: usize,
    x2: usize,
    x3: usize,
    x4: usize,
    x5: usize,
    c: Cmp<T>,
) {
    sort4(v, x1, x2, x3, x4, c);
    if c(&v[x5], &v[x4]) {
        v.swap(x4, x5);
        if c(&v[x4], &v[x3]) {
            v.swap(x3, x4);
            if c(&v[x3], &v[x2]) {
                v.swap(x2, x3);
                if c(&v[x2], &v[x1]) {
                    v.swap(x1, x2);
                }
            }
        }
    }
}

/// `__selection_sort` over `[first, last)`, `last - first >= 1`.
fn selection_sort<T: Copy>(v: &mut [T], first: usize, last: usize, c: Cmp<T>) {
    let lm1 = last - 1;
    let mut f = first;
    while f != lm1 {
        // `__min_element`: the first smallest.
        let mut i = f;
        let mut j = f + 1;
        while j != last {
            if c(&v[j], &v[i]) {
                i = j;
            }
            j += 1;
        }
        if i != f {
            v.swap(f, i);
        }
        f += 1;
    }
}

/// `__insertion_sort` (guarded).
fn insertion_sort<T: Copy>(v: &mut [T], first: usize, last: usize, c: Cmp<T>) {
    if first == last {
        return;
    }
    let mut i = first + 1;
    while i != last {
        let mut j = i - 1;
        if c(&v[i], &v[j]) {
            let t = v[i];
            let mut k = j;
            j = i;
            loop {
                v[j] = v[k];
                j = k;
                if j == first {
                    break;
                }
                k -= 1;
                if !c(&t, &v[k]) {
                    break;
                }
            }
            v[j] = t;
        }
        i += 1;
    }
}

/// `__insertion_sort_unguarded`: assumes `v[first - 1]` is not greater than any element.
fn insertion_sort_unguarded<T: Copy>(v: &mut [T], first: usize, last: usize, c: Cmp<T>) {
    if first == last {
        return;
    }
    let mut i = first + 1;
    while i != last {
        let mut j = i - 1;
        if c(&v[i], &v[j]) {
            let t = v[i];
            let mut k = j;
            j = i;
            loop {
                v[j] = v[k];
                j = k;
                k -= 1;
                if !c(&t, &v[k]) {
                    break;
                }
            }
            v[j] = t;
        }
        i += 1;
    }
}

/// `__insertion_sort_incomplete`: true when it finished sorting.
fn insertion_sort_incomplete<T: Copy>(v: &mut [T], first: usize, last: usize, c: Cmp<T>) -> bool {
    match last - first {
        0 | 1 => return true,
        2 => {
            if c(&v[last - 1], &v[first]) {
                v.swap(first, last - 1);
            }
            return true;
        }
        3 => {
            sort3(v, first, first + 1, last - 1, c);
            return true;
        }
        4 => {
            sort4(v, first, first + 1, first + 2, last - 1, c);
            return true;
        }
        5 => {
            sort5(v, first, first + 1, first + 2, first + 3, last - 1, c);
            return true;
        }
        _ => {}
    }
    let mut j = first + 2;
    sort3(v, first, first + 1, j, c);
    const LIMIT: u32 = 8;
    let mut count = 0u32;
    let mut i = j + 1;
    while i != last {
        if c(&v[i], &v[j]) {
            let t = v[i];
            let mut k = j;
            j = i;
            loop {
                v[j] = v[k];
                j = k;
                if j == first {
                    break;
                }
                k -= 1;
                if !c(&t, &v[k]) {
                    break;
                }
            }
            v[j] = t;
            count += 1;
            if count == LIMIT {
                i += 1;
                return i == last;
            }
        }
        j = i;
        i += 1;
    }
    true
}

/// `__partition_with_equals_on_right`: returns (pivot position, already partitioned).
fn partition_eq_right<T: Copy>(
    v: &mut [T],
    mut first: usize,
    mut last: usize,
    c: Cmp<T>,
) -> (usize, bool) {
    let begin = first;
    let pivot = v[first];
    loop {
        first += 1;
        if !c(&v[first], &pivot) {
            break;
        }
    }
    if begin == first - 1 {
        while first < last {
            last -= 1;
            if c(&v[last], &pivot) {
                break;
            }
        }
    } else {
        loop {
            last -= 1;
            if c(&v[last], &pivot) {
                break;
            }
        }
    }
    let already = first >= last;
    while first < last {
        v.swap(first, last);
        loop {
            first += 1;
            if !c(&v[first], &pivot) {
                break;
            }
        }
        loop {
            last -= 1;
            if c(&v[last], &pivot) {
                break;
            }
        }
    }
    let pivot_pos = first - 1;
    if begin != pivot_pos {
        v[begin] = v[pivot_pos];
    }
    v[pivot_pos] = pivot;
    (pivot_pos, already)
}

/// `__partition_with_equals_on_left`: returns the position after the pivot.
fn partition_eq_left<T: Copy>(v: &mut [T], mut first: usize, mut last: usize, c: Cmp<T>) -> usize {
    let begin = first;
    let pivot = v[first];
    if c(&pivot, &v[last - 1]) {
        loop {
            first += 1;
            if c(&pivot, &v[first]) {
                break;
            }
        }
    } else {
        loop {
            first += 1;
            if !(first < last && !c(&pivot, &v[first])) {
                break;
            }
        }
    }
    if first < last {
        loop {
            last -= 1;
            if !c(&pivot, &v[last]) {
                break;
            }
        }
    }
    while first < last {
        v.swap(first, last);
        loop {
            first += 1;
            if c(&pivot, &v[first]) {
                break;
            }
        }
        loop {
            last -= 1;
            if !c(&pivot, &v[last]) {
                break;
            }
        }
    }
    let pivot_pos = first - 1;
    if begin != pivot_pos {
        v[begin] = v[pivot_pos];
    }
    v[pivot_pos] = pivot;
    first
}

fn sift_down<T: Copy>(v: &mut [T], first: usize, c: Cmp<T>, len: usize, start: usize) {
    let mut start = start;
    let mut child = start - first;
    if len < 2 || (len - 2) / 2 < child {
        return;
    }
    child = 2 * child + 1;
    let mut child_i = first + child;
    if child + 1 < len && c(&v[child_i], &v[child_i + 1]) {
        child_i += 1;
        child += 1;
    }
    if c(&v[child_i], &v[start]) {
        return;
    }
    let top = v[start];
    loop {
        v[start] = v[child_i];
        start = child_i;
        if (len - 2) / 2 < child {
            break;
        }
        child = 2 * child + 1;
        child_i = first + child;
        if child + 1 < len && c(&v[child_i], &v[child_i + 1]) {
            child_i += 1;
            child += 1;
        }
        if c(&v[child_i], &top) {
            break;
        }
    }
    v[start] = top;
}

fn make_heap<T: Copy>(v: &mut [T], first: usize, last: usize, c: Cmp<T>) {
    let n = last - first;
    if n > 1 {
        let mut start = (n as isize - 2) / 2;
        while start >= 0 {
            sift_down(v, first, c, n, first + start as usize);
            start -= 1;
        }
    }
}

fn sift_up<T: Copy>(v: &mut [T], first: usize, last: usize, c: Cmp<T>, len: usize) {
    if len > 1 {
        let mut len = (len - 2) / 2;
        let mut ptr = first + len;
        let mut last = last - 1;
        if c(&v[ptr], &v[last]) {
            let t = v[last];
            loop {
                v[last] = v[ptr];
                last = ptr;
                if len == 0 {
                    break;
                }
                len = (len - 1) / 2;
                ptr = first + len;
                if !c(&v[ptr], &t) {
                    break;
                }
            }
            v[last] = t;
        }
    }
}

/// `__floyd_sift_down`: returns the hole.
fn floyd_sift_down<T: Copy>(v: &mut [T], first: usize, c: Cmp<T>, len: usize) -> usize {
    let mut hole = first;
    let mut child_i = first;
    let mut child = 0usize;
    loop {
        child_i += child + 1;
        child = 2 * child + 1;
        if child + 1 < len && c(&v[child_i], &v[child_i + 1]) {
            child_i += 1;
            child += 1;
        }
        v[hole] = v[child_i];
        hole = child_i;
        if child > (len - 2) / 2 {
            return hole;
        }
    }
}

fn pop_heap<T: Copy>(v: &mut [T], first: usize, last: usize, c: Cmp<T>, len: usize) {
    if len > 1 {
        let top = v[first];
        let mut hole = floyd_sift_down(v, first, c, len);
        let last = last - 1;
        if hole == last {
            v[hole] = top;
        } else {
            v[hole] = v[last];
            hole += 1;
            v[last] = top;
            sift_up(v, first, hole, c, hole - first);
        }
    }
}

fn sort_heap<T: Copy>(v: &mut [T], first: usize, last: usize, c: Cmp<T>) {
    let mut last = last;
    let mut n = last - first;
    while n > 1 {
        pop_heap(v, first, last, c, n);
        last -= 1;
        n -= 1;
    }
}

/// `std::partial_sort(first, middle, last)` over `v[first..last]`.
fn partial_sort_range<T: Copy>(v: &mut [T], first: usize, middle: usize, last: usize, c: Cmp<T>) {
    if first == middle {
        return;
    }
    make_heap(v, first, middle, c);
    let len = middle - first;
    let mut i = middle;
    while i != last {
        if c(&v[i], &v[first]) {
            v.swap(i, first);
            sift_down(v, first, c, len, first);
        }
        i += 1;
    }
    sort_heap(v, first, middle, c);
}

/// `std::partial_sort(v.begin(), v.begin() + middle, v.end(), less)`.
pub fn partial_sort<T: Copy>(v: &mut [T], middle: usize, less: &mut dyn FnMut(&T, &T) -> bool) {
    let n = v.len();
    partial_sort_range(v, 0, middle, n, less);
}

fn introsort<T: Copy>(
    v: &mut [T],
    mut first: usize,
    mut last: usize,
    c: Cmp<T>,
    mut depth: isize,
    mut leftmost: bool,
) {
    const LIMIT: usize = 24;
    const NINTHER: usize = 128;
    loop {
        let len = last - first;
        match len {
            0 | 1 => return,
            2 => {
                last -= 1;
                if c(&v[last], &v[first]) {
                    v.swap(first, last);
                }
                return;
            }
            3 => {
                sort3(v, first, first + 1, last - 1, c);
                return;
            }
            4 => {
                sort4(v, first, first + 1, first + 2, last - 1, c);
                return;
            }
            5 => {
                sort5(v, first, first + 1, first + 2, first + 3, last - 1, c);
                return;
            }
            _ => {}
        }
        if len < LIMIT {
            if leftmost {
                insertion_sort(v, first, last, c);
            } else {
                insertion_sort_unguarded(v, first, last, c);
            }
            return;
        }
        if depth == 0 {
            partial_sort_range(v, first, last, last, c);
            return;
        }
        depth -= 1;
        {
            let half = len / 2;
            if len > NINTHER {
                sort3(v, first, first + half, last - 1, c);
                sort3(v, first + 1, first + (half - 1), last - 2, c);
                sort3(v, first + 2, first + (half + 1), last - 3, c);
                sort3(v, first + (half - 1), first + half, first + (half + 1), c);
                v.swap(first, first + half);
            } else {
                sort3(v, first + half, first, last - 1, c);
            }
        }
        if !leftmost && !c(&v[first - 1], &v[first]) {
            first = partition_eq_left(v, first, last, c);
            continue;
        }
        let (mut i, already) = partition_eq_right(v, first, last, c);
        if already {
            let fs = insertion_sort_incomplete(v, first, i, c);
            if insertion_sort_incomplete(v, i + 1, last, c) {
                if fs {
                    return;
                }
                last = i;
                continue;
            } else if fs {
                i += 1;
                first = i;
                continue;
            }
        }
        introsort(v, first, i, c, depth, leftmost);
        leftmost = false;
        i += 1;
        first = i;
    }
}

/// `std::sort(v.begin(), v.end(), less)` on a non-arithmetic element type.
pub fn sort<T: Copy>(v: &mut [T], less: &mut dyn FnMut(&T, &T) -> bool) {
    let n = v.len();
    if n == 0 {
        return;
    }
    let depth = 2 * (usize::BITS - 1 - n.leading_zeros()) as isize;
    introsort(v, 0, n, less, depth, true);
}

/// `__nth_element_find_guard`
fn nth_find_guard<T: Copy>(v: &[T], i: &mut usize, j: &mut usize, m: usize, c: Cmp<T>) -> bool {
    loop {
        *j -= 1;
        if *i == *j {
            return false;
        }
        if c(&v[*j], &v[m]) {
            return true;
        }
    }
}

#[allow(unused_assignments)]
fn nth_element_range<T: Copy>(
    v: &mut [T],
    mut first: usize,
    nth: usize,
    mut last: usize,
    c: Cmp<T>,
) {
    const LIMIT: usize = 7;
    loop {
        if nth == last {
            return;
        }
        let len = last - first;
        match len {
            0 | 1 => return,
            2 => {
                last -= 1;
                if c(&v[last], &v[first]) {
                    v.swap(first, last);
                }
                return;
            }
            3 => {
                sort3(v, first, first + 1, last - 1, c);
                return;
            }
            _ => {}
        }
        if len <= LIMIT {
            selection_sort(v, first, last, c);
            return;
        }
        let mut m = first + len / 2;
        let lm1 = last - 1;
        let mut n_swaps = sort3(v, first, m, lm1, c) as u32;
        let mut i = first;
        let mut j = lm1;
        if !c(&v[i], &v[m]) {
            if nth_find_guard(v, &mut i, &mut j, m, c) {
                v.swap(i, j);
                n_swaps += 1;
            } else {
                i += 1;
                j = last;
                j -= 1;
                if !c(&v[first], &v[j]) {
                    loop {
                        if i == j {
                            return;
                        } else if c(&v[first], &v[i]) {
                            v.swap(i, j);
                            n_swaps += 1;
                            i += 1;
                            break;
                        }
                        i += 1;
                    }
                }
                if i == j {
                    return;
                }
                loop {
                    while !c(&v[first], &v[i]) {
                        i += 1;
                    }
                    loop {
                        j -= 1;
                        if !c(&v[first], &v[j]) {
                            break;
                        }
                    }
                    if i >= j {
                        break;
                    }
                    v.swap(i, j);
                    n_swaps += 1;
                    i += 1;
                }
                if nth < i {
                    return;
                }
                first = i;
                continue;
            }
        }
        i += 1;
        if i < j {
            loop {
                while c(&v[i], &v[m]) {
                    i += 1;
                }
                loop {
                    j -= 1;
                    if c(&v[j], &v[m]) {
                        break;
                    }
                }
                if i >= j {
                    break;
                }
                v.swap(i, j);
                n_swaps += 1;
                if m == i {
                    m = j;
                }
                i += 1;
            }
        }
        if i != m && c(&v[m], &v[i]) {
            v.swap(i, m);
            n_swaps += 1;
        }
        if nth == i {
            return;
        }
        if n_swaps == 0 {
            if nth < i {
                j = first;
                m = first;
                let mut sorted = false;
                loop {
                    j += 1;
                    if j == i {
                        sorted = true;
                        break;
                    }
                    if c(&v[j], &v[m]) {
                        break;
                    }
                    m = j;
                }
                if sorted {
                    return;
                }
            } else {
                j = i;
                m = i;
                let mut sorted = false;
                loop {
                    j += 1;
                    if j == last {
                        sorted = true;
                        break;
                    }
                    if c(&v[j], &v[m]) {
                        break;
                    }
                    m = j;
                }
                if sorted {
                    return;
                }
            }
        }
        if nth < i {
            last = i;
        } else {
            i += 1;
            first = i;
        }
    }
}

/// `std::nth_element(v.begin(), v.begin() + nth, v.end(), less)`.
pub fn nth_element<T: Copy>(v: &mut [T], nth: usize, less: &mut dyn FnMut(&T, &T) -> bool) {
    let n = v.len();
    if nth == n {
        return;
    }
    nth_element_range(v, 0, nth, n, less);
}
