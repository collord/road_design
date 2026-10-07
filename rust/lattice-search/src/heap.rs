/// A binary min-heap of (key, id) pairs ordered by key, then id.
///
/// The total order makes the pop sequence independent of how the heap is
/// arranged: equal keys come out lowest id first, every time.
#[derive(Default)]
pub struct MinHeap {
    keys: Vec<f64>,
    ids: Vec<u64>,
}

#[inline(always)]
fn less(k1: f64, i1: u64, k2: f64, i2: u64) -> bool {
    k1 < k2 || (k1 == k2 && i1 < i2)
}

impl MinHeap {
    pub fn with_capacity(n: usize) -> Self {
        MinHeap { keys: Vec::with_capacity(n), ids: Vec::with_capacity(n) }
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn push(&mut self, key: f64, id: u64) {
        let mut j = self.keys.len();
        self.keys.push(key);
        self.ids.push(id);
        while j > 0 {
            let p = (j - 1) >> 1;
            if less(self.keys[j], self.ids[j], self.keys[p], self.ids[p]) {
                self.keys.swap(j, p);
                self.ids.swap(j, p);
                j = p;
            } else {
                break;
            }
        }
    }

    pub fn pop(&mut self) -> Option<(f64, u64)> {
        let n = self.keys.len();
        if n == 0 {
            return None;
        }
        let top = (self.keys[0], self.ids[0]);
        let last_k = self.keys.pop().unwrap();
        let last_i = self.ids.pop().unwrap();
        let n = n - 1;
        if n > 0 {
            self.keys[0] = last_k;
            self.ids[0] = last_i;
            let mut j = 0;
            loop {
                let l = 2 * j + 1;
                if l >= n {
                    break;
                }
                let mut m = l;
                if l + 1 < n && less(self.keys[l + 1], self.ids[l + 1], self.keys[l], self.ids[l]) {
                    m = l + 1;
                }
                if less(self.keys[m], self.ids[m], self.keys[j], self.ids[j]) {
                    self.keys.swap(j, m);
                    self.ids.swap(j, m);
                    j = m;
                } else {
                    break;
                }
            }
        }
        Some(top)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pops_in_key_then_id_order() {
        let mut h = MinHeap::default();
        for (k, i) in [(3.0, 1), (1.0, 9), (1.0, 2), (2.0, 5), (0.5, 7), (1.0, 4)] {
            h.push(k, i);
        }
        let got: Vec<_> = std::iter::from_fn(|| h.pop()).collect();
        assert_eq!(got, vec![(0.5, 7), (1.0, 2), (1.0, 4), (1.0, 9), (2.0, 5), (3.0, 1)]);
    }
}
