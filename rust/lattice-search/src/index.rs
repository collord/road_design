/// Compact ids for the active cells of an `nrows x ncols` grid, in row-major order.
///
/// State ids are `cell_id * n_theta + h`. Because cell ids increase with
/// `row * ncols + col`, state ids sort the same way dense ids would.
#[derive(Clone, Debug)]
pub struct CellIndex {
    nrows: usize,
    ncols: usize,
    /// Compact id per grid cell, `u32::MAX` where inactive.
    id: Vec<u32>,
    /// Flat grid index (`row * ncols + col`) per compact id.
    cells: Vec<u32>,
}

const NONE: u32 = u32::MAX;

impl CellIndex {
    /// Every cell of the grid.
    pub fn full(nrows: usize, ncols: usize) -> Self {
        let n = nrows * ncols;
        assert!(n < NONE as usize, "grid too large for 32-bit cell ids");
        CellIndex { nrows, ncols, id: (0..n as u32).collect(), cells: (0..n as u32).collect() }
    }

    /// The cells where `mask` is true (row-major, `nrows * ncols` long).
    pub fn from_mask(nrows: usize, ncols: usize, mask: &[bool]) -> Self {
        assert_eq!(mask.len(), nrows * ncols, "mask length must be nrows * ncols");
        assert!(mask.len() < NONE as usize, "grid too large for 32-bit cell ids");
        let mut id = vec![NONE; mask.len()];
        let mut cells = Vec::with_capacity(mask.iter().filter(|&&m| m).count());
        for (k, &m) in mask.iter().enumerate() {
            if m {
                id[k] = cells.len() as u32;
                cells.push(k as u32);
            }
        }
        CellIndex { nrows, ncols, id, cells }
    }

    pub fn nrows(&self) -> usize {
        self.nrows
    }

    pub fn ncols(&self) -> usize {
        self.ncols
    }

    /// Number of active cells.
    pub fn len(&self) -> usize {
        self.cells.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// Compact id of cell (r, c), or None if it is outside the grid or inactive.
    #[inline(always)]
    pub fn get(&self, r: i64, c: i64) -> Option<u32> {
        if r < 0 || c < 0 || r >= self.nrows as i64 || c >= self.ncols as i64 {
            return None;
        }
        let k = self.id[r as usize * self.ncols + c as usize];
        if k == NONE {
            None
        } else {
            Some(k)
        }
    }

    /// (row, col) of a compact cell id.
    #[inline(always)]
    pub fn cell(&self, id: u32) -> (i32, i32) {
        let k = self.cells[id as usize] as usize;
        ((k / self.ncols) as i32, (k % self.ncols) as i32)
    }

    /// Flat grid index (`row * ncols + col`) of each active cell, in id order.
    pub fn cells(&self) -> &[u32] {
        &self.cells
    }

    /// State id of (r, c, h) with `n_theta` headings.
    #[inline(always)]
    pub fn state(&self, r: i64, c: i64, h: u16, n_theta: usize) -> Option<u64> {
        self.get(r, c).map(|k| k as u64 * n_theta as u64 + h as u64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mask_ids_are_row_major() {
        let mask = [true, false, true, false, true, true];
        let ix = CellIndex::from_mask(2, 3, &mask);
        assert_eq!(ix.len(), 4);
        assert_eq!(ix.get(0, 0), Some(0));
        assert_eq!(ix.get(0, 1), None);
        assert_eq!(ix.get(0, 2), Some(1));
        assert_eq!(ix.get(1, 1), Some(2));
        assert_eq!(ix.cell(3), (1, 2));
        assert_eq!(ix.get(-1, 0), None);
        assert_eq!(ix.get(2, 0), None);
    }
}
