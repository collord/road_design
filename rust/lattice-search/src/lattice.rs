/// One move of the lattice: from heading `h0` at a cell to heading `h1` at
/// the cell `(row + dr, col + dc)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Action {
    pub h0: u16,
    pub h1: u16,
    pub dr: i32,
    pub dc: i32,
}

/// The actions of a lattice, grouped by the heading they leave from.
#[derive(Clone, Debug)]
pub struct Lattice {
    n_theta: usize,
    actions: Vec<Action>,
    /// Actions leaving heading h: `out[start[h]..start[h + 1]]`, in expansion order.
    start: Vec<usize>,
    out: Vec<u32>,
}

impl Lattice {
    /// Actions leaving each heading in action-index order.
    pub fn new(n_theta: usize, actions: Vec<Action>) -> Self {
        let mut by_heading = vec![Vec::new(); n_theta];
        for (i, a) in actions.iter().enumerate() {
            by_heading[a.h0 as usize].push(i as u32);
        }
        Self::with_order(n_theta, actions, by_heading)
    }

    /// Actions leaving each heading in a given order (the order states are
    /// relaxed in, which decides between equal-cost parents).
    pub fn with_order(n_theta: usize, actions: Vec<Action>, by_heading: Vec<Vec<u32>>) -> Self {
        assert!(n_theta > 0 && n_theta <= u16::MAX as usize);
        assert!(actions.len() <= i16::MAX as usize, "at most 32767 actions (parents are stored as i16)");
        assert_eq!(by_heading.len(), n_theta);
        let mut start = Vec::with_capacity(n_theta + 1);
        let mut out = Vec::new();
        for (h, list) in by_heading.iter().enumerate() {
            start.push(out.len());
            for &a in list {
                let act = actions.get(a as usize).expect("action index out of range");
                assert_eq!(act.h0 as usize, h, "action {a} listed under heading {h} leaves heading {}", act.h0);
                assert!((act.h1 as usize) < n_theta);
                out.push(a);
            }
        }
        start.push(out.len());
        Lattice { n_theta, actions, start, out }
    }

    pub fn n_theta(&self) -> usize {
        self.n_theta
    }

    pub fn actions(&self) -> &[Action] {
        &self.actions
    }

    #[inline(always)]
    pub fn action(&self, a: usize) -> &Action {
        &self.actions[a]
    }

    /// Actions leaving heading h, in expansion order.
    #[inline(always)]
    pub fn leaving(&self, h: usize) -> &[u32] {
        &self.out[self.start[h]..self.start[h + 1]]
    }
}
