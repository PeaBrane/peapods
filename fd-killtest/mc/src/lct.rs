//! Link-cut trees (Sleator-Tarjan) with lazy reversal: link, cut, connectivity in O(log n) amortized.

const NIL: u32 = u32::MAX;

pub struct Lct {
    ch: Vec<[u32; 2]>,
    p: Vec<u32>,
    rev: Vec<bool>,
    stack: Vec<u32>,
}

impl Lct {
    pub fn new(n: usize) -> Self {
        Lct { ch: vec![[NIL; 2]; n], p: vec![NIL; n], rev: vec![false; n], stack: Vec::with_capacity(64) }
    }

    /// Load a rooted forest directly: every node is its own preferred path, path-parent = parent.
    pub fn load_parents(&mut self, parent: &[u32]) {
        for (x, &q) in parent.iter().enumerate() {
            self.ch[x] = [NIL; 2];
            self.rev[x] = false;
            self.p[x] = q;
        }
    }

    #[inline]
    fn is_root(&self, x: u32) -> bool {
        let p = self.p[x as usize];
        p == NIL || {
            let c = self.ch[p as usize];
            c[0] != x && c[1] != x
        }
    }

    #[inline]
    fn push(&mut self, x: u32) {
        let xi = x as usize;
        if self.rev[xi] {
            self.rev[xi] = false;
            self.ch[xi].swap(0, 1);
            for k in 0..2 {
                let c = self.ch[xi][k];
                if c != NIL {
                    self.rev[c as usize] ^= true;
                }
            }
        }
    }

    #[inline]
    fn rotate(&mut self, x: u32) {
        let y = self.p[x as usize];
        let z = self.p[y as usize];
        let dx = (self.ch[y as usize][1] == x) as usize;
        if !self.is_root(y) {
            let dy = (self.ch[z as usize][1] == y) as usize;
            self.ch[z as usize][dy] = x;
        }
        self.p[x as usize] = z;
        let b = self.ch[x as usize][dx ^ 1];
        self.ch[y as usize][dx] = b;
        if b != NIL {
            self.p[b as usize] = y;
        }
        self.ch[x as usize][dx ^ 1] = y;
        self.p[y as usize] = x;
    }

    fn splay(&mut self, x: u32) {
        let mut st = std::mem::take(&mut self.stack);
        st.clear();
        let mut y = x;
        st.push(y);
        while !self.is_root(y) {
            y = self.p[y as usize];
            st.push(y);
        }
        while let Some(z) = st.pop() {
            self.push(z);
        }
        self.stack = st;
        while !self.is_root(x) {
            let y = self.p[x as usize];
            if !self.is_root(y) {
                let z = self.p[y as usize];
                let zigzig = (self.ch[z as usize][0] == y) == (self.ch[y as usize][0] == x);
                if zigzig {
                    self.rotate(y);
                } else {
                    self.rotate(x);
                }
            }
            self.rotate(x);
        }
    }

    fn access(&mut self, x: u32) {
        let mut last = NIL;
        let mut y = x;
        while y != NIL {
            self.splay(y);
            self.ch[y as usize][1] = last;
            last = y;
            y = self.p[y as usize];
        }
        self.splay(x);
    }

    fn make_root(&mut self, x: u32) {
        self.access(x);
        self.rev[x as usize] ^= true;
    }

    pub fn find_root(&mut self, x: u32) -> u32 {
        self.access(x);
        let mut r = x;
        loop {
            self.push(r);
            let l = self.ch[r as usize][0];
            if l == NIL {
                break;
            }
            r = l;
        }
        self.splay(r);
        r
    }

    #[inline]
    pub fn connected(&mut self, a: u32, b: u32) -> bool {
        a == b || self.find_root(a) == self.find_root(b)
    }

    /// Caller guarantees a and b are in different trees.
    pub fn link(&mut self, a: u32, b: u32) {
        self.make_root(a);
        self.p[a as usize] = b;
    }

    /// Caller guarantees a-b is a tree edge.
    pub fn cut(&mut self, a: u32, b: u32) {
        self.make_root(a);
        self.access(b);
        debug_assert_eq!(self.ch[b as usize][0], a);
        self.ch[b as usize][0] = NIL;
        self.p[a as usize] = NIL;
    }
}
