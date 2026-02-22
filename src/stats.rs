/// Running statistics accumulator for per-temperature observables.
pub struct Statistics {
    pub count: usize,
    pub aggregate: Vec<f64>,
    pub power: u32,
}

impl Statistics {
    pub fn new(n_temps: usize, power: u32) -> Self {
        Self {
            count: 0,
            aggregate: vec![0.0; n_temps],
            power,
        }
    }

    pub fn update(&mut self, values: &[f32]) {
        self.count += 1;
        for (agg, &v) in self.aggregate.iter_mut().zip(values.iter()) {
            let v = v as f64;
            *agg += if self.power == 1 {
                v
            } else {
                v.powi(self.power as i32)
            };
        }
    }

    pub fn average(&self) -> Vec<f64> {
        if self.count == 0 {
            return self.aggregate.clone();
        }
        let c = self.count as f64;
        self.aggregate.iter().map(|&a| a / c).collect()
    }
}
