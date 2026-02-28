pub mod autocorrelation;
pub mod equilibration;
pub mod results;
mod stats;

pub use autocorrelation::{sokal_tau, AutocorrAccum};
pub use equilibration::{EquilCheckpoint, EquilDiagnosticAccum};
pub use results::{ClusterStats, Diagnostics, SweepResult, OVERLAP_HIST_BINS};
pub use stats::Statistics;
