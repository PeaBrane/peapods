use validator::{Validate, ValidationError};

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SweepMode {
    Metropolis,
    Gibbs,
}

impl TryFrom<&str> for SweepMode {
    type Error = String;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s {
            "metropolis" => Ok(Self::Metropolis),
            "gibbs" => Ok(Self::Gibbs),
            _ => Err(format!(
                "unknown sweep_mode '{s}', expected 'metropolis' or 'gibbs'"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ClusterMode {
    Wolff,
    Sw,
}

impl TryFrom<&str> for ClusterMode {
    type Error = String;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s {
            "wolff" => Ok(Self::Wolff),
            "sw" => Ok(Self::Sw),
            _ => Err(format!(
                "unknown cluster_mode '{s}', expected 'wolff' or 'sw'"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OverlapClusterBuildMode {
    Houdayer,
    Jorg,
    Cmr(usize),
}

impl OverlapClusterBuildMode {
    pub fn group_size(&self) -> usize {
        match self {
            Self::Cmr(n) => *n,
            _ => 2,
        }
    }
}

impl TryFrom<&str> for OverlapClusterBuildMode {
    type Error = String;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s {
            "houdayer" => Ok(Self::Houdayer),
            "jorg" => Ok(Self::Jorg),
            "cmr" | "cmr2" => Ok(Self::Cmr(2)),
            _ if s.starts_with("cmr") => {
                let n: usize = s[3..].parse().map_err(|_| {
                    format!("invalid CMR group size in '{s}', expected 'cmrN' with integer N >= 2")
                })?;
                if n < 2 {
                    return Err(format!("CMR group size must be >= 2, got {n}"));
                }
                Ok(Self::Cmr(n))
            }
            _ => Err(format!(
                "unknown overlap_cluster_build_mode '{s}', expected 'houdayer', 'jorg', or 'cmrN'"
            )),
        }
    }
}

#[derive(Debug)]
pub struct ClusterConfig {
    pub interval: usize,
    pub mode: ClusterMode,
    pub collect_csd: bool,
}

#[derive(Debug)]
pub struct OverlapClusterConfig {
    pub interval: usize,
    pub mode: OverlapClusterBuildMode,
    pub cluster_mode: ClusterMode,
    pub collect_csd: bool,
    pub collect_top_clusters: bool,
}

fn validate_sim_config(cfg: &SimConfig) -> Result<(), ValidationError> {
    if cfg.n_sweeps < 1 {
        return Err(ValidationError::new("n_sweeps must be >= 1"));
    }
    if cfg.warmup_sweeps > cfg.n_sweeps {
        return Err(ValidationError::new("warmup_sweeps must be <= n_sweeps"));
    }
    if let Some(ref c) = cfg.cluster_update {
        if c.interval < 1 {
            return Err(ValidationError::new("cluster_update interval must be >= 1"));
        }
    }
    if let Some(ref h) = cfg.overlap_cluster {
        if h.interval < 1 {
            return Err(ValidationError::new(
                "overlap_cluster interval must be >= 1",
            ));
        }
    }
    Ok(())
}

#[derive(Debug, Validate)]
#[validate(schema(function = "validate_sim_config"))]
pub struct SimConfig {
    pub n_sweeps: usize,
    pub warmup_sweeps: usize,
    pub sweep_mode: SweepMode,
    pub cluster_update: Option<ClusterConfig>,
    pub pt_interval: Option<usize>,
    pub overlap_cluster: Option<OverlapClusterConfig>,
    pub autocorrelation_max_lag: Option<usize>,
    pub sequential: bool,
    pub equilibration_diagnostic: bool,
}
