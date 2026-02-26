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
pub enum HoudayerMode {
    Houdayer,
    Jorg,
    Cmr,
}

impl TryFrom<&str> for HoudayerMode {
    type Error = String;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s {
            "houdayer" => Ok(Self::Houdayer),
            "jorg" => Ok(Self::Jorg),
            "cmr" => Ok(Self::Cmr),
            _ => Err(format!(
                "unknown houdayer_mode '{s}', expected 'houdayer', 'jorg', or 'cmr'"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OverlapUpdateMode {
    Swap,
    Free,
}

impl TryFrom<&str> for OverlapUpdateMode {
    type Error = String;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s {
            "swap" => Ok(Self::Swap),
            "free" => Ok(Self::Free),
            _ => Err(format!(
                "unknown overlap_update_mode '{s}', expected 'swap' or 'free'"
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

fn validate_houdayer_config(cfg: &HoudayerConfig) -> Result<(), ValidationError> {
    if cfg.update_mode == OverlapUpdateMode::Free && cfg.mode != HoudayerMode::Cmr {
        return Err(ValidationError::new(
            "overlap_update_mode 'free' requires houdayer_mode 'cmr'",
        ));
    }
    Ok(())
}

#[derive(Debug, Validate)]
#[validate(schema(function = "validate_houdayer_config"))]
pub struct HoudayerConfig {
    pub interval: usize,
    pub mode: HoudayerMode,
    pub cluster_mode: ClusterMode,
    pub update_mode: OverlapUpdateMode,
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
    if let Some(ref h) = cfg.houdayer {
        if h.interval < 1 {
            return Err(ValidationError::new("houdayer interval must be >= 1"));
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
    #[validate]
    pub houdayer: Option<HoudayerConfig>,
}
