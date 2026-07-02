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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClusterAction {
    Update,
    Observe,
}

impl TryFrom<&str> for ClusterAction {
    type Error = String;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s {
            "update" => Ok(Self::Update),
            "observe" => Ok(Self::Observe),
            _ => Err(format!(
                "unknown cluster action '{s}', expected 'update' or 'observe'"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PtSchedule {
    SingleRandomEdge,
    FullLadder,
}

impl TryFrom<&str> for PtSchedule {
    type Error = String;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s {
            "single_random_edge" => Ok(Self::SingleRandomEdge),
            "full_ladder" => Ok(Self::FullLadder),
            _ => Err(format!(
                "unknown pt_schedule '{s}', expected 'single_random_edge' or 'full_ladder'"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AutocorrelationBackend {
    Ring,
    Fft,
}

impl TryFrom<&str> for AutocorrelationBackend {
    type Error = String;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s {
            "ring" => Ok(Self::Ring),
            "fft" => Ok(Self::Fft),
            _ => Err(format!(
                "unknown autocorrelation_backend '{s}', expected 'ring' or 'fft'"
            )),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OverlapClusterBuildMode {
    Houdayer(usize),
    Jorg,
    Cmr,
}

impl OverlapClusterBuildMode {
    pub fn group_size(&self) -> usize {
        match self {
            Self::Houdayer(n) => *n,
            _ => 2,
        }
    }
}

impl TryFrom<&str> for OverlapClusterBuildMode {
    type Error = String;
    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s {
            "houdayer" | "houd2" => Ok(Self::Houdayer(2)),
            "jorg" => Ok(Self::Jorg),
            "cmr" | "cmr2" => Ok(Self::Cmr),
            _ if s.starts_with("houd") => {
                let n: usize = s[4..].parse().map_err(|_| {
                    format!(
                        "invalid Houdayer group size in '{s}', expected 'houdN' with even integer N >= 2"
                    )
                })?;
                if n < 2 || !n.is_multiple_of(2) {
                    return Err(format!(
                        "Houdayer group size must be even and >= 2, got {n}"
                    ));
                }
                if n > 2 {
                    eprintln!(
                        "WARNING: houd{n} (group_size > 2) is experimental and very likely \
                         does not satisfy detailed balance"
                    );
                }
                Ok(Self::Houdayer(n))
            }
            _ => Err(format!(
                "unknown overlap_cluster_build_mode '{s}', expected 'houdayer', 'houdN', 'jorg', or 'cmr'"
            )),
        }
    }
}

#[derive(Debug)]
pub struct ClusterConfig {
    pub interval: usize,
    pub mode: ClusterMode,
    pub action: ClusterAction,
    pub collect_stats: bool,
}

#[derive(Debug)]
pub struct OverlapClusterConfig {
    pub interval: usize,
    pub modes: Vec<OverlapClusterBuildMode>,
    pub cluster_mode: ClusterMode,
    pub action: ClusterAction,
    pub collect_stats: bool,
    pub snapshot_interval: Option<usize>,
}

impl OverlapClusterConfig {
    pub fn max_group_size(&self) -> usize {
        self.modes.iter().map(|m| m.group_size()).max().unwrap_or(2)
    }
}

pub fn parse_overlap_modes(s: &str) -> Result<Vec<OverlapClusterBuildMode>, String> {
    s.split('+')
        .map(|part| OverlapClusterBuildMode::try_from(part.trim()))
        .collect()
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
        if c.action == ClusterAction::Observe && c.mode == ClusterMode::Wolff {
            return Err(ValidationError::new(
                "cluster_action='observe' requires cluster_mode='sw'",
            ));
        }
    }
    if cfg.pt_interval == Some(0) {
        return Err(ValidationError::new("pt_interval must be >= 1"));
    }
    if cfg.autocorrelation_backend == AutocorrelationBackend::Fft
        && cfg.autocorrelation_max_lag.is_none()
    {
        return Err(ValidationError::new(
            "autocorrelation_backend='fft' requires autocorrelation_max_lag",
        ));
    }
    if let Some(ref h) = cfg.overlap_cluster {
        if h.interval < 1 {
            return Err(ValidationError::new(
                "overlap_cluster interval must be >= 1",
            ));
        }
        if let Some(si) = h.snapshot_interval {
            if si < 1 || si % h.interval != 0 {
                return Err(ValidationError::new(
                    "snapshot_interval must be a positive multiple of overlap_cluster interval",
                ));
            }
        }
        if h.modes.is_empty() {
            return Err(ValidationError::new(
                "overlap_cluster modes must not be empty",
            ));
        }
        if h.action == ClusterAction::Observe {
            if h.cluster_mode == ClusterMode::Wolff {
                return Err(ValidationError::new(
                    "overlap_cluster_action='observe' requires overlap_cluster_mode='sw'",
                ));
            }
            if h.snapshot_interval.is_some() {
                return Err(ValidationError::new(
                    "snapshot_interval is not supported with overlap_cluster_action='observe'",
                ));
            }
            if h.modes
                .iter()
                .any(|mode| matches!(mode, OverlapClusterBuildMode::Houdayer(n) if *n > 2))
            {
                return Err(ValidationError::new(
                    "overlap_cluster_action='observe' does not support experimental houdN with N > 2",
                ));
            }
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
    pub pt_schedule: PtSchedule,
    pub overlap_cluster: Option<OverlapClusterConfig>,
    pub autocorrelation_max_lag: Option<usize>,
    pub autocorrelation_backend: AutocorrelationBackend,
    pub sequential: bool,
    pub equilibration_diagnostic: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> SimConfig {
        SimConfig {
            n_sweeps: 1,
            warmup_sweeps: 0,
            sweep_mode: SweepMode::Metropolis,
            cluster_update: None,
            pt_interval: None,
            pt_schedule: PtSchedule::SingleRandomEdge,
            overlap_cluster: None,
            autocorrelation_max_lag: None,
            autocorrelation_backend: AutocorrelationBackend::Ring,
            sequential: true,
            equilibration_diagnostic: false,
        }
    }

    #[test]
    fn rejects_zero_parallel_tempering_interval() {
        let mut config = config();
        config.pt_interval = Some(0);
        assert!(config.validate().is_err());
    }

    #[test]
    fn rejects_empty_overlap_mode_list() {
        let mut config = config();
        config.overlap_cluster = Some(OverlapClusterConfig {
            interval: 1,
            modes: vec![],
            cluster_mode: ClusterMode::Sw,
            action: ClusterAction::Update,
            collect_stats: false,
            snapshot_interval: None,
        });
        assert!(config.validate().is_err());
    }

    #[test]
    fn rejects_unsupported_observe_modes() {
        let mut config = config();
        config.cluster_update = Some(ClusterConfig {
            interval: 1,
            mode: ClusterMode::Wolff,
            action: ClusterAction::Observe,
            collect_stats: true,
        });
        assert!(config.validate().is_err());

        config.cluster_update = None;
        config.overlap_cluster = Some(OverlapClusterConfig {
            interval: 1,
            modes: vec![OverlapClusterBuildMode::Houdayer(4)],
            cluster_mode: ClusterMode::Sw,
            action: ClusterAction::Observe,
            collect_stats: true,
            snapshot_interval: None,
        });
        assert!(config.validate().is_err());
    }

    #[test]
    fn rejects_fft_without_autocorrelation_lag() {
        let mut config = config();
        config.autocorrelation_backend = AutocorrelationBackend::Fft;
        assert!(config.validate().is_err());

        config.autocorrelation_max_lag = Some(8);
        assert!(config.validate().is_ok());
    }
}
