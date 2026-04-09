//! cuda-edge-runtime — GPU edge runtime for autonomous agents

/// Trust scoring on GPU
pub struct TrustEngine {
    pub agents: std::collections::HashMap<String, f64>,
}

impl TrustEngine {
    pub fn new() -> Self { TrustEngine { agents: std::collections::HashMap::new() } }
    pub fn trust(&self, agent: &str) -> f64 { *self.agents.get(agent).unwrap_or(&0.5) }
    pub fn record(&mut self, agent: &str, outcome: f64) -> f64 {
        let old = self.trust(agent);
        let new = old + (outcome - old) * 0.1;
        self.agents.insert(agent.to_string(), new);
        new
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_trust() {
        let mut e = TrustEngine::new();
        e.record("a", 1.0); e.record("a", 1.0);
        assert!(e.trust("a") > 0.5);
    }
}
