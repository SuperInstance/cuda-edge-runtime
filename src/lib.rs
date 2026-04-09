//! # cuda-edge-runtime
//!
//! Edge runtime for fleet vessels — trust engine, reflex actions, fleet coordination.
//! Built on cuda-equipment for shared types.
//!
//! ```rust
//! use cuda_edge_runtime::{TrustEngine, ReflexEngine, FleetCoordinator};
//! use cuda_equipment::{Confidence, VesselId, BaseAgent, Fleet};
//! ```

pub use cuda_equipment::{Confidence, VesselId, FleetMessage, MessageType,
    Agent, BaseAgent, Fleet, AgentBuilder, EquipmentRegistry, SensorType, ActuatorType,
    Tile, TileGrid, TileId};

use std::collections::HashMap;

// ============================================================
// TRUST ENGINE
// ============================================================

/// Trust level between vessels.
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct TrustLevel(pub f64);

impl TrustLevel {
    pub const NEW: Self = TrustLevel(0.5);
    pub const VERIFIED: Self = TrustLevel(0.8);
    pub const TRUSTED: Self = TrustLevel(0.95);
    pub const REVOKED: Self = TrustLevel(0.0);

    pub fn new(v: f64) -> Self { TrustLevel(v.clamp(0.0, 1.0)) }
    pub fn value(&self) -> f64 { self.0 }

    pub fn upgrade(self, delta: f64) -> Self { TrustLevel::new((self.0 + delta).min(1.0)) }
    pub fn degrade(self, delta: f64) -> Self { TrustLevel::new((self.0 - delta).max(0.0)) }
    pub fn decay(self, rounds: u32, rate: f64) -> Self { TrustLevel::new(self.0 * rate.powi(rounds as i32)) }
    pub fn to_confidence(self) -> Confidence { Confidence::new(self.0) }
}

/// Records of trust interactions.
#[derive(Debug, Clone)]
pub struct TrustRecord {
    pub from: VesselId,
    pub action: String,
    pub trust_delta: f64,
    pub new_level: TrustLevel,
    pub timestamp: u64,
}

/// Trust engine — manages trust relationships between fleet vessels.
pub struct TrustEngine {
    trust: HashMap<(u64, u64), TrustLevel>,  // (observer, target) → trust
    records: Vec<TrustRecord>,
    default_trust: TrustLevel,
}

impl TrustEngine {
    pub fn new(default_trust: TrustLevel) -> Self {
        Self { trust: HashMap::new(), records: vec![], default_trust }
    }

    pub fn trust_level(&self, observer: u64, target: u64) -> TrustLevel {
        *self.trust.get(&(observer, target)).unwrap_or(&self.default_trust)
    }

    /// Record a positive interaction (target did good).
    pub fn reward(&mut self, observer: u64, target: u64, action: &str, delta: f64) {
        let current = self.trust_level(observer, target);
        let new_level = current.upgrade(delta);
        self.trust.insert((observer, target), new_level);
        self.records.push(TrustRecord {
            from: VesselId(target), action: action.to_string(),
            trust_delta: delta, new_level, timestamp: now_ms(),
        });
    }

    /// Record a negative interaction.
    pub fn penalize(&mut self, observer: u64, target: u64, action: &str, delta: f64) {
        let current = self.trust_level(observer, target);
        let new_level = current.degrade(delta);
        self.trust.insert((observer, target), new_level);
        self.records.push(TrustRecord {
            from: VesselId(target), action: action.to_string(),
            trust_delta: -delta, new_level, timestamp: now_ms(),
        });
    }

    /// Get top N most trusted vessels from observer's perspective.
    pub fn top_trusted(&self, observer: u64, n: usize) -> Vec<(u64, TrustLevel)> {
        let mut all: Vec<(u64, TrustLevel)> = self.trust.iter()
            .filter_map(|((obs, target), level)| if *obs == observer { Some((*target, *level)) } else { None })
            .collect();
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        all.truncate(n)
    }

    pub fn record_count(&self) -> usize { self.records.len() }
    pub fn decay_all(&mut self, rate: f64) {
        for level in self.trust.values_mut() { *level = level.decay(1, rate); }
    }
}

// ============================================================
// REFLEX ENGINE
// ============================================================

/// A reflex action — triggered without deliberation.
#[derive(Debug, Clone)]
pub struct ReflexAction {
    pub id: String,
    pub trigger: String,
    pub response: String,
    pub priority: u8,        // 0=highest
    pub confidence_gate: f64, // minimum confidence to trigger
    pub enabled: bool,
    pub fire_count: u64,
}

impl ReflexAction {
    pub fn new(id: &str, trigger: &str, response: &str, priority: u8) -> Self {
        Self { id: id.to_string(), trigger: trigger.to_lowercase(),
            response: response.to_string(), priority, confidence_gate: 0.7,
            enabled: true, fire_count: 0 }
    }

    pub fn with_gate(mut self, threshold: f64) -> Self { self.confidence_gate = threshold; self }
    pub fn check(&self, input: &str, confidence: f64) -> bool {
        self.enabled && confidence >= self.confidence_gate && input.to_lowercase().contains(&self.trigger)
    }
}

/// Reflex engine — fast-path actions that bypass deliberation.
pub struct ReflexEngine {
    actions: Vec<ReflexAction>,
    fire_log: Vec<(String, u64)>, // (action_id, timestamp)
}

impl ReflexEngine {
    pub fn new() -> Self { Self { actions: vec![], fire_log: vec![] } }

    pub fn add_action(&mut self, action: ReflexAction) { self.actions.push(action); }

    /// Evaluate input against all reflexes. Returns matching responses sorted by priority.
    pub fn evaluate(&mut self, input: &str, confidence: Confidence) -> Vec<&str> {
        let conf = confidence.value();
        let mut matches: Vec<&ReflexAction> = self.actions.iter()
            .filter(|a| a.check(input, conf))
            .collect();
        matches.sort_by_key(|a| a.priority);
        matches.iter().map(|a| {
            self.fire_log.push((a.id.clone(), now_ms()));
            a.response.as_str()
        }).collect()
    }

    /// Add default safety reflexes.
    pub fn add_default_safety_reflexes(&mut self) {
        self.add_action(ReflexAction::new("collision", "collision imminent", "STOP", 0).with_gate(0.8));
        self.add_action(ReflexAction::new("overheat", "temperature critical", "SHUTDOWN", 0).with_gate(0.9));
        self.add_action(ReflexAction::new("low_power", "battery critical", "CONSERVE", 1).with_gate(0.8));
        self.add_action(ReflexAction::new("comms_lost", "communication lost", "RECONNECT", 1).with_gate(0.7));
        self.add_action(ReflexAction::new("unauthorized", "unauthorized access", "DENY", 0).with_gate(0.9));
    }

    pub fn fire_count(&self, action_id: &str) -> u64 {
        self.fire_log.iter().filter(|(id, _)| id == action_id).count() as u64
    }
}

// ============================================================
// FLEET COORDINATOR
// ============================================================

/// Task assignment for fleet coordination.
#[derive(Debug, Clone)]
pub struct TaskAssignment {
    pub task_id: u64,
    pub task_type: String,
    pub assigned_to: VesselId,
    pub priority: u8,
    pub confidence: Confidence,
    pub status: TaskStatus,
}

#[derive(Debug, Clone, PartialEq)]
pub enum TaskStatus {
    Pending,
    InProgress,
    Completed,
    Failed,
    Abandoned,
}

/// Fleet coordinator — assigns tasks to agents based on capabilities and trust.
pub struct FleetCoordinator {
    trust: TrustEngine,
    pending_tasks: Vec<TaskAssignment>,
    completed_tasks: Vec<TaskAssignment>,
    next_task_id: u64,
}

impl FleetCoordinator {
    pub fn new() -> Self {
        Self { trust: TrustEngine::new(TrustLevel::NEW),
            pending_tasks: vec![], completed_tasks: vec![], next_task_id: 1 }
    }

    pub fn submit_task(&mut self, task_type: &str, priority: u8) -> u64 {
        let id = self.next_task_id;
        self.next_task_id += 1;
        self.pending_tasks.push(TaskAssignment {
            task_id: id, task_type: task_type.to_string(),
            assigned_to: VesselId(0), priority, confidence: Confidence::HALF,
            status: TaskStatus::Pending,
        });
        id
    }

    /// Assign task to most trusted capable agent.
    pub fn assign(&mut self, task_id: u64, agent_id: u64) -> Option<u64> {
        let task = self.pending_tasks.iter_mut().find(|t| t.task_id == task_id)?;
        task.assigned_to = VesselId(agent_id);
        task.status = TaskStatus::InProgress;
        task.confidence = self.trust.trust_level(0, agent_id).to_confidence();
        Some(task_id)
    }

    /// Complete a task.
    pub fn complete(&mut self, task_id: u64, success: bool) {
        if let Some(pos) = self.pending_tasks.iter().position(|t| t.task_id == task_id) {
            let mut task = self.pending_tasks.remove(pos);
            task.status = if success { TaskStatus::Completed } else { TaskStatus::Failed };
            // Update trust
            if let VesselId(agent) = task.assigned_to {
                if success { self.trust.reward(0, agent, "task_complete", 0.05); }
                else { self.trust.penalize(0, agent, "task_failed", 0.1); }
            }
            self.completed_tasks.push(task);
        }
    }

    pub fn pending_count(&self) -> usize { self.pending_tasks.iter().filter(|t| t.status == TaskStatus::Pending).count() }
    pub fn active_count(&self) -> usize { self.pending_tasks.iter().filter(|t| t.status == TaskStatus::InProgress).count() }
    pub fn completed_count(&self) -> usize { self.completed_tasks.len() }

    pub fn most_trusted_agent(&self, observer: u64) -> Option<(u64, TrustLevel)> {
        let top = self.trust.top_trusted(observer, 1);
        top.into_iter().next()
    }
}

fn now_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now().duration_since(UNIX_EPOCH).map_or(0, |d| d.as_millis() as u64)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trust_reward() {
        let mut te = TrustEngine::new(TrustLevel::NEW);
        te.reward(0, 1, "good_response", 0.1);
        assert!(te.trust_level(0, 1).value() > 0.5);
    }

    #[test]
    fn test_trust_penalize() {
        let mut te = TrustEngine::new(TrustLevel::VERIFIED);
        te.penalize(0, 1, "bad_response", 0.2);
        assert!(te.trust_level(0, 1).value() < 0.8);
    }

    #[test]
    fn test_trust_decay() {
        let mut te = TrustEngine::new(TrustLevel::TRUSTED);
        te.reward(0, 1, "init", 0.0);
        te.decay_all(0.9);
        assert!(te.trust_level(0, 1).value() < 0.95);
    }

    #[test]
    fn test_top_trusted() {
        let mut te = TrustEngine::new(TrustLevel::NEW);
        te.reward(0, 1, "a", 0.3);
        te.reward(0, 2, "b", 0.1);
        let top = te.top_trusted(0, 5);
        assert_eq!(top[0].0, 1);
    }

    #[test]
    fn test_reflex_trigger() {
        let mut re = ReflexEngine::new();
        re.add_default_safety_reflexes();
        let responses = re.evaluate("collision imminent ahead", Confidence::SURE);
        assert!(!responses.is_empty());
        assert!(responses[0] == "STOP");
    }

    #[test]
    fn test_reflex_gate() {
        let mut re = ReflexEngine::new();
        re.add_action(ReflexAction::new("test", "danger", "ACT", 0).with_gate(0.9));
        let responses = re.evaluate("danger detected", Confidence::UNLIKELY); // 0.25 < 0.9
        assert!(responses.is_empty());
    }

    #[test]
    fn test_fleet_task_lifecycle() {
        let mut fc = FleetCoordinator::new();
        let id = fc.submit_task("scan", 1);
        fc.assign(id, 1);
        assert_eq!(fc.active_count(), 1);
        fc.complete(id, true);
        assert_eq!(fc.completed_count(), 1);
        assert_eq!(fc.pending_count(), 0);
    }

    #[test]
    fn test_task_failure_degrades_trust() {
        let mut fc = FleetCoordinator::new();
        let id = fc.submit_task("compute", 0);
        fc.assign(id, 1);
        let trust_before = fc.most_trusted_agent(0).unwrap().1.value();
        fc.complete(id, false);
        let trust_after = fc.most_trusted_agent(0).unwrap().1.value();
        assert!(trust_after < trust_before);
    }

    #[test]
    fn test_trust_level_bounds() {
        let t = TrustLevel::new(1.5);
        assert_eq!(t.value(), 1.0);
        let t2 = TrustLevel::new(-0.5);
        assert_eq!(t2.value(), 0.0);
    }

    #[test]
    fn test_trust_to_confidence() {
        let t = TrustLevel::VERIFIED;
        let c = t.to_confidence();
        assert_eq!(c.value(), 0.8);
    }

    #[test]
    fn test_reflex_priority() {
        let mut re = ReflexEngine::new();
        re.add_action(ReflexAction::new("low", "emergency", "LOW", 5));
        re.add_action(ReflexAction::new("high", "emergency", "HIGH", 0));
        let responses = re.evaluate("emergency now", Confidence::SURE);
        assert_eq!(responses[0], "HIGH"); // priority 0 first
    }
}
