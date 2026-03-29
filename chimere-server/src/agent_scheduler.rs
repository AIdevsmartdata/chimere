//! Multi-agent context switching scheduler.
//!
//! Manages N agent conversations sharing a single LlamaForward context.
//! State is saved/restored via llama_state_seq_{get,set}_data when switching.
//! Zero contamination: full KV cache + GDN recurrent state serialized per agent.
//!
//! # Architecture
//!
//! All agents share seq_id=0 (single sequence, serialized access enforced by
//! the tokio::sync::Mutex wrapping LlamaForward in the server). When switching
//! from agent A to agent B:
//!
//! 1. Save A's state (KV + GDN recurrent) via `llama_state_seq_get_data(ctx, 0)`
//! 2. Clear the KV cache
//! 3. Restore B's state via `llama_state_seq_set_data(ctx, 0)`
//! 4. Restore B's position counter
//!
//! The position counter (`pos`) lives in `LlamaForward` and is NOT part of
//! the serialized state, so it must be saved/restored manually.
//!
//! # Token history
//!
//! Each agent keeps a `token_history` as a fallback: if the serialized state
//! is ever lost or corrupted, the scheduler can re-prefill from the token
//! history to reconstruct the KV cache. This is not used in the normal path.

use std::time::Instant;

use crate::llama_backend::LlamaForward;

// ---------------------------------------------------------------------------
// Agent state
// ---------------------------------------------------------------------------

/// Saved state for one agent conversation.
pub struct AgentState {
    /// Agent identifier (e.g., "kevin", "melanie").
    pub name: String,
    /// Serialized KV cache + GDN state (from llama_state_seq_get_data).
    state_data: Vec<u8>,
    /// Token history for this conversation (for re-prefill if state is lost).
    pub token_history: Vec<u32>,
    /// Position counter (mirrors LlamaForward.pos for this agent).
    pub pos: i32,
    /// Whether this agent has been initialized (first message received).
    pub active: bool,
    /// Timestamp of last activity.
    pub last_active: Instant,
    /// Number of context switches into this agent.
    switch_count: u64,
}

impl AgentState {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            state_data: Vec::new(),
            token_history: Vec::new(),
            pos: 0,
            active: false,
            last_active: Instant::now(),
            switch_count: 0,
        }
    }

    /// Size of the serialized state in bytes.
    pub fn state_size(&self) -> usize {
        self.state_data.len()
    }
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

/// Multi-agent scheduler managing N conversations on 1 LlamaForward.
///
/// Usage:
/// ```ignore
/// let mut sched = AgentScheduler::new(4);
/// let kevin = sched.register_agent("kevin");
/// let melanie = sched.register_agent("melanie");
///
/// // Switch to kevin (first time: no state to save, no state to restore)
/// sched.switch_to(kevin, &mut llama)?;
/// // ... run inference for kevin ...
///
/// // Switch to melanie (saves kevin's state, restores melanie's)
/// sched.switch_to(melanie, &mut llama)?;
/// // ... run inference for melanie ...
/// ```
pub struct AgentScheduler {
    /// Per-agent saved states, indexed by agent_id (0..max_agents-1).
    agents: Vec<AgentState>,
    /// Currently loaded agent_id (-1 = none loaded).
    current_agent: i32,
    /// Maximum number of agents.
    max_agents: usize,
    /// Total number of context switches performed.
    total_switches: u64,
    /// Cumulative switch time in milliseconds.
    total_switch_time_ms: f64,
}

impl AgentScheduler {
    /// Create a new scheduler for `max_agents` conversations.
    pub fn new(max_agents: usize) -> Self {
        Self {
            agents: Vec::with_capacity(max_agents),
            current_agent: -1,
            max_agents,
            total_switches: 0,
            total_switch_time_ms: 0.0,
        }
    }

    /// Register an agent with a name. Returns agent_id.
    ///
    /// # Panics
    /// Panics if `max_agents` slots are already registered.
    pub fn register_agent(&mut self, name: &str) -> usize {
        assert!(
            self.agents.len() < self.max_agents,
            "AgentScheduler: cannot register more than {} agents",
            self.max_agents,
        );
        let id = self.agents.len();
        self.agents.push(AgentState::new(name));
        eprintln!(
            "[AGENT_SCHED] Registered agent {} (id={}), {}/{} slots used",
            name,
            id,
            self.agents.len(),
            self.max_agents,
        );
        id
    }

    /// Switch to `agent_id`. Saves current agent's state, restores target agent's state.
    ///
    /// Returns `Ok(switch_time_ms)` on success. Returns 0.0 if the target agent
    /// is already loaded (no-op).
    ///
    /// # Errors
    /// Returns `Err` if state save or restore fails (should not happen in practice
    /// unless the context is corrupted or OOM).
    pub fn switch_to(
        &mut self,
        agent_id: usize,
        llama: &mut LlamaForward,
    ) -> Result<f64, String> {
        // Validate agent_id.
        if agent_id >= self.agents.len() {
            return Err(format!(
                "AgentScheduler: agent_id {} out of range (registered: {})",
                agent_id,
                self.agents.len(),
            ));
        }

        // No-op if already on the target agent.
        if self.current_agent == agent_id as i32 {
            return Ok(0.0);
        }

        let t0 = Instant::now();

        // 1. Save current agent's state (if one is loaded).
        if self.current_agent >= 0 {
            let cur = self.current_agent as usize;
            let state_data = llama.state_seq_save(0)?;
            let cur_pos = llama.pos();
            self.agents[cur].state_data = state_data;
            self.agents[cur].pos = cur_pos;
            eprintln!(
                "[AGENT_SCHED] Saved agent {} ({}) state: {} bytes, pos={}",
                cur,
                self.agents[cur].name,
                self.agents[cur].state_data.len(),
                cur_pos,
            );
        }

        // 2. Clear KV cache (full wipe before restoring new agent).
        llama.reset();

        // 3. Restore target agent's state (if it has any).
        if !self.agents[agent_id].state_data.is_empty() {
            llama.state_seq_restore(0, &self.agents[agent_id].state_data)?;
            llama.set_pos(self.agents[agent_id].pos);
            eprintln!(
                "[AGENT_SCHED] Restored agent {} ({}) state: {} bytes, pos={}",
                agent_id,
                self.agents[agent_id].name,
                self.agents[agent_id].state_data.len(),
                self.agents[agent_id].pos,
            );
        } else {
            // First time for this agent, or agent was reset — pos stays at 0.
            eprintln!(
                "[AGENT_SCHED] Agent {} ({}) has no saved state (new conversation)",
                agent_id,
                self.agents[agent_id].name,
            );
        }

        // 4. Update bookkeeping.
        self.current_agent = agent_id as i32;
        self.agents[agent_id].active = true;
        self.agents[agent_id].last_active = Instant::now();
        self.agents[agent_id].switch_count += 1;

        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
        self.total_switches += 1;
        self.total_switch_time_ms += elapsed_ms;

        eprintln!(
            "[AGENT_SCHED] Switched to agent {} ({}) in {:.2} ms",
            agent_id,
            self.agents[agent_id].name,
            elapsed_ms,
        );

        Ok(elapsed_ms)
    }

    /// Get the currently loaded agent id, or `None` if no agent is loaded.
    pub fn current(&self) -> Option<usize> {
        if self.current_agent >= 0 {
            Some(self.current_agent as usize)
        } else {
            None
        }
    }

    /// Get the name of the currently loaded agent, or `None`.
    pub fn current_name(&self) -> Option<&str> {
        self.current().map(|id| self.agents[id].name.as_str())
    }

    /// Reset an agent's state (start a new conversation).
    ///
    /// If the agent is currently loaded, also clears the live KV cache.
    pub fn reset_agent(&mut self, agent_id: usize, llama: &mut LlamaForward) {
        if agent_id >= self.agents.len() {
            return;
        }

        self.agents[agent_id].state_data.clear();
        self.agents[agent_id].token_history.clear();
        self.agents[agent_id].pos = 0;
        self.agents[agent_id].active = false;

        // If this agent is currently loaded, clear the live context too.
        if self.current_agent == agent_id as i32 {
            llama.reset();
            // Mark no agent loaded — the next switch_to will be a clean start.
            self.current_agent = -1;
        }

        eprintln!(
            "[AGENT_SCHED] Reset agent {} ({})",
            agent_id,
            self.agents[agent_id].name,
        );
    }

    /// Append tokens to an agent's history (call after each generation step).
    pub fn push_tokens(&mut self, agent_id: usize, tokens: &[u32]) {
        if agent_id < self.agents.len() {
            self.agents[agent_id].token_history.extend_from_slice(tokens);
        }
    }

    /// Get the number of registered agents.
    pub fn num_agents(&self) -> usize {
        self.agents.len()
    }

    /// Get a reference to an agent's state by id.
    pub fn agent(&self, agent_id: usize) -> Option<&AgentState> {
        self.agents.get(agent_id)
    }

    /// Find an agent id by name.
    pub fn find_agent(&self, name: &str) -> Option<usize> {
        self.agents.iter().position(|a| a.name == name)
    }

    /// Print stats: state sizes, switch counts, per-agent positions.
    pub fn print_stats(&self) {
        eprintln!("=== AgentScheduler Stats ===");
        eprintln!(
            "  Agents: {}/{} registered",
            self.agents.len(),
            self.max_agents,
        );
        eprintln!(
            "  Current: {}",
            self.current()
                .map(|id| format!("{} ({})", id, self.agents[id].name))
                .unwrap_or_else(|| "none".into()),
        );
        eprintln!(
            "  Total switches: {}, total switch time: {:.1} ms, avg: {:.2} ms",
            self.total_switches,
            self.total_switch_time_ms,
            if self.total_switches > 0 {
                self.total_switch_time_ms / self.total_switches as f64
            } else {
                0.0
            },
        );
        eprintln!("  Per-agent:");
        for (i, a) in self.agents.iter().enumerate() {
            let state_kb = a.state_data.len() as f64 / 1024.0;
            let history_len = a.token_history.len();
            let idle_secs = a.last_active.elapsed().as_secs();
            eprintln!(
                "    [{}] {:12} | active={:<5} | pos={:<6} | state={:>8.1} KB | history={:<6} tok | switches={:<4} | idle={}s",
                i,
                a.name,
                a.active,
                a.pos,
                state_kb,
                history_len,
                a.switch_count,
                idle_secs,
            );
        }
        eprintln!("============================");
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_agents() {
        let mut sched = AgentScheduler::new(4);
        let a = sched.register_agent("kevin");
        let b = sched.register_agent("melanie");

        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(sched.num_agents(), 2);
        assert_eq!(sched.current(), None);
        assert_eq!(sched.find_agent("kevin"), Some(0));
        assert_eq!(sched.find_agent("melanie"), Some(1));
        assert_eq!(sched.find_agent("unknown"), None);
    }

    #[test]
    #[should_panic(expected = "cannot register more than 2 agents")]
    fn test_register_overflow() {
        let mut sched = AgentScheduler::new(2);
        sched.register_agent("a");
        sched.register_agent("b");
        sched.register_agent("c"); // should panic
    }

    #[test]
    fn test_agent_state_default() {
        let state = AgentState::new("test");
        assert_eq!(state.name, "test");
        assert!(state.state_data.is_empty());
        assert!(state.token_history.is_empty());
        assert_eq!(state.pos, 0);
        assert!(!state.active);
        assert_eq!(state.switch_count, 0);
        assert_eq!(state.state_size(), 0);
    }

    #[test]
    fn test_push_tokens() {
        let mut sched = AgentScheduler::new(2);
        let id = sched.register_agent("test");
        sched.push_tokens(id, &[1, 2, 3]);
        sched.push_tokens(id, &[4, 5]);
        assert_eq!(sched.agent(id).unwrap().token_history, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_switch_to_invalid_id() {
        // We cannot test switch_to with a real LlamaForward (requires GPU + GGUF),
        // but we can test the validation path by creating a scheduler with no
        // registered agents and checking the error.
        let sched = AgentScheduler::new(4);
        // No agents registered, so agent_id=0 is out of range.
        // We need a LlamaForward to call switch_to, but we can't create one
        // in unit tests. Instead, we just verify the scheduler state logic.
        assert_eq!(sched.current(), None);
        assert_eq!(sched.num_agents(), 0);
    }
}
