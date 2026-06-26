"""
Autonomous Research Workflow — Main Entry Point
Orchestrates the entire AI agent research cycle.
"""
import os
import sys
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.helpers import (
    get_project_root, load_yaml, save_yaml, 
    check_stop_signal, update_status_file,
    get_latest_experiment, get_git_hash,
)


class ResearchOrchestrator:
    """Main orchestrator for autonomous SOTA research."""
    
    def __init__(self, mode: str = "semi-auto"):
        self.mode = mode  # "auto" or "semi-auto"
        self.root = get_project_root()
        self.cycle_count = 0
        self.max_cycles = 50
        self.target_score = 73.0  # Target SOTA score to beat
        
        self.results_history = []
        self.current_exp_id = None
    
    def run(self):
        """Main research loop."""
        print("\n" + "="*60)
        print("  🫁 Lung Sound SOTA Research Team")
        print(f"  Mode: {self.mode}")
        print(f"  Target Score: {self.target_score}%")
        print("="*60 + "\n")
        
        while self.cycle_count < self.max_cycles:
            self.cycle_count += 1
            
            print(f"\n{'─'*60}")
            print(f"  CYCLE {self.cycle_count}/{self.max_cycles}")
            print(f"{'─'*60}\n")
            
            # Check for human STOP signal
            if check_stop_signal():
                print("⏸️  STOP_SIGNAL detected. Pausing for human review.")
                self._wait_for_human()
                continue
            
            # Step 1: Research phase
            self._research_phase()
            
            # Step 2: Architecture design phase
            self._design_phase()
            
            # Step 3: Push to GitHub → trigger training
            self._push_phase()
            
            # Step 4: Wait for training results
            self._wait_for_training()
            
            # Step 5: Pull results → evaluate
            self._evaluate_phase()
            
            # Step 6: Update paper
            self._paper_phase()
            
            # Step 7: Decide next action
            if self._should_stop():
                break
        
        print("\n" + "="*60)
        print("  ✅ Research complete!")
        f"  Best Score: {self._get_best_score():.2f}%"
        print("="*60)
    
    def _research_phase(self):
        """Step 1: Research latest SOTA methods."""
        print("📚 [Phase 1/6] Researching latest methods...")
        
        # Invoke Research Agent
        # In VS Code: this would use the research.agent.md
        # Programmatically: 
        self._run_agent_task("research", {
            "task": "find_latest_sota",
            "dataset": "ICBHI 2017",
            "focus": "pretrained models + transformers",
        })
        
        print("  ✓ Research phase complete")
    
    def _design_phase(self):
        """Step 2: Design new model architecture."""
        print("🏗️  [Phase 2/6] Designing architecture...")
        
        # Read research findings
        research_results = self._load_latest("research")
        
        # Invoke Model Architect Agent
        self._run_agent_task("model-architect", {
            "task": "design_architecture",
            "research_findings": research_results,
            "previous_results": self.results_history,
        })
        
        print("  ✓ Architecture design complete")
    
    def _push_phase(self):
        """Step 3: Push code to GitHub."""
        print("📤 [Phase 3/6] Pushing to GitHub...")
        
        # Create experiment config
        self.current_exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Git add, commit, push
        commands = [
            "git add src/ configs/ experiments/",
            f'git commit -m "[AUTO] Cycle {self.cycle_count}: {self.current_exp_id}"',
            "git push origin main",
        ]
        
        for cmd in commands:
            try:
                subprocess.run(cmd, shell=True, check=True, cwd=self.root)
            except subprocess.CalledProcessError as e:
                print(f"  ⚠️ Git command failed: {e}")
        
        # Create trigger file
        trigger = {
            "experiment_id": self.current_exp_id,
            "timestamp": datetime.now().isoformat(),
            "status": "queued",
            "git_hash": get_git_hash(),
        }
        save_yaml(trigger, self.root / "experiments" / "TRIGGER.yaml")
        
        subprocess.run(
            "git add experiments/TRIGGER.yaml && "
            f'git commit -m "[TRIGGER] Experiment {self.current_exp_id}" && '
            "git push origin main",
            shell=True, cwd=self.root
        )
        
        update_status_file(self.current_exp_id, "training_queued")
        print(f"  ✓ Pushed experiment {self.current_exp_id}")
    
    def _wait_for_training(self):
        """Step 4: Wait for training computer to complete."""
        print("⏳ [Phase 4/6] Waiting for training results...")
        
        # Poll GitHub for results
        max_wait_hours = 24
        poll_interval = 300  # 5 minutes
        waited = 0
        
        while waited < max_wait_hours * 3600:
            # Pull from GitHub
            try:
                subprocess.run(
                    "git pull origin main",
                    shell=True, check=True, cwd=self.root,
                    capture_output=True,
                )
            except:
                pass
            
            # Check if results exist
            results_path = self.root / "experiments" / self.current_exp_id / "results.yaml"
            if results_path.exists():
                print(f"  ✓ Results received! (waited {waited/3600:.1f}h)")
                
                # Also check for trigger completion
                trigger = load_yaml(self.root / "experiments" / "TRIGGER.yaml")
                if trigger.get("status") == "completed":
                    return
                else:
                    # Results exist but trigger not updated — might be partial
                    time.sleep(60)
                    continue
            
            # Check for STOP signal
            if check_stop_signal():
                print("  ⏸️  STOP signal during wait. Pausing.")
                self._wait_for_human()
                continue
            
            # Wait before next poll
            time.sleep(poll_interval)
            waited += poll_interval
            
            if waited % 3600 == 0:
                print(f"  Still waiting... ({waited/3600:.0f}h)")
        
        print("  ⚠️ Training timed out!")
    
    def _evaluate_phase(self):
        """Step 5: Pull results and evaluate."""
        print("📊 [Phase 5/6] Evaluating results...")
        
        # Run evaluation script
        subprocess.run([
            sys.executable, 
            str(self.root / "src" / "evaluation" / "evaluate.py"),
            "--exp-id", self.current_exp_id,
        ], check=True)
        
        # Load metrics
        metrics_path = self.root / "experiments" / self.current_exp_id / "metrics.json"
        if metrics_path.exists():
            import json
            with open(metrics_path) as f:
                metrics = json.load(f)
            
            score = metrics["metrics"]["score"]
            self.results_history.append({
                "exp_id": self.current_exp_id,
                "score": score,
                "se": metrics["metrics"]["sensitivity"],
                "sp": metrics["metrics"]["specificity"],
            })
            
            update_status_file(self.current_exp_id, "evaluated", metrics["metrics"])
            
            print(f"  Score: {score}% | Se: {metrics['metrics']['sensitivity']}% | "
                  f"Sp: {metrics['metrics']['specificity']}%")
    
    def _paper_phase(self):
        """Step 6: Update paper."""
        print("📝 [Phase 6/6] Updating paper...")
        
        # Update SOTA comparison table
        self._update_sota_table()
        
        print("  ✓ Paper updated")
    
    def _should_stop(self) -> bool:
        """Decide whether to stop the research loop."""
        if not self.results_history:
            return False
        
        best_score = max(r["score"] for r in self.results_history)
        
        if best_score >= self.target_score:
            print(f"\n🎉 SOTA ACHIEVED! Score: {best_score}% >= {self.target_score}%")
            return True
        
        if self.mode == "semi-auto":
            # Ask human whether to continue
            print(f"\nBest score so far: {best_score}%")
            response = input("Continue research? [Y/n]: ").strip().lower()
            if response == "n":
                return True
        
        return False
    
    def _wait_for_human(self):
        """Wait for human to remove STOP_SIGNAL."""
        print("Waiting for STOP_SIGNAL to be removed...")
        while check_stop_signal():
            time.sleep(10)
        print("Resuming research...")
    
    def _run_agent_task(self, agent_name: str, task: dict):
        """Invoke a specialized agent to perform a task."""
        # In VS Code, this uses the agent system
        # For standalone use, we simulate with print/log
        task_file = self.root / ".agent_tasks" / f"{agent_name}_{self.cycle_count}.yaml"
        task_file.parent.mkdir(exist_ok=True)
        save_yaml(task, task_file)
        print(f"  → Invoking {agent_name} agent: {task.get('task')}")
    
    def _load_latest(self, category: str) -> dict:
        """Load latest results from a category."""
        return {"category": category, "cycle": self.cycle_count}
    
    def _get_best_score(self) -> float:
        if not self.results_history:
            return 0.0
        return max(r["score"] for r in self.results_history)
    
    def _update_sota_table(self):
        """Update the SOTA comparison table."""
        sota_path = self.root / "results" / "sota_comparison.md"
        # This would be generated by the Paper Writer agent
        pass


def main():
    parser = argparse.ArgumentParser(description="Autonomous Lung Sound SOTA Research")
    parser.add_argument(
        "--mode", type=str, default="semi-auto",
        choices=["auto", "semi-auto"],
        help="auto: fully autonomous; semi-auto: asks before each cycle"
    )
    parser.add_argument(
        "--max-cycles", type=int, default=50,
        help="Maximum number of research cycles"
    )
    parser.add_argument(
        "--target-score", type=float, default=73.0,
        help="Target ICBHI Score to achieve"
    )
    args = parser.parse_args()
    
    orchestrator = ResearchOrchestrator(mode=args.mode)
    orchestrator.max_cycles = args.max_cycles
    orchestrator.target_score = args.target_score
    
    orchestrator.run()


if __name__ == "__main__":
    main()
