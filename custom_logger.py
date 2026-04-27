"""Custom logger to force inequality violations and shield metrics into TensorBoard."""
import torch
from torch.utils.tensorboard import SummaryWriter
from omnisafe.common.logger import Logger


class CustomSafetyLogger:
    """Logger that directly writes safety metrics to TensorBoard, bypassing OmniSafe's filtering."""
    
    def __init__(self, log_dir: str):
        """Initialize custom logger with TensorBoard writer.
        
        Args:
            log_dir: Directory where TensorBoard logs are written
        """
        self.writer = SummaryWriter(log_dir=log_dir)
        self.episode_ineq_sum = 0.0
        self.episode_shield_sum = 0
        self.episode_count = 0
        self.global_step = 0
        
    def log_step(self, info: dict):
        """Log per-step information, accumulating episode metrics.
        
        Args:
            info: Info dict from environment step
        """
        self.global_step += 1
        
        # Check if episode ended and metrics are present
        if 'Metrics/EpisodeIneqViolations' in info:
            self.episode_count += 1
            ineq_viol = float(info['Metrics/EpisodeIneqViolations'])
            shield_count = float(info['Metrics/EpisodeShieldInterventions'])
            shield_rate = float(info['Metrics/ShieldInterventionRate'])
            
            # Write directly to TensorBoard
            self.writer.add_scalar('Safety/InequalityViolations', ineq_viol, self.episode_count)
            self.writer.add_scalar('Safety/ShieldInterventions', shield_count, self.episode_count)
            self.writer.add_scalar('Safety/ShieldInterventionRate', shield_rate, self.episode_count)
            
            # Also write per global step for smoother curves
            self.writer.add_scalar('Safety/IneqViolations_Step', ineq_viol, self.global_step)
            self.writer.add_scalar('Safety/ShieldCount_Step', shield_count, self.global_step)
            
            # Accumulate for averaging
            self.episode_ineq_sum += ineq_viol
            self.episode_shield_sum += shield_count
            
            # Print to console for verification
            print(f"\n📊 Episode {self.episode_count} Safety Metrics:")
            print(f"   Inequality Violations: {ineq_viol:.2f}")
            print(f"   Shield Interventions: {shield_count}")
            print(f"   Shield Rate: {shield_rate:.2%}")
            
    def log_epoch(self, epoch: int):
        """Log averaged metrics at end of epoch.
        
        Args:
            epoch: Current epoch number
        """
        if self.episode_count > 0:
            avg_ineq = self.episode_ineq_sum / self.episode_count
            avg_shield = self.episode_shield_sum / self.episode_count
            
            self.writer.add_scalar('Safety/AvgIneqViolations_Epoch', avg_ineq, epoch)
            self.writer.add_scalar('Safety/AvgShieldInterventions_Epoch', avg_shield, epoch)
            
            print(f"\n🎯 Epoch {epoch} Average Safety Metrics:")
            print(f"   Avg Inequality Violations: {avg_ineq:.2f}")
            print(f"   Avg Shield Interventions: {avg_shield:.2f}")
            
            # Reset accumulators
            self.episode_ineq_sum = 0.0
            self.episode_shield_sum = 0
            self.episode_count = 0
            
    def close(self):
        """Close TensorBoard writer."""
        self.writer.close()
