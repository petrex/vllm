from vllm.config import CacheConfig, SchedulerConfig
from vllm.core.scheduler import Scheduler

from tests.core.utils import create_dummy_prompt


def _create_priority_scheduler() -> Scheduler:
    """Create a scheduler with priority policy enabled."""
    scheduler_config = SchedulerConfig(
        max_num_seqs=1,
        max_num_batched_tokens=64,
        max_model_len=64,
        policy="priority",
    )
    cache_config = CacheConfig(block_size=4,
                               gpu_memory_utilization=1.0,
                               swap_space=1,
                               cache_dtype="auto")
    cache_config.num_cpu_blocks = 16
    cache_config.num_gpu_blocks = 16
    return Scheduler(scheduler_config, cache_config, lora_config=None)


def test_high_priority_preempts_lower_priority() -> None:
    scheduler = _create_priority_scheduler()

    # Schedule a low-priority request and run it.
    _, low_group = create_dummy_prompt("low", prompt_length=4, block_size=4)
    low_group.priority = 1
    scheduler.add_seq_group(low_group)
    scheduler.schedule()
    assert [g.request_id for g in scheduler.running] == ["low"]

    # Add a higher-priority request.
    _, high_group = create_dummy_prompt("high", prompt_length=4, block_size=4)
    high_group.priority = 0
    scheduler.add_seq_group(high_group)

    # First scheduling pass should preempt the low-priority request.
    scheduler.schedule()
    assert [g.request_id for g in scheduler.waiting] == ["high", "low"]
    assert not scheduler.running

    # Next scheduling pass should run the high-priority request first.
    scheduler.schedule()
    assert [g.request_id for g in scheduler.running] == ["high"]
