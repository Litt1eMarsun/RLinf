import torch
from transformers import AutoProcessor

from transformers.generation.logits_process import LogitsProcessor


# Helper functions from alpamayo_r1/helper.py
MIN_PIXELS = 163840
MAX_PIXELS = 196608
BASE_PROCESSOR_NAME = "Qwen/Qwen3-VL-2B-Instruct"


def create_message(frames: torch.Tensor):
    """Construct the message using images."""
    assert frames.ndim == 4, f"{frames.ndim=}, expected (N, C, H, W)"
    
    num_traj_token = 48
    hist_traj_placeholder = (
        f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"
    )
    
    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a driving assistant that generates safe and accurate actions.",
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "image", "image": frame} for frame in frames]
            + [
                {
                    "type": "text",
                    "text": f"{hist_traj_placeholder}output the chain-of-thought reasoning of the driving process, then output the future trajectory.",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "<|cot_start|>",
                }
            ],
        },
    ]


def get_processor(tokenizer) -> AutoProcessor:
    """Get the processor for the Qwen3-VL-2B-Instruct model."""
    processor_kwargs = {
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
    }
    
    processor = AutoProcessor.from_pretrained(BASE_PROCESSOR_NAME, **processor_kwargs)
    processor.tokenizer = tokenizer
    return processor


class ConditionalTrajLogitsProcessor(LogitsProcessor):
    """
    两阶段条件 LogitsProcessor，用于 AR 轨迹 token 生成：

    Phase 1 – CoT 阶段（还未见到 <|traj_future_start|>）：
        屏蔽掉轨迹 token 区间，让 VLM 只生成文字（与 ExpertLogitsProcessor 行为相同）。

    Phase 2 – 轨迹生成阶段（已见到 <|traj_future_start|>，剩余 traj token 数 < tokens_per_traj）：
        屏蔽掉所有非轨迹 token，强制 VLM 只从 [traj_token_offset, traj_token_offset + traj_vocab_size) 采样。

    Phase 3 – 轨迹结束（已生成 tokens_per_traj 个轨迹 token）：
        强制输出 <|traj_future_end|>，然后由 StopAfterTrajEnd 停止生成。
    """

    def __init__(
        self,
        traj_start_id: int,
        traj_end_id: int,
        traj_token_offset: int,
        traj_vocab_size: int,
        tokens_per_traj: int,
    ):
        super().__init__()
        self.traj_start_id = traj_start_id
        self.traj_end_id = traj_end_id
        self.traj_token_offset = traj_token_offset
        self.traj_vocab_size = traj_vocab_size
        self.tokens_per_traj = tokens_per_traj

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        for b in range(input_ids.shape[0]):
            seq = input_ids[b]
            start_mask = seq == self.traj_start_id

            if start_mask.any():
                # Phase 2/3: 已经到了轨迹生成阶段 取出索引
                start_pos = start_mask.nonzero(as_tuple=True)[0][0].item()
                n_traj_generated = len(seq) - start_pos - 1  # start token 之后生成的 token 数

                if n_traj_generated >= self.tokens_per_traj:
                    # Phase 3: 强制 <|traj_future_end|> 即 所有的logits输出都为负无穷，只有end为0，采样出来的只能是end_id
                    scores[b] = float("-inf")
                    scores[b, self.traj_end_id] = 0.0
                else:
                    # Phase 2: 只允许轨迹 token 即 所有的非轨迹的id全部都设置为inf，剩下的轨迹id保留，后续采样策略也只能在轨迹中进行采样。
                    scores[b, : self.traj_token_offset] = float("-inf")
                    scores[b, self.traj_token_offset + self.traj_vocab_size :] = float("-inf")
            else:
                # Phase 1: CoT 阶段，屏蔽轨迹 token（让 VLM 生成干净的文字） 与上面同理
                scores[b, self.traj_token_offset : self.traj_token_offset + self.traj_vocab_size] = float("-inf")

        return scores
