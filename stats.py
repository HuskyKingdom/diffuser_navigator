# flops
from fvcore.nn import FlopCountAnalysis, flop_count_str

input_seq = (
inputids,
None,
transformed_images_tensor,
None, # expect labels during train, expect prev_actions during inference
None,
None,
None,
None,
None,
None,
None,
img_ori_shape,
collected_data['lengths'],
True,
transformed_his_tensor,
None,
None,
None,
)
logging.getLogger("fvcore").setLevel(logging.ERROR)
with torch.cuda.amp.autocast(dtype=cast_type):
    flop_analyzer = FlopCountAnalysis(
        self.vlm,
        input_seq
    )
    total_flops: int = flop_analyzer.total()
    print(f"Total FLOPs: {total_flops / 1e12:.2f}T")

with torch.cuda.amp.autocast(dtype=cast_type):
    modelout = self.vlm(input_ids=inputids, attention_mask=None,pixel_values=transformed_images_tensor, labels = None, img_ori_shape = img_ori_shape, sample_valid_len = collected_data['lengths'], inference = True, full_his = transformed_his_tensor)




# memory peak