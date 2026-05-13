import copy  # Import Python's copy module so we can duplicate the model safely.

import torch  # Import PyTorch, which provides tensors, models, and inference utilities.

import torch.ao.quantization as q  # Import PyTorch's quantization API under the short alias q.

model_to_quantize = copy.deepcopy(model_fp32)  # Deep-copy the original FP32 model because quantization modifies the model structure.

model_to_quantize.eval()  # Put the copied model in evaluation mode so layers like dropout and batch norm behave correctly for inference.

model_to_quantize.qconfig = q.get_default_qconfig("qnnpack")  # Attach the default post-training quantization config; use "qnnpack" for ARM/mobile CPUs and usually "fbgemm"/"x86" for server x86 CPUs.

prepared_model = q.prepare(model_to_quantize)  # Insert observer modules into the model so activation statistics can be collected during calibration.

with torch.inference_mode():  # Disable autograd and inference-only overhead because calibration only needs forward passes.

    for x in dataset:  # Loop through representative calibration samples from the dataset.

        prepared_model(x)  # Run each sample through the prepared model so observers record tensor statistics.

model_quantized = q.convert(prepared_model)  # Convert the calibrated model into a quantized model by replacing eligible float modules with quantized versions.