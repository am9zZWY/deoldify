# DeOldify on Browser

<img src="img/screenshot.jpg" title="screenshot" alt="screenshot" style="text-align: center">


### **Description / Rationale**
This repository demonstrates web-based implementation of DeOldify, a Deep Learning based project for colorizing and restoring old images. This project demonstrates implementation of 2 models:
1. Original artistic DeOldify model ("original" folder)
2. Quantized DeOldify model ("quantized" folder).

### **Instructions**
To use a DeOldify example, copy the corresponding html file contents. To dowload and locally serve models download them from the link provided:
1. Original artistic model: https://cdn.glitch.me/2046b88b-673a-457f-b1b8-7169ce9bf13a/deoldify-art.onnx (~243mb)
2. Quantized model: https://cdn.glitch.me/2046b88b-673a-457f-b1b8-7169ce9bf13a/deoldify-quant.onnx (~61mb)


### **Onnx files and Quantization**
Original onnx files were taken from releases page of <a href='https://github.com/instant-high/deoldify-onnx/releases/tag/deoldify-onnx'>Deoldify Onnx repository by Thomas De</a>.
To quantize an onnx file do the following:
1. Open Google Colab and create a new Notebook.
2. Upload onnx file downloaded from releases page above. Also upload "remove_initializer_from_input.py". The file content is given below:
   
```python
# /content/remove_initializer_from_input.py
import argparse
import onnx
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input model")
    parser.add_argument("--output", required=True, help="output model")
    args = parser.parse_args()
    return args
def remove_initializer_from_input():
    args = get_args()
    model = onnx.load(args.input)
    if model.ir_version < 4:
        print("Model with ir_version below 4 requires to include initilizer in graph input")
        return
    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input
    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])
    onnx.save(model, args.output)

if __name__ == "__main__":
    remove_initializer_from_input()
```
 
3. Run the following code:
   
```python
# Install dependencies and 
!pip install onnxruntime
!pip install onnx
```

Run onnx preprocess:

```python
!python -m onnxruntime.quantization.preprocess --input '/content/deoldify.onnx' --output '/content/deoldify-final.onnx'
```

Generate quantized file:

```python
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = '/content/deoldify-final.onnx'
model_quant = '/content/deoldify-quant.onnx'
quantized_model = quantize_dynamic(model_fp32, model_quant, weight_type=QuantType.QUInt8)
```
Remove initializer from input of deoldify-quant.onnx file (otherwise the model will be throwing an error related to initializer):

```python
!python /content/remove_initializer_from_input.py --input /content/deoldify-quant.onnx --output /content/deoldify-quant-clear.onnx
```


### **Demo**
To see quantized DeOldify model at work, visit the following page: <a href="https://deoldify.glitch.me/">Demo</a> 
