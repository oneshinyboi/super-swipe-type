import sys

import onnx

model = onnx.load(sys.argv[1])

# Replace all symbolic dims named "dec_seq" or "num_beams" with concrete values
# throughout the entire graph (inputs, outputs, and value_info).
DIM_REPLACEMENTS = {
    "dec_seq": 20,
    #"num_beams": 5,
}

def fix_type_proto(type_proto):
    if type_proto.HasField("tensor_type"):
        shape = type_proto.tensor_type.shape
        for dim in shape.dim:
            if dim.dim_param in DIM_REPLACEMENTS:
                dim.dim_value = DIM_REPLACEMENTS[dim.dim_param]
                dim.ClearField("dim_param")

for vi in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
    fix_type_proto(vi.type)

onnx.checker.check_model(model)
onnx.save(model, sys.argv[1])
print("Saved fixed model.")