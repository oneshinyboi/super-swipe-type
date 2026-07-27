import onnxruntime as ort
import onnx
import sys

# Load the ONNX model
try:
    session = ort.InferenceSession(sys.argv[1])

    # Get the model's inputs
    inputs = session.get_inputs()

    print(f"The model has {len(inputs)} input(s):")
    print("-" * 30)

    for input_info in inputs:
        print(f"Input Name: {input_info.name}")
        print(f"Input Shape: {input_info.shape}")
        print(f"Input Type: {input_info.type}")
        print("-" * 30)

    # Get the model's outputs
    outputs = session.get_outputs()

    print(f"\nThe model has {len(outputs)} output(s):")
    print("-" * 30)

    for output in outputs:
        print(f"Output Name: {output.name}")
        print(f"Output Shape: {output.shape}")
        print(f"Output Type: {output.type}")
        print("-" * 30)

    # Load the model with onnx to access the graph
    model = onnx.load(sys.argv[1])

    # Extract unique operators
    unique_ops = set()
    for node in model.graph.node:
        unique_ops.add(node.op_type)

    # Display unique operators
    print(f"\nThe model uses {len(unique_ops)} unique operator(s):")
    print("-" * 30)
    for op in sorted(unique_ops):
        print(f"  - {op}")
    print("-" * 30)

except Exception as e:
    print(f"Error loading or inspecting the ONNX model: {e}")