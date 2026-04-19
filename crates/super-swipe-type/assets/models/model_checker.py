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

    # Track unique operators using a dictionary
    model = onnx.load_model(sys.argv[1])
    operators = {}
    for node in model.graph.node:
        operators[node.op_type] = operators.get(node.op_type, 0) + 1

    print(f"\nUnique Operators ({len(operators)} types):")
    print("-" * 30)
    for op_type, count in sorted(operators.items()):
        print(f"{op_type}: {count}")

except Exception as e:
    print(f"Error loading or inspecting the ONNX model: {e}")