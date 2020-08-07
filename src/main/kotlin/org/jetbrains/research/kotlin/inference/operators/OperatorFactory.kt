package org.jetbrains.research.kotlin.inference.operators

import org.jetbrains.research.kotlin.inference.attributes.Attribute
import org.jetbrains.research.kotlin.inference.data.ONNXData
import org.jetbrains.research.kotlin.inference.onnx.NodeProto
import org.jetbrains.research.kotlin.inference.operators.activations.*
import org.jetbrains.research.kotlin.inference.operators.flow.Loop
import org.jetbrains.research.kotlin.inference.operators.layer.recurrent.lstm.LSTM
import org.jetbrains.research.kotlin.inference.operators.math.Add
import org.jetbrains.research.kotlin.inference.operators.math.FastGelu
import org.jetbrains.research.kotlin.inference.operators.math.MatMul
import org.jetbrains.research.kotlin.inference.operators.seq.ConcatFromSequence
import org.jetbrains.research.kotlin.inference.operators.seq.SplitToSequence
import org.jetbrains.research.kotlin.inference.operators.tensor.*

object OperatorFactory {
    @Suppress("UNCHECKED_CAST")
    fun create(name: String?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (name) {
        "Add" -> Add(attributes, inputs, outputs)
        "Cast" -> Cast(attributes, inputs, outputs)
        "Concat" -> Concat(attributes, inputs, outputs)
        "ConcatFromSequence" -> ConcatFromSequence(attributes, inputs, outputs)
        "Constant" -> Constant(attributes, inputs, outputs)
        "FastGelu" -> FastGelu(attributes, inputs, outputs)
        "Gather" -> Gather(attributes, inputs, outputs)
        "Identity" -> Identity(attributes, inputs, outputs)
        "LSTM" -> LSTM(attributes, inputs, outputs)
        "Loop" -> Loop(attributes, inputs, outputs)
        "MatMul" -> MatMul(attributes, inputs, outputs)
        "Relu" -> Relu(attributes, inputs, outputs)
        "Reshape" -> Reshape(attributes, inputs, outputs)
        "Shape" -> Shape(attributes, inputs, outputs)
        "Sigmoid" -> Sigmoid(attributes, inputs, outputs)
        "Slice" -> Slice(attributes, inputs, outputs)
        "Softmax" -> Softmax(attributes, inputs, outputs)
        "Split" -> Split(attributes, inputs, outputs)
        "SplitToSequence" -> SplitToSequence(attributes, inputs, outputs)
        "Squeeze" -> Squeeze(attributes, inputs, outputs)
        "Tanh" -> Tanh(attributes, inputs, outputs)
        "Transpose" -> Transpose(attributes, inputs, outputs)
        "Unsqueeze" -> Unsqueeze(attributes, inputs, outputs)
        else -> error("Unsupported operator $name")
    } as Operator<ONNXData, ONNXData>

    fun create(proto: NodeProto) = create(proto.op_type, proto.attribute.map { Attribute.create(it) }.associateBy(Attribute<Any>::name), proto.input, proto.output)
}
