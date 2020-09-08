package io.kinference.operators

import io.kinference.attributes.Attribute
import io.kinference.data.ONNXData
import io.kinference.onnx.NodeProto
import io.kinference.operators.activations.*
import io.kinference.operators.flow.Loop
import io.kinference.operators.layer.attention.Attention
import io.kinference.operators.layer.normalization.*
import io.kinference.operators.layer.recurrent.lstm.LSTM
import io.kinference.operators.logical.Equal
import io.kinference.operators.logical.Not
import io.kinference.operators.math.*
import io.kinference.operators.seq.ConcatFromSequence
import io.kinference.operators.seq.SplitToSequence
import io.kinference.operators.tensor.*

@ExperimentalUnsignedTypes
object OperatorFactory {
    @Suppress("UNCHECKED_CAST")
    fun create(name: String?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (name) {
        "Add" -> Add(attributes, inputs, outputs)
        "Sub" -> Sub(attributes, inputs, outputs)
        "Attention" -> Attention(attributes, inputs, outputs)
        "BiasGelu" -> BiasGelu(attributes, inputs, outputs)
        "Cast" -> Cast(attributes, inputs, outputs)
        "Concat" -> Concat(attributes, inputs, outputs)
        "ConcatFromSequence" -> ConcatFromSequence(attributes, inputs, outputs)
        "Constant" -> Constant(attributes, inputs, outputs)
        "ConstantOfShape" -> ConstantOfShape(attributes, inputs, outputs)
        "CumSum" -> CumSum(attributes, inputs, outputs)
        "EmbedLayerNormalization" -> EmbedLayerNormalization(attributes, inputs, outputs)
        "Equal" -> Equal(attributes, inputs, outputs)
        "Erf" -> Erf(attributes, inputs, outputs)
        "FastGelu" -> FastGelu(attributes, inputs, outputs)
        "Gather" -> Gather(attributes, inputs, outputs)
        "Gelu" -> Gelu(attributes, inputs, outputs)
        "Gemm" -> Gemm(attributes, inputs, outputs)
        "Identity" -> Identity(attributes, inputs, outputs)
        "LSTM" -> LSTM(attributes, inputs, outputs)
        "Loop" -> Loop(attributes, inputs, outputs)
        "LayerNormalization" -> LayerNormalization(attributes, inputs, outputs)
        "MatMul" -> MatMul(attributes, inputs, outputs)
        "Mul" -> Mul(attributes, inputs, outputs)
        "Not" -> Not(attributes, inputs, outputs)
        "Relu" -> Relu(attributes, inputs, outputs)
        "Reshape" -> Reshape(attributes, inputs, outputs)
        "Shape" -> Shape(attributes, inputs, outputs)
        "Sigmoid" -> Sigmoid(attributes, inputs, outputs)
        "SkipLayerNormalization" -> SkipLayerNormalization(attributes, inputs, outputs)
        "Slice" -> Slice(attributes, inputs, outputs)
        "Softmax" -> Softmax(attributes, inputs, outputs)
        "Split" -> Split(attributes, inputs, outputs)
        "SplitToSequence" -> SplitToSequence(attributes, inputs, outputs)
        "Squeeze" -> Squeeze(attributes, inputs, outputs)
        "Tanh" -> Tanh(attributes, inputs, outputs)
        "Transpose" -> Transpose(attributes, inputs, outputs)
        "Unsqueeze" -> Unsqueeze(attributes, inputs, outputs)
        else -> error("Unsupported operator: $name")
    } as Operator<ONNXData, ONNXData>

    fun create(proto: NodeProto) = create(proto.op_type, proto.attribute.map { Attribute.create(it) }.associateBy(Attribute<Any>::name), proto.input, proto.output)
}
