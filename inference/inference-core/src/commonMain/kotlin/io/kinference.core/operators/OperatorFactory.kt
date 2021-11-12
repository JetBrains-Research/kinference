package io.kinference.core.operators

import io.kinference.core.KIONNXData
import io.kinference.core.attributes.Attribute
import io.kinference.core.operators.activations.*
import io.kinference.core.operators.flow.*
import io.kinference.core.operators.layer.attention.Attention
import io.kinference.core.operators.layer.attention.QAttention
import io.kinference.core.operators.layer.normalization.*
import io.kinference.core.operators.layer.recurrent.gru.GRU
import io.kinference.core.operators.layer.recurrent.lstm.LSTM
import io.kinference.core.operators.logical.*
import io.kinference.core.operators.math.*
import io.kinference.core.operators.ml.*
import io.kinference.core.operators.quantization.*
import io.kinference.core.operators.seq.ConcatFromSequence
import io.kinference.core.operators.seq.SplitToSequence
import io.kinference.core.operators.tensor.*
import io.kinference.core.operators.quantization.lstm.DynamicQuantizeLSTM
import io.kinference.protobuf.message.NodeProto
import kotlin.time.ExperimentalTime

@ExperimentalTime
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
        "DequantizeLinear" -> DequantizeLinear(attributes, inputs, outputs)
        "Div" -> Div(attributes, inputs, outputs)
        "DynamicQuantizeLinear" -> DynamicQuantizeLinear(attributes, inputs, outputs)
        "DynamicQuantizeLSTM" -> DynamicQuantizeLSTM(attributes, inputs, outputs)
        "DynamicQuantizeMatMul" -> DynamicQuantizeMatMul(attributes, inputs, outputs)
        "EmbedLayerNormalization" -> EmbedLayerNormalization(attributes, inputs, outputs)
        "Equal" -> Equal(attributes, inputs, outputs)
        "Erf" -> Erf(attributes, inputs, outputs)
        "Expand" -> Expand(attributes, inputs, outputs)
        "ReduceSum" -> ReduceSum(attributes, inputs, outputs)
        "FastGelu" -> FastGelu(attributes, inputs, outputs)
        "Flatten" -> Flatten(attributes, inputs, outputs)
        "FusedMatMul" -> FusedMatMul(attributes, inputs, outputs)
        "Gather" -> Gather(attributes, inputs, outputs)
        "GatherElements" -> GatherElements(attributes, inputs, outputs)
        "GatherND" -> GatherND(attributes, inputs, outputs)
        "Gelu" -> Gelu(attributes, inputs, outputs)
        "Gemm" -> Gemm(attributes, inputs, outputs)
        "Greater" -> Greater(attributes, inputs, outputs)
        "GRU" -> GRU(attributes, inputs, outputs)
        "Identity" -> Identity(attributes, inputs, outputs)
        "If" -> If(attributes, inputs, outputs)
        "LSTM" -> LSTM(attributes, inputs, outputs)
        "Loop" -> Loop(attributes, inputs, outputs)
        "LayerNormalization" -> LayerNormalization(attributes, inputs, outputs)
        "LeakyRelu" -> LeakyRelu(attributes, inputs, outputs)
        "Log" -> Log(attributes, inputs, outputs)
        "LogSoftmax" -> LogSoftmax(attributes, inputs, outputs)
        "MatMul" -> MatMul(attributes, inputs, outputs)
        "MatMulInteger" -> MatMulInteger(attributes, inputs, outputs)
        "MatMulIntegerToFloat" -> MatMulIntegerToFloat(attributes, inputs, outputs)
        "Mul" -> Mul(attributes, inputs, outputs)
        "NonZero" -> NonZero(attributes, inputs, outputs)
        "Not" -> Not(attributes, inputs, outputs)
        "Or" -> Or(attributes, inputs, outputs)
        "QAttention" -> QAttention(attributes, inputs, outputs)
        "Relu" -> Relu(attributes, inputs, outputs)
        "Reshape" -> Reshape(attributes, inputs, outputs)
        "ScatterElements" -> ScatterElements(attributes, inputs, outputs)
        "ScatterND" -> ScatterND(attributes, inputs, outputs)
        "Shape" -> Shape(attributes, inputs, outputs)
        "Sigmoid" -> Sigmoid(attributes, inputs, outputs)
        "SkipLayerNormalization" -> SkipLayerNormalization(attributes, inputs, outputs)
        "Slice" -> Slice(attributes, inputs, outputs)
        "Softmax" -> Softmax(attributes, inputs, outputs)
        "Split" -> Split(attributes, inputs, outputs)
        "SplitToSequence" -> SplitToSequence(attributes, inputs, outputs)
        "Squeeze" -> Squeeze(attributes, inputs, outputs)
        "Tanh" -> Tanh(attributes, inputs, outputs)
        "TopK" -> TopK(attributes, inputs, outputs)
        "Transpose" -> Transpose(attributes, inputs, outputs)
        "TreeEnsembleClassifier" -> TreeEnsembleClassifier(attributes, inputs, outputs)
        "TreeEnsembleRegressor" -> TreeEnsembleRegressor(attributes, inputs, outputs)
        "Pad" -> Pad(attributes, inputs, outputs)
        "Unsqueeze" -> Unsqueeze(attributes, inputs, outputs)
        "Where" -> Where(attributes, inputs, outputs)
        "ZipMap" -> ZipMap(attributes, inputs, outputs)
        else -> error("Unsupported operator: $name")
    } as Operator<KIONNXData<*>, KIONNXData<*>>

    fun create(proto: NodeProto) = create(proto.opType, proto.attribute.map { Attribute.create(it) }.associateBy(Attribute<Any>::name), proto.input, proto.output)
}
