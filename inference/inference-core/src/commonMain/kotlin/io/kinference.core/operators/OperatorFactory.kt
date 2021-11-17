package io.kinference.core.operators

import io.kinference.core.KIONNXData
import io.kinference.core.attributes.Attribute
import io.kinference.core.model.KIModel
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
    fun create(name: String?, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (name) {
        "Add" -> Add(version, attributes, inputs, outputs)
        "Sub" -> Sub(version, attributes, inputs, outputs)
        "Attention" -> Attention(version, attributes, inputs, outputs)
        "BiasGelu" -> BiasGelu(version, attributes, inputs, outputs)
        "Cast" -> Cast(version, attributes, inputs, outputs)
        "Concat" -> Concat(version, attributes, inputs, outputs)
        "ConcatFromSequence" -> ConcatFromSequence(version, attributes, inputs, outputs)
        "Constant" -> Constant(version, attributes, inputs, outputs)
        "ConstantOfShape" -> ConstantOfShape(version, attributes, inputs, outputs)
        "CumSum" -> CumSum(version, attributes, inputs, outputs)
        "DequantizeLinear" -> DequantizeLinear(version, attributes, inputs, outputs)
        "Div" -> Div(version, attributes, inputs, outputs)
        "DynamicQuantizeLinear" -> DynamicQuantizeLinear(version, attributes, inputs, outputs)
        "DynamicQuantizeLSTM" -> DynamicQuantizeLSTM(version, attributes, inputs, outputs)
        "DynamicQuantizeMatMul" -> DynamicQuantizeMatMul(version, attributes, inputs, outputs)
        "EmbedLayerNormalization" -> EmbedLayerNormalization(version, attributes, inputs, outputs)
        "Equal" -> Equal(version, attributes, inputs, outputs)
        "Erf" -> Erf(version, attributes, inputs, outputs)
        "Expand" -> Expand(version, attributes, inputs, outputs)
        "ReduceSum" -> ReduceSum(version, attributes, inputs, outputs)
        "FastGelu" -> FastGelu(version, attributes, inputs, outputs)
        "Flatten" -> Flatten(version, attributes, inputs, outputs)
        "FusedMatMul" -> FusedMatMul(version, attributes, inputs, outputs)
        "Gather" -> Gather(version, attributes, inputs, outputs)
        "GatherElements" -> GatherElements(attributes, inputs, outputs)
        "GatherND" -> GatherND(version, attributes, inputs, outputs)
        "Gelu" -> Gelu(version, attributes, inputs, outputs)
        "Gemm" -> Gemm(version, attributes, inputs, outputs)
        "Greater" -> Greater(version, attributes, inputs, outputs)
        "GRU" -> GRU(version, attributes, inputs, outputs)
        "Identity" -> Identity(version, attributes, inputs, outputs)
        "If" -> If(version, attributes, inputs, outputs)
        "LSTM" -> LSTM(version, attributes, inputs, outputs)
        "Loop" -> Loop(version, attributes, inputs, outputs)
        "LayerNormalization" -> LayerNormalization(version, attributes, inputs, outputs)
        "LeakyRelu" -> LeakyRelu(version, attributes, inputs, outputs)
        "Log" -> Log(version, attributes, inputs, outputs)
        "LogSoftmax" -> LogSoftmax(version, attributes, inputs, outputs)
        "MatMul" -> MatMul(version, attributes, inputs, outputs)
        "MatMulInteger" -> MatMulInteger(version, attributes, inputs, outputs)
        "MatMulIntegerToFloat" -> MatMulIntegerToFloat(version, attributes, inputs, outputs)
        "Mul" -> Mul(version, attributes, inputs, outputs)
        "NonZero" -> NonZero(version, attributes, inputs, outputs)
        "Not" -> Not(version, attributes, inputs, outputs)
        "Or" -> Or(version, attributes, inputs, outputs)
        "QAttention" -> QAttention(version, attributes, inputs, outputs)
        "QEmbedLayerNormalization" -> QEmbedLayerNormalization(attributes, inputs, outputs)
        "Relu" -> Relu(version, attributes, inputs, outputs)
        "Reshape" -> Reshape(version, attributes, inputs, outputs)
        "ScatterElements" -> ScatterElements(version, attributes, inputs, outputs)
        "ScatterND" -> ScatterND(version, attributes, inputs, outputs)
        "Shape" -> Shape(version, attributes, inputs, outputs)
        "Sigmoid" -> Sigmoid(version, attributes, inputs, outputs)
        "SkipLayerNormalization" -> SkipLayerNormalization(version, attributes, inputs, outputs)
        "Slice" -> Slice(version, attributes, inputs, outputs)
        "Softmax" -> Softmax(version, attributes, inputs, outputs)
        "Split" -> Split(version, attributes, inputs, outputs)
        "SplitToSequence" -> SplitToSequence(version, attributes, inputs, outputs)
        "Squeeze" -> Squeeze(version, attributes, inputs, outputs)
        "Tanh" -> Tanh(version, attributes, inputs, outputs)
        "TopK" -> TopK(version, attributes, inputs, outputs)
        "Transpose" -> Transpose(version, attributes, inputs, outputs)
        "TreeEnsembleClassifier" -> TreeEnsembleClassifier(version, attributes, inputs, outputs)
        "TreeEnsembleRegressor" -> TreeEnsembleRegressor(version, attributes, inputs, outputs)
        "Tile" -> Tile(version, attributes, inputs, outputs)
        "Pad" -> Pad(version, attributes, inputs, outputs)
        "Unsqueeze" -> Unsqueeze(version, attributes, inputs, outputs)
        "Where" -> Where(version, attributes, inputs, outputs)
        "ZipMap" -> ZipMap(version, attributes, inputs, outputs)
        else -> error("Unsupported operator: $name")
    } as Operator<KIONNXData<*>, KIONNXData<*>>

    fun create(proto: NodeProto, opSetRegistry: KIModel.OperatorSetRegistry): Operator<KIONNXData<*>, KIONNXData<*>> {
        val version = opSetRegistry.getVersion(proto.domain)
        return create(proto.opType, version, proto.attribute.associate { it.name!! to Attribute.create(it, opSetRegistry) }, proto.input, proto.output)
    }
}
