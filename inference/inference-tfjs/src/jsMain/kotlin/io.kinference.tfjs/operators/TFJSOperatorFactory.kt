package io.kinference.tfjs.operators

import io.kinference.attribute.Attribute
import io.kinference.attribute.AttributeFactory
import io.kinference.graph.Graph
import io.kinference.operator.*
import io.kinference.protobuf.message.GraphProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.graph.TFJSGraph
import io.kinference.tfjs.operators.activations.*
import io.kinference.tfjs.operators.flow.*
import io.kinference.tfjs.operators.layer.attention.Attention
import io.kinference.tfjs.operators.layer.attention.QAttention
import io.kinference.tfjs.operators.layer.normalization.*
import io.kinference.tfjs.operators.layer.recurrent.gru.GRU
import io.kinference.tfjs.operators.layer.recurrent.lstm.LSTM
import io.kinference.tfjs.operators.logical.*
import io.kinference.tfjs.operators.math.*
import io.kinference.tfjs.operators.ml.*
import io.kinference.tfjs.operators.quantization.DequantizeLinear
import io.kinference.tfjs.operators.quantization.DynamicQuantizeLinear
import io.kinference.tfjs.operators.seq.*
import io.kinference.tfjs.operators.tensor.*

object TFJSAttributeFactory : AttributeFactory<TFJSData<*>> {
    override fun createTensor(proto: TensorProto): TFJSData<*> = TFJSTensor.create(proto)
    override suspend fun createGraph(proto: GraphProto, opSet: OperatorSetRegistry): Graph<TFJSData<*>> = TFJSGraph(proto, opSet)
}

object TFJSOperatorFactory : OperatorFactory<TFJSData<*>> {
    override fun attributeFactory(): AttributeFactory<TFJSData<*>> = TFJSAttributeFactory

    @Suppress("UNCHECKED_CAST")
    override fun create(name: String, opType: String?, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (opType) {
        "Abs" -> Abs(name, version, attributes, inputs, outputs)
        "Acos" -> Acos(name, version, attributes, inputs, outputs)
        "Acosh" -> Acosh(name, version, attributes, inputs, outputs)
        "Add" -> Add(name, version, attributes, inputs, outputs)
        "And" -> And(name, version, attributes, inputs, outputs)
        "ArgMax" -> ArgMax(name, version, attributes, inputs, outputs)
        "ArgMin" -> ArgMin(name, version, attributes, inputs, outputs)
        "Asin" -> Asin(name, version, attributes, inputs, outputs)
        "Asinh" -> Asinh(name, version, attributes, inputs, outputs)
        "Atan" -> Atan(name, version, attributes, inputs, outputs)
        "Atanh" -> Atanh(name, version, attributes, inputs, outputs)
        "Attention" -> Attention(name, version, attributes, inputs, outputs)
        "BiasGelu" -> BiasGelu(name, version, attributes, inputs, outputs)
        "Cast" -> Cast(name, version, attributes, inputs, outputs)
        "CastLike" -> CastLike(name, version, attributes, inputs, outputs)
        "Ceil" -> Ceil(name, version, attributes, inputs, outputs)
        "Celu" -> Celu(name, version, attributes, inputs, outputs)
        "Concat" -> Concat(name, version, attributes, inputs, outputs)
        "ConcatFromSequence" -> ConcatFromSequence(name, version, attributes, inputs, outputs)
        "Constant" -> Constant(name, version, attributes, inputs, outputs)
        "ConstantOfShape" -> ConstantOfShape(name, version, attributes, inputs, outputs)
        "Cos" -> Cos(name, version, attributes, inputs, outputs)
        "Cosh" -> Cosh(name, version, attributes, inputs, outputs)
        "CumSum" -> CumSum(name, version, attributes, inputs, outputs)
        "DequantizeLinear" -> DequantizeLinear(name, version, attributes, inputs, outputs)
        "Det" -> Det(name, version, attributes, inputs, outputs)
        "Div" -> Div(name, version, attributes, inputs, outputs)
        "DynamicQuantizeLinear" -> DynamicQuantizeLinear(name, version, attributes, inputs, outputs)
        "EmbedLayerNormalization" -> EmbedLayerNormalization(name, version, attributes, inputs, outputs)
        "Elu" -> Elu(name, version, attributes, inputs, outputs)
        "Equal" -> Equal(name, version, attributes, inputs, outputs)
        "Erf" -> Erf(name, version, attributes, inputs, outputs)
        "Exp" -> Exp(name, version, attributes, inputs, outputs)
        "Expand" -> Expand(name, version, attributes, inputs, outputs)
        "FastGelu" -> FastGelu(name, version, attributes, inputs, outputs)
        "Flatten" -> Flatten(name, version, attributes, inputs, outputs)
        "Floor" -> Floor(name, version, attributes, inputs, outputs)
        "FusedMatMul" -> FusedMatMul(name, version, attributes, inputs, outputs)
        "Gather" -> Gather(name, version, attributes, inputs, outputs)
        "GatherElements" -> GatherElements(name, version, attributes, inputs, outputs)
        "GatherND" -> GatherND(name, version, attributes, inputs, outputs)
        "Gemm" -> Gemm(name, version, attributes, inputs, outputs)
        "Greater" -> Greater(name, version, attributes, inputs, outputs)
        "GRU" -> GRU(name, version, attributes, inputs, outputs)
        "Hardmax" -> Hardmax(name, version, attributes, inputs, outputs)
        "IsInf" -> IsInf(name, version, attributes, inputs, outputs)
        "IsNaN" -> IsNaN(name, version, attributes, inputs, outputs)
        "Identity" -> Identity(name, version, attributes, inputs, outputs)
        "If" -> If(name, version, attributes, inputs, outputs)
        "LayerNormalization" -> LayerNormalization(name, version, attributes, inputs, outputs)
        "LeakyRelu" -> LeakyRelu(name, version, attributes, inputs, outputs)
        "Less" -> Less(name, version, attributes, inputs, outputs)
        "Log" -> Log(name, version, attributes, inputs, outputs)
        "LogSoftmax" -> LogSoftmax(name, version, attributes, inputs, outputs)
        "Loop" -> Loop(name, version, attributes, inputs, outputs)
        "LSTM" -> LSTM(name, version, attributes, inputs, outputs)
        "MatMul" -> MatMul(name, version, attributes, inputs, outputs)
        "MatMulInteger" -> MatMulInteger(name, version, attributes, inputs, outputs)
        "Max" -> Max(name, version, attributes, inputs, outputs)
        "Mean" -> Mean(name, version, attributes, inputs, outputs)
        "Mul" -> Mul(name, version, attributes, inputs, outputs)
        "NonZero" -> NonZero(name, version, attributes, inputs, outputs)
        "Not" -> Not(name, version, attributes, inputs, outputs)
        "Or" -> Or(name, version, attributes, inputs, outputs)
        "Pad" -> Pad(name, version, attributes, inputs, outputs)
        "QAttention" -> QAttention(name, version, attributes, inputs, outputs)
        "QEmbedLayerNormalization" -> QEmbedLayerNormalization(name, version, attributes, inputs, outputs)
        "Range" -> Range(name, version, attributes, inputs, outputs)
        "ReduceSum" -> ReduceSum(name, version, attributes, inputs, outputs)
        "Relu" -> Relu(name, version, attributes, inputs, outputs)
        "Reshape" -> Reshape(name, version, attributes, inputs, outputs)
        "ScatterElements" -> ScatterElements(name, version, attributes, inputs, outputs)
        "ScatterND" -> ScatterND(name, version, attributes, inputs, outputs)
        "SequenceInsert" -> SequenceInsert(name, version, attributes, inputs, outputs)
        "SequenceLength" -> SequenceLength(name, version, attributes, inputs, outputs)
        "Shape" -> Shape(name, version, attributes, inputs, outputs)
        "Sigmoid" -> Sigmoid(name, version, attributes, inputs, outputs)
        "Sign" -> Sign(name, version, attributes, inputs, outputs)
        "Sin" -> Sin(name, version, attributes, inputs, outputs)
        "Sinh" -> Sinh(name, version, attributes, inputs, outputs)
        "Size" -> Size(name, version, attributes, inputs, outputs)
        "SkipLayerNormalization" -> SkipLayerNormalization(name, version, attributes, inputs, outputs)
        "Slice" -> Slice(name, version, attributes, inputs, outputs)
        "Softmax" -> Softmax(name, version, attributes, inputs, outputs)
        "Split" -> Split(name, version, attributes, inputs, outputs)
        "SplitToSequence" -> SplitToSequence(name, version, attributes, inputs, outputs)
        "Sqrt" -> Sqrt(name, version, attributes, inputs, outputs)
        "Squeeze" -> Squeeze(name, version, attributes, inputs, outputs)
        "Sub" -> Sub(name, version, attributes, inputs, outputs)
        "Sum" -> Sum(name, version, attributes, inputs, outputs)
        "Tan" -> Tan(name, version, attributes, inputs, outputs)
        "Tanh" -> Tanh(name, version, attributes, inputs, outputs)
        "Tile" -> Tile(name, version, attributes, inputs, outputs)
        "Transpose" -> Transpose(name, version, attributes, inputs, outputs)
        "TreeEnsembleClassifier" -> TreeEnsembleClassifier(name, version, attributes, inputs, outputs)
        "TreeEnsembleRegressor" -> TreeEnsembleRegressor(name, version, attributes, inputs, outputs)
        "Trilu" -> Trilu(name, version, attributes, inputs, outputs)
        "Unsqueeze" -> Unsqueeze(name, version, attributes, inputs, outputs)
        "Where" -> Where(name, version, attributes, inputs, outputs)
        "Xor" -> Xor(name, version, attributes, inputs, outputs)
        "ZipMap" -> ZipMap(name, version, attributes, inputs, outputs)
        else -> error("Unsupported operator: $opType")
    } as Operator<TFJSData<*>, TFJSData<*>>
}
