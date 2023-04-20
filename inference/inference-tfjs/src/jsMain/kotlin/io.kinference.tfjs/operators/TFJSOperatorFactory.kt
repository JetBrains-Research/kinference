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
import io.kinference.tfjs.operators.activations.Softmax
import io.kinference.tfjs.operators.flow.Loop
import io.kinference.tfjs.operators.flow.Where
import io.kinference.tfjs.operators.layer.attention.Attention
import io.kinference.tfjs.operators.layer.attention.QAttention
import io.kinference.tfjs.operators.layer.normalization.*
import io.kinference.tfjs.operators.logical.*
import io.kinference.tfjs.operators.math.*
import io.kinference.tfjs.operators.quantization.DequantizeLinear
import io.kinference.tfjs.operators.quantization.DynamicQuantizeLinear
import io.kinference.tfjs.operators.seq.ConcatFromSequence
import io.kinference.tfjs.operators.seq.SplitToSequence
import io.kinference.tfjs.operators.tensor.*

object TFJSAttributeFactory : AttributeFactory<TFJSData<*>> {
    override fun createTensor(proto: TensorProto): TFJSData<*> = TFJSTensor.create(proto)
    override suspend fun createGraph(proto: GraphProto, opSet: OperatorSetRegistry): Graph<TFJSData<*>> = TFJSGraph(proto, opSet)
}

object TFJSOperatorFactory : OperatorFactory<TFJSData<*>> {
    override fun attributeFactory(): AttributeFactory<TFJSData<*>> = TFJSAttributeFactory

    @Suppress("UNCHECKED_CAST")
    override fun create(name: String, opType: String?, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (opType) {
        "Add" -> Add(name, version, attributes, inputs, outputs)
        "ArgMax" -> ArgMax(name, version, attributes, inputs, outputs)
        "Attention" -> Attention(name, version, attributes, inputs, outputs)
        "BiasGelu" -> BiasGelu(name, version, attributes, inputs, outputs)
        "Cast" -> Cast(name, version, attributes, inputs, outputs)
        "Concat" -> Concat(name, version, attributes, inputs, outputs)
        "ConcatFromSequence" -> ConcatFromSequence(name, version, attributes, inputs, outputs)
        "Constant" -> Constant(name, version, attributes, inputs, outputs)
        "ConstantOfShape" -> ConstantOfShape(name, version, attributes, inputs, outputs)
        "CumSum" -> CumSum(name, version, attributes, inputs, outputs)
        "DequantizeLinear" -> DequantizeLinear(name, version, attributes, inputs, outputs)
        "Div" -> Div(name, version, attributes, inputs, outputs)
        "DynamicQuantizeLinear" -> DynamicQuantizeLinear(name, version, attributes, inputs, outputs)
        "EmbedLayerNormalization" -> EmbedLayerNormalization(name, version, attributes, inputs, outputs)
        "Equal" -> Equal(name, version, attributes, inputs, outputs)
        "Expand" -> Expand(name, version, attributes, inputs, outputs)
        "FastGelu" -> FastGelu(name, version, attributes, inputs, outputs)
        "FusedMatMul" -> FusedMatMul(name, version, attributes, inputs, outputs)
        "Gather" -> Gather(name, version, attributes, inputs, outputs)
        "GatherElements" -> GatherElements(name, version, attributes, inputs, outputs)
        "LayerNormalization" -> LayerNormalization(name, version, attributes, inputs, outputs)
        "LeakyRelu" -> LeakyRelu(name, version, attributes, inputs, outputs)
        "Less" -> Less(name, version, attributes, inputs, outputs)
        "Loop" -> Loop(name, version, attributes, inputs, outputs)
        "MatMul" -> MatMul(name, version, attributes, inputs, outputs)
        "MatMulInteger" -> MatMulInteger(name, version, attributes, inputs, outputs)
        "Mul" -> Mul(name, version, attributes, inputs, outputs)
        "Not" -> Not(name, version, attributes, inputs, outputs)
        "QAttention" -> QAttention(name, version, attributes, inputs, outputs)
        "QEmbedLayerNormalization" -> QEmbedLayerNormalization(name, version, attributes, inputs, outputs)
        "Range" -> Range(name, version, attributes, inputs, outputs)
        "Reshape" -> Reshape(name, version, attributes, inputs, outputs)
        "Shape" -> Shape(name, version, attributes, inputs, outputs)
        "SkipLayerNormalization" -> SkipLayerNormalization(name, version, attributes, inputs, outputs)
        "Slice" -> Slice(name, version, attributes, inputs, outputs)
        "Softmax" -> Softmax(name, version, attributes, inputs, outputs)
        "SplitToSequence" -> SplitToSequence(name, version, attributes, inputs, outputs)
        "Squeeze" -> Squeeze(name, version, attributes, inputs, outputs)
        "Tile" -> Tile(name, version, attributes, inputs, outputs)
        "Transpose" -> Transpose(name, version, attributes, inputs, outputs)
        "Unsqueeze" -> Unsqueeze(name, version, attributes, inputs, outputs)
        "Where" -> Where(name, version, attributes, inputs, outputs)
        else -> error("Unsupported operator: $opType")
    } as Operator<TFJSData<*>, TFJSData<*>>
}
