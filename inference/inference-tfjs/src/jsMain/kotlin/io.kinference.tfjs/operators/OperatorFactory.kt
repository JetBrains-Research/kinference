package io.kinference.tfjs.operators

import io.kinference.protobuf.message.NodeProto
import io.kinference.tfjs.TFJSData
import io.kinference.tfjs.attributes.Attribute
import io.kinference.tfjs.operators.layer.attention.Attention
import io.kinference.tfjs.operators.layer.attention.QAttention
import io.kinference.tfjs.operators.layer.normalization.*
import io.kinference.tfjs.operators.math.*
import io.kinference.tfjs.operators.quantization.DequantizeLinear
import io.kinference.tfjs.operators.quantization.DynamicQuantizeLinear
import io.kinference.tfjs.operators.tensor.*

object OperatorFactory {
    @Suppress("UNCHECKED_CAST")
    fun create(opType: String?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (opType) {
        "Attention" -> Attention(attributes, inputs, outputs)
        "Add" -> Add(attributes, inputs, outputs)
        "Shape" -> Shape(attributes, inputs, outputs)
        "ConstantOfShape" -> ConstantOfShape(attributes, inputs, outputs)
        "Cast" -> Cast(attributes, inputs, outputs)
        "EmbedLayerNormalization" -> EmbedLayerNormalization(attributes, inputs, outputs)
        "MatMul" -> MatMul(attributes, inputs, outputs)
        "SkipLayerNormalization" -> SkipLayerNormalization(attributes, inputs, outputs)
        "BiasGelu" -> BiasGelu(attributes, inputs, outputs)
        "DequantizeLinear" -> DequantizeLinear(attributes, inputs, outputs)
        "DynamicQuantizeLinear" -> DynamicQuantizeLinear(attributes, inputs, outputs)
        "LayerNormalization" -> LayerNormalization(attributes, inputs, outputs)
        "QAttention" -> QAttention(attributes, inputs, outputs)
        "Gather" -> Gather(attributes, inputs, outputs)
        "Unsqueeze" -> Unsqueeze(attributes, inputs, outputs)
        "Concat" -> Concat(attributes, inputs, outputs)
        "Reshape" -> Reshape(attributes, inputs, outputs)
        "Mul" -> Mul(attributes, inputs, outputs)
        "FastGelu" -> FastGelu(attributes, inputs, outputs)
        "MatMulInteger" -> MatMulInteger(attributes, inputs, outputs)
        "Slice" -> Slice(attributes, inputs, outputs)
        "Squeeze" -> Squeeze(attributes, inputs, outputs)
        else -> error("Unsupported operator: $opType")
    } as Operator<TFJSData<*>, TFJSData<*>>

    fun create(proto: NodeProto) = create(proto.opType, proto.attribute.map { Attribute.create(it) }.associateBy(Attribute<Any>::name), proto.input, proto.output)
}
