package io.kinference.core.optimizer.rules

import io.kinference.core.KIONNXData
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.operators.layer.attention.*
import io.kinference.core.operators.quantization.DynamicQuantizeLinear
import io.kinference.graph.Graph
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.tryDequantize
import io.kinference.operator.Operator
import io.kinference.optimizer.*

object DequantizeQAttention : OptimizerRule<KIONNXData<*>>("Dequantize QAttention", type = RuleType.MERGE) {
    override fun shouldApply(graph: Graph<KIONNXData<*>>, name: String): Boolean {
        val op = graph.operators.indexOfFirst { it.name == name }
        return op != -1 && graph.findPath(listOf("DynamicQuantizeLinear", "QAttention"), op) != null
    }

    private fun dequantizeQAttention(graph: Graph<KIONNXData<*>>, op: QAttention, inputs: MutableList<String>): Attention {
        val weightsQuant = graph.findInitializer(op.inputs[1])!!
        val weightsScale = graph.findInitializer(op.inputs[4])!!.data as FloatNDArray
        val weightsZero = graph.findInitializer(op.inputs[7])!!.data as NumberNDArrayCore
        val numHeads = op.getAttribute<Number>("num_heads").toInt()

        val weights = (weightsQuant.data as NumberNDArrayCore).tryDequantize(weightsZero, weightsScale).asTensor("${PREFIX}_${weightsQuant.name}")
        graph.addInitializer(weights)
        graph.addInitializer(AttentionContext.prepareWeights(weights, numHeads))
        inputs.add(1, weights.name!!)

        return Attention("${PREFIX}_$name", 1, op.attributes, inputs, op.outputs)
    }

    override fun transform(graph: Graph<KIONNXData<*>>, name: String) {
        val quantize = graph.operators.singleOrNull { it.name == name } as DynamicQuantizeLinear
        val opIdx = graph.operators.indexOfFirst { it.name == name }
        val path = graph.findPath(listOf("DynamicQuantizeLinear", "QAttention"), opIdx)!!
        val qAttention = path[1] as QAttention

        val inputs = ArrayList<String>(AttentionVer1.INPUTS_INFO.size)
        inputs.add(quantize.inputs[0])
        inputs.add(qAttention.inputs[2])
        inputs.add(qAttention.inputs[5])
        inputs.add(qAttention.inputs[8])

        val attention = dequantizeQAttention(graph, qAttention, inputs)

        graph.mergeOperators(listOf(quantize.name, qAttention.name), attention as Operator<KIONNXData<*>, KIONNXData<*>>)
    }
}
