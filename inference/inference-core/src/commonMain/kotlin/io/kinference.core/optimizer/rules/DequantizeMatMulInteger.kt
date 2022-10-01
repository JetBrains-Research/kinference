package io.kinference.core.optimizer.rules

import io.kinference.core.KIONNXData
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.operators.math.*
import io.kinference.core.operators.quantization.DynamicQuantizeLinear
import io.kinference.graph.Graph
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.tryZeroPoint
import io.kinference.operator.Operator
import io.kinference.optimizer.*

fun IntNDArray.toFloatNDArray(): FloatNDArray {
    val intPointer = this.array.pointer()
    return FloatNDArray(this.shape) { intPointer.getAndIncrement().toFloat() }
}
object DequantizeMatMulInteger : OptimizerRule<KIONNXData<*>>(name = "Dequantize MatMulInteger", type = RuleType.MERGE) {
    override fun shouldApply(graph: Graph<KIONNXData<*>>, name: String): Boolean {
        val op = graph.operators.indexOfFirst { it.name == name }
        if (op == -1) return false

        val firstPath = graph.findPath(listOf("DynamicQuantizeLinear", "Mul", "Mul"), op) ?: return false
        val secondPath = graph.findPath(listOf("DynamicQuantizeLinear", "MatMulInteger", "Cast", "Mul"), op) ?: return false
        return firstPath.first().name == secondPath.first().name && firstPath.last().name == secondPath.last().name
    }

    fun dequantizeMatMulInteger(graph: Graph<KIONNXData<*>>, dynamicQLin: DynamicQuantizeLinear, matMulInt: MatMulInteger, mul: Mul, nextMul: Mul): MatMul? {
        val matMulRight = graph.findInitializer(matMulInt.inputs[1])
        val matMulZeroRight = graph.findInitializer(matMulInt.inputs[3])?.data as? NumberNDArrayCore
        val scaleRight = graph.findInitializer(mul.inputs[1])?.data as? NumberNDArray
        if (matMulRight?.data == null || matMulZeroRight == null || scaleRight == null) return null

        val newName = "${PREFIX}_${matMulRight.name}"
        val dequantizedRight = (matMulRight.data as NumberNDArrayCore).tryZeroPoint(matMulZeroRight).toFloatNDArray().times(scaleRight)
        graph.addInitializer(dequantizedRight.asTensor(newName))
        return MatMul(
            name = "${PREFIX}_${matMulInt.name}",
            version = 1, attributes = emptyMap(),
            inputs = dynamicQLin.inputs + newName,
            outputs = nextMul.outputs
        )
    }

    override fun transform(graph: Graph<KIONNXData<*>>, name: String) {
        val opIdx = graph.operators.indexOfFirst { it.name == name }
        val firstPath = graph.findPath(listOf("DynamicQuantizeLinear", "Mul", "Mul"), opIdx)!!
        val secondPath = graph.findPath(listOf("DynamicQuantizeLinear", "MatMulInteger", "Cast", "Mul"), opIdx)!!
        val operator = dequantizeMatMulInteger(
            graph = graph,
            dynamicQLin = secondPath[0] as DynamicQuantizeLinear,
            matMulInt = secondPath[1] as MatMulInteger,
            mul = firstPath[1] as Mul,
            nextMul = firstPath[2] as Mul
        ) ?: return
        val names = firstPath.map { it.name }.drop(1) + secondPath.dropLast(1).map { it.name }
        graph.mergeOperators(names, operator as Operator<KIONNXData<*>, KIONNXData<*>>)
    }
}
