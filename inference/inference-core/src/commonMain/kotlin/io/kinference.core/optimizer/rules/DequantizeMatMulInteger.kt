package io.kinference.core.optimizer.rules

import io.kinference.core.KIONNXData
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.operators.math.*
import io.kinference.core.operators.quantization.DynamicQuantizeLinear
import io.kinference.graph.Graph
import io.kinference.ndarray.arrays.*
import io.kinference.operator.Operator
import io.kinference.optimizer.*

fun IntNDArray.toFloatNDArray(): FloatNDArray {
    val intPointer = this.array.pointer()
    return FloatNDArray(this.shape) { intPointer.getAndIncrement().toFloat() }
}
object DequantizeMatMulInteger : OptimizerRule<KIONNXData<*>>(name = "Dequantize MatMulInteger") {
    override fun shouldApply(graph: Graph<KIONNXData<*>>, name: String): Boolean {
        graph.operators.singleOrNull() { it.name == name } ?: return false
        val op = graph.operators.indexOfFirst { it.name == name }
        val firstPath = graph.findPath(listOf("DynamicQuantizeLinear", "Mul", "Mul"), op)
        val secondPath = graph.findPath(listOf("DynamicQuantizeLinear", "MatMulInteger", "Cast", "Mul"), op)
        return firstPath != null && secondPath != null && firstPath.last().idx == secondPath.last().idx
    }

    fun dequantizeMatMulInteger(graph: Graph<KIONNXData<*>>, dynamicQLin: DynamicQuantizeLinear, matMulInt: MatMulInteger, mul: Mul, nextMul: Mul): MatMul {
        val matmulB = graph.findInitializer(matMulInt.inputs[1])!!
        val matmulZeroB = graph.findInitializer(matMulInt.inputs[3])
        val scaleB = graph.findInitializer(mul.inputs[1])!!.data as NumberNDArray
        val dequantB = (matmulB.data as NumberNDArray).withZeroPoint(matmulZeroB!!.data as NumberNDArray).toFloatNDArray().times(scaleB).asTensor("optimized_${matmulB.name}")
        graph.addInitializer(dequantB)
        return MatMul("MatMul_${matMulInt.name}", 1, emptyMap(), dynamicQLin.inputs + dequantB.name!!, nextMul.outputs)
    }

    override fun transform(graph: Graph<KIONNXData<*>>, name: String) {
        val opIdx = graph.operators.indexOfFirst { it.name == name }
        val firstPath = graph.findPath(listOf("DynamicQuantizeLinear", "Mul", "Mul"), opIdx)!!.map { it.operator }
        val secondPath = graph.findPath(listOf("DynamicQuantizeLinear", "MatMulInteger", "Cast", "Mul"), opIdx)!!.map { it.operator }
        val operator = dequantizeMatMulInteger(graph, secondPath[0] as DynamicQuantizeLinear, secondPath[1] as MatMulInteger, firstPath[1] as Mul, firstPath[2] as Mul)
        val names = firstPath.map { it.name } + secondPath.dropLast(1).map { it.name }
        graph.mergeOperators(names, operator as Operator<KIONNXData<*>, KIONNXData<*>>)
    }
}
