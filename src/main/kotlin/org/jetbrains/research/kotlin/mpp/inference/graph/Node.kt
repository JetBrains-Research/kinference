package org.jetbrains.research.kotlin.mpp.inference.graph

import NodeProto
import org.jetbrains.research.kotlin.mpp.inference.operators.Operator
import org.jetbrains.research.kotlin.mpp.inference.tensors.Tensor
import org.jetbrains.research.kotlin.mpp.inference.types.resolveKClass
import kotlin.reflect.KClass

class Node(proto: NodeProto) {
    val inputs: NodeIO = NodeIO()
    val outputs: NodeIO = NodeIO()
    private val operatorName = proto.op_type!!

    init {
        proto.input.forEach { inputs.addValue(it) }
        proto.output.forEach { outputs.addValue(it) }
    }

    private val mutableInputs: HashSet<String> = HashSet()

    fun setMutable(inputMarks: Set<String>) {
        mutableInputs += inputMarks
    }

    private fun clearMutableInputs() {
        mutableInputs.forEach { inputs[it] = null }
    }

    fun process(): NodeIO {
        outputs.clearValues()

        val out = Operator(operatorName, inputs.values.resolveType(), inputs.values.requireNoNulls().toList()).toList()
        outputs.keys.forEachIndexed { i, name -> outputs[name] = out.getOrNull(i) }

        clearMutableInputs()
        return outputs
    }

    companion object {
        private fun Collection<Tensor<*>?>.resolveType(): TensorProto.DataType? {
            require(this.toHashSet().size <= 1)
            return this.first()?.type
        }
    }
}
