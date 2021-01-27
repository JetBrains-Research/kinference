package io.kinference.operators.ml

import io.kinference.attributes.Attribute
import io.kinference.data.tensors.Tensor
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.arrays.pointers.DoublePointer
import io.kinference.onnx.TensorProto
import io.kinference.operators.*

abstract class TreeEnsembleOperator(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>)
    : Operator<Tensor, Tensor>(info, attributes, inputs, outputs) {
    companion object {
        //TODO: Support integer types
        private val TYPE_CONSTRAINTS = setOf(
            //TensorProto.DataType.INT32,
            //TensorProto.DataType.INT64,
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.DOUBLE
        )

        internal val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "X", optional = false)
        )

        internal fun NDArray.toFloatNDArray() = if (this is FloatNDArray) {
            this
        } else {
            require(this is DoubleNDArray)
            val pointer = DoublePointer(this.array)
            FloatNDArray(this.shape) { pointer.getAndIncrement().toFloat() }
        }
    }

    @Suppress("PropertyName")
    open class BaseEnsembleInfo(val map: Map<String, Any?>) {
        val aggregate_function: String? by map
        val base_values: List<Number>? by map
        val nodes_falsenodeids: List<Number> by map
        val nodes_featureids: List<Number> by map
        val nodes_modes: List<String> by map
        val nodes_nodeids: List<Number> by map
        val nodes_treeids: List<Number> by map
        val nodes_truenodeids: List<Number> by map
        val nodes_values: List<Number> by map
        val post_transform: String? by map
    }
}
