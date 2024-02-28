package io.kinference.core.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.operator.*
import io.kinference.primitives.types.DataType
import io.kinference.protobuf.message.TensorProto
import io.kinference.protobuf.resolveProtoDataType
import kotlin.math.ceil

sealed class Range(name: String, info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 11)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in RangeVer11.VERSION.asRange() -> RangeVer11(name, attributes, inputs, outputs)
            else -> error("Unsupported version of Constant operator: $version")
        }
    }
}


class RangeVer11(name: String, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Range(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.DOUBLE, TensorProto.DataType.FLOAT, TensorProto.DataType.INT16,
            TensorProto.DataType.INT32, TensorProto.DataType.INT64
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "start", optional = false, scalar = true),
            IOInfo(1, TYPE_CONSTRAINTS, "limit", optional = false, scalar = true),
            IOInfo(2, TYPE_CONSTRAINTS, "delta", optional = false, scalar = true)
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, TYPE_CONSTRAINTS, "output", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 5, untilVersion = 14)
        private val INFO = OperatorInfo("Range", emptyMap(), INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)

        private fun <T> range(type: DataType, start: T, limit: T, delta: T): NumberNDArrayCore {
            return when (type.resolveProtoDataType()) {
                TensorProto.DataType.DOUBLE -> {
                    start as Double; limit as Double; delta as Double
                    val size = ceil((limit - start) / delta).toInt()
                    DoubleNDArray(intArrayOf(size)) { start + (it.value * delta) }
                }
                TensorProto.DataType.FLOAT-> {
                    start as Float; limit as Float; delta as Float
                    val size = ceil((limit - start) / delta).toInt()
                    FloatNDArray(intArrayOf(size)) { start + (it.value * delta) }
                }
                TensorProto.DataType.INT16 -> {
                    start as Short; limit as Short; delta as Short
                    val size = ceil((limit - start).toDouble() / delta).toInt()
                    ShortNDArray(intArrayOf(size)) { (start + (it.value * delta)).toShort() }
                }
                TensorProto.DataType.INT32 -> {
                    start as Int; limit as Int; delta as Int
                    val size = ceil((limit - start).toDouble() / delta).toInt()
                    IntNDArray(intArrayOf(size)) { start + (it.value * delta) }
                }
                TensorProto.DataType.INT64 -> {
                    start as Long; limit as Long; delta as Long
                    val size = ceil((limit - start).toDouble() / delta).toInt()
                    LongNDArray(intArrayOf(size)) { start + (it.value * delta) }
                }
                else -> error("Unsupported data type: $type")
            }
        }
    }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<KITensor?>): List<KITensor?> {
        val start = inputs[0]!!.data
        val limit = inputs[1]!!.data
        val delta = inputs[2]!!.data

        require(start.type == limit.type && limit.type == delta.type) { "Start, limit and delta tensors must have the same data type" }
        require(start.isScalar() && limit.isScalar() && delta.isScalar()) { "Start, limit and delta tensors must be scalars" }

        return listOf(range(inputs[0]!!.data.type, start.singleValue(), limit.singleValue(), delta.singleValue()).asTensor("output"))
    }
}
