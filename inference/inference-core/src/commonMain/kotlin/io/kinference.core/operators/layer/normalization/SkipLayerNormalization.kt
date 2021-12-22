package io.kinference.core.operators.layer.normalization

import io.kinference.attribute.Attribute
import io.kinference.core.data.tensor.KITensor
import io.kinference.core.data.tensor.asTensor
import io.kinference.core.graph.KIContext
import io.kinference.data.ONNXData
import io.kinference.graph.Context
import io.kinference.profiler.ProfilingContext
import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.arrays.MutableFloatNDArray
import io.kinference.ndarray.arrays.pointers.*
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import kotlin.math.sqrt
import kotlin.time.ExperimentalTime

sealed class SkipLayerNormalization(info: OperatorInfo, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : Operator<KITensor, KITensor>(info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 1)

        operator fun invoke(version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) = when (version ?: DEFAULT_VERSION.sinceVersion) {
            in SkipLayerNormalizationVer1.VERSION.asRange() -> SkipLayerNormalizationVer1(attributes, inputs, outputs)
            else -> error("Unsupported version of SkipLayerNormalization operator: $version")
        }
    }
}

@ExperimentalTime
class SkipLayerNormalizationVer1(attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>) : SkipLayerNormalization(INFO, attributes, inputs, outputs) {
    private val epsilon: Float by attribute()

    companion object {
        private val TYPE_CONSTRAINTS = setOf(
            TensorProto.DataType.FLOAT,
            TensorProto.DataType.FLOAT16
        )

        private val ATTRIBUTES_INFO = listOf(
            AttributeInfo("epsilon", setOf(AttributeProto.AttributeType.FLOAT), false, 0.00001f)
        )

        private val INPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "input", false),
            IOInfo(1, TYPE_CONSTRAINTS, "skip", false),
            IOInfo(2, TYPE_CONSTRAINTS, "gamma", false),
            IOInfo(3, TYPE_CONSTRAINTS, "beta", false),
            IOInfo(4, TYPE_CONSTRAINTS, "bias", true)
        )

        private val OUTPUTS_INFO = listOf(
            IOInfo(0, TYPE_CONSTRAINTS, "output", false),
            //Only for training, not supported now
            IOInfo(1, TYPE_CONSTRAINTS, "mean", true),
            IOInfo(2, TYPE_CONSTRAINTS, "inv_std_var", true)
        )

        internal val VERSION = VersionInfo(sinceVersion = 1)
        private val INFO = OperatorInfo("SkipLayerNormalization", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, domain = "com.microsoft")

        private fun FloatNDArray.normalize(
            skip: FloatNDArray,
            gamma: FloatNDArray,
            beta: FloatNDArray,
            bias: FloatNDArray?,
            epsilon: Float,
            dst: MutableFloatNDArray
        ) {
            val (batchSize, seqLen, hiddenSize) = this.shape
            val steps = batchSize * seqLen

            for (i in 0 until steps) {
                val offset = hiddenSize * i

                val dstPointer = dst.array.pointer(offset)
                val srcPointer = this.array.pointer(offset)
                val skipPointer = skip.array.pointer(offset)
                if (bias == null) {
                    dstPointer.acceptDouble(srcPointer, skipPointer, hiddenSize) { _, src, sk -> src + sk }
                } else {
                    val biasPointer = bias.array.pointer()
                    dstPointer.acceptTriple(srcPointer, skipPointer, biasPointer, hiddenSize) { _, src, sk, b -> src + sk + b }
                }

                var mean = 0.0f
                var meanSquare = 0.0f

                dstPointer.linearIndex = offset
                dstPointer.forEach(hiddenSize) {
                    mean += it
                    meanSquare += it * it
                }

                mean /= hiddenSize
                meanSquare = sqrt(meanSquare / hiddenSize - mean * mean + epsilon)

                dstPointer.linearIndex = offset
                val gammaPointer = gamma.array.pointer()
                val betaPointer = beta.array.pointer()
                dstPointer.acceptDouble(gammaPointer, betaPointer, hiddenSize) { d, g, b ->
                    (d - mean) / meanSquare * g + b
                }
            }
        }
    }


    override fun <D : ONNXData<*, *>> apply(context: Context<D>, inputs: List<KITensor?>, profilingContext: ProfilingContext?, checkCancelled: () -> Unit): List<KITensor?> {
        val input = inputs[0]!!.data as FloatNDArray
        val output = input.allocateNDArray(input.strides)
        input.normalize(
            skip = inputs[1]!!.data as FloatNDArray,
            gamma = inputs[2]!!.data as FloatNDArray,
            beta = inputs[3]!!.data as FloatNDArray,
            bias = inputs.getOrNull(4)?.data as FloatNDArray?,
            epsilon = epsilon,
            dst = output
        )
        return listOf(output.asTensor())
    }
}
