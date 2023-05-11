package io.kinference.tfjs.operators.tensor

import io.kinference.attribute.Attribute
import io.kinference.data.ONNXData
import io.kinference.graph.Contexts
import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.*
import io.kinference.operator.*
import io.kinference.protobuf.message.AttributeProto
import io.kinference.protobuf.message.TensorProto
import io.kinference.tfjs.data.tensors.TFJSTensor
import io.kinference.tfjs.data.tensors.asTensor
import io.kinference.utils.closeAll

sealed class GatherND(
    name: String,
    info: OperatorInfo,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : Operator<TFJSTensor, TFJSTensor>(name, info, attributes, inputs, outputs) {
    companion object {
        private val DEFAULT_VERSION = VersionInfo(sinceVersion = 11)

        operator fun invoke(name: String, version: Int?, attributes: Map<String, Attribute<Any>>, inputs: List<String>, outputs: List<String>): GatherND {
            return when (version ?: DEFAULT_VERSION.sinceVersion) {
                in GatherNDVer11.VERSION.asRange() -> GatherNDVer11(name, attributes, inputs, outputs)
                else -> error("Unsupported version of GatherND operator: $version")
            }
        }
    }
}

class GatherNDVer11(
    name: String,
    attributes: Map<String, Attribute<Any>>,
    inputs: List<String>,
    outputs: List<String>
) : GatherND(name, INFO, attributes, inputs, outputs) {
    companion object {
        private val ATTRIBUTES_INFO = listOf(AttributeInfo("batch_dims", setOf(AttributeProto.AttributeType.INT), false, 0L))

        private val INPUTS_INFO = listOf(
            IOInfo(0, ALL_DATA_TYPES, "data", optional = false, differentiable = true),
            IOInfo(1, setOf(TensorProto.DataType.INT64), "indices", optional = false, differentiable = false),
        )

        private val OUTPUTS_INFO = listOf(IOInfo(0, ALL_DATA_TYPES, "output", optional = false))

        internal val VERSION = VersionInfo(sinceVersion = 11)
        private val INFO = OperatorInfo("GatherND", ATTRIBUTES_INFO, INPUTS_INFO, OUTPUTS_INFO, VERSION, OperatorInfo.DEFAULT_DOMAIN)

        private suspend fun indicesMeshgrid(indicesShape: IntArray, batchDims: Int): Array<NDArrayTFJS> {
            return tidyNDArrays {
                val gridIndices = Array(batchDims) {
                    NDArrayTFJS.intRange(start = 0, stop = indicesShape[it], step = 1)
                }
                val sizes = IntArray(batchDims) { gridIndices[it].linearSize }

                val unsqueezed = Array(batchDims) { i ->
                    val newShape = IntArray(batchDims) { if (it == i) -1 else 1 }
                    gridIndices[i].reshape(newShape)
                }
                Array(batchDims) {
                    val array = unsqueezed[it]
                    val targetShape = broadcastShape(listOf(array.shape, sizes))
                    array.broadcastTo(targetShape.toTypedArray())
                }
            }
        }

        private suspend fun NDArrayTFJS.gatherNDWithBatchDims(indices: NDArrayTFJS, batchDims: Int): NDArrayTFJS {
            val indicesShape = indices.shape
            val indicesRank = indices.rank

            return tidyNDArray {
                val batchIndicesGrid = indicesMeshgrid(indicesShape, batchDims)
                val batchIndices = batchIndicesGrid.stack(axis = -1)

                val batchShape = IntArray(indicesRank)
                val batchTileRepeats = IntArray(indicesRank)
                for (i in 0 until indicesRank - 1) {
                    if (i < batchDims) {
                        batchShape[i] = indicesShape[i]
                        batchTileRepeats[i] = 1
                    } else {
                        batchShape[i] = 1
                        batchTileRepeats[i] = indicesShape[i]
                    }
                }
                batchShape[indicesRank - 1] = batchDims
                batchTileRepeats[indicesRank - 1] = 1
                val batchIndicesFull = batchIndices.reshape(batchShape).tile(batchTileRepeats)

                val actualIndices = batchIndicesFull.concat(listOf(indices), axis = -1)
                this.gatherNd(actualIndices)
            }
        }
    }

    private val batchDims: Int by attribute("batch_dims") { it: Number -> it.toInt() }

    override suspend fun <D : ONNXData<*, *>> apply(contexts: Contexts<D>, inputs: List<TFJSTensor?>): List<TFJSTensor?> {
        val input = inputs[0]!!.data
        val indices = inputs[1]!!.data

        require(indices.rank > batchDims) {
            "Indices tensor rank should be less than batch_dims. Indices rank=${indices.rank}, batch_dims=${batchDims}"
        }

        val output = if (batchDims == 0) {
            input.gatherNd(indices)
        } else {
            input.gatherNDWithBatchDims(indices, batchDims)
        }
        return listOf(output.asTensor("output"))
    }
}
