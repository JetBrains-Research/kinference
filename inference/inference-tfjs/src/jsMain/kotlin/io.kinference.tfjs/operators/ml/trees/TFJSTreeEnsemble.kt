package io.kinference.tfjs.operators.ml.trees

import io.kinference.ndarray.arrays.*
import io.kinference.ndarray.extensions.dataFloat
import io.kinference.primitives.types.DataType
import io.kinference.trees.*

class TFJSTreeEnsemble(
    aggregator: Aggregator,
    transform: PostTransform,
    treeDepths: IntArray,
    treeSizes: IntArray,
    featureIds: IntArray,
    nodeFloatSplits: FloatArray,
    nonLeafValuesCount: IntArray,
    leafValues: FloatArray,
    biases: FloatArray,
    numTargets: Int,
    splitMode: TreeSplitType
) : SingleModeTreeEnsemble<NumberNDArrayTFJS>(
    aggregator, transform, treeDepths, treeSizes, featureIds, nodeFloatSplits,
    nonLeafValuesCount, leafValues, biases, numTargets, splitMode
) {
    override suspend fun execute(input: NumberNDArrayTFJS): NumberNDArrayTFJS {
        require(input.type == DataType.DOUBLE || input.type == DataType.FLOAT) { "Integer inputs are not supported yet" }

        val inputArray = input.dataFloat()
        val n = if (input.rank == 1) 1 else input.shape[0]
        val outputShape = if (numTargets == 1) arrayOf(n) else arrayOf(n, numTargets)
        val outputArray = FloatArray(n * numTargets)

        val leadDim = input.shape[0]

        if (input.rank == 1 || leadDim == 1) {
            applyEntry(inputArray, outputArray)
        } else {
            val stride = input.shape[1]
            for (i in 0 until leadDim) {
                applyEntry(inputArray, outputArray, srcIdx = i * stride, dstIdx = i * numTargets)
            }
        }
        val output = NDArrayTFJS.float(outputArray, outputShape).asMutable()
        return transform.apply(output) as NumberNDArrayTFJS
    }

    companion object {
        fun fromInfo(info: TreeEnsembleInfo): TFJSTreeEnsemble {
            return TFJSTreeEnsemble(
                aggregator = info.aggregator,
                transform = info.transform,
                treeDepths = info.treeDepths,
                treeSizes = info.treeSizes,
                featureIds = info.featureIds,
                nodeFloatSplits = info.nodeFloatSplits,
                nonLeafValuesCount = info.nonLeafValuesCount,
                leafValues = info.leafValues,
                biases = info.biases,
                numTargets = info.numTargets,
                splitMode = info.splitMode
            )
        }
    }
}
