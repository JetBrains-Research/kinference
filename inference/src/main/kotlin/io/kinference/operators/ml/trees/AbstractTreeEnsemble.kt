package io.kinference.operators.ml.trees

import io.kinference.ndarray.arrays.NDArray
import io.kinference.ndarray.arrays.NumberNDArray

abstract class AbstractTreeEnsemble(
    protected val aggregator: Aggregator,
    protected val transform: PostTransform,
    protected val treeDepths: IntArray,
    protected val treeSizes: IntArray,
    protected val featureIds: IntArray,
    protected val nodeFloatSplits: FloatArray,
    protected val leafValues: FloatArray,
    protected val biases: FloatArray,
    protected val numTargets: Int
) {

    protected fun FloatArray.computeSplit(srcIdx: Int, splitIdx: Int): Int {
        return if (this[srcIdx + featureIds[splitIdx]] > nodeFloatSplits[splitIdx]) 1 else 0
    }

    abstract fun execute(input: NumberNDArray): NDArray
}
