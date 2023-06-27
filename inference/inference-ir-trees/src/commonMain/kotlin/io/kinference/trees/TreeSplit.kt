package io.kinference.trees

enum class TreeSplitType {
    BRANCH_GT,
    BRANCH_GTE,
    BRANCH_LT,
    BRANCH_LEQ
}

sealed class TreeSplitter(val featureIds: IntArray, val nodeSplitValues: FloatArray) {
    abstract fun split(input: FloatArray, srcIdx: Int, splitIdx: Int): Int

    class GTTreeSplitter(featureIds: IntArray, nodeSplitValues: FloatArray): TreeSplitter(featureIds, nodeSplitValues) {
        override fun split(input: FloatArray, srcIdx: Int, splitIdx: Int): Int {
            return if (input[srcIdx + featureIds[splitIdx]] > nodeSplitValues[splitIdx]) 1 else 0
        }
    }

    class GTETreeSplitter(featureIds: IntArray, nodeSplitValues: FloatArray): TreeSplitter(featureIds, nodeSplitValues) {
        override fun split(input: FloatArray, srcIdx: Int, splitIdx: Int): Int {
            return if (input[srcIdx + featureIds[splitIdx]] >= nodeSplitValues[splitIdx]) 1 else 0
        }
    }

    class LTTreeSplitter(featureIds: IntArray, nodeSplitValues: FloatArray): TreeSplitter(featureIds, nodeSplitValues) {
        override fun split(input: FloatArray, srcIdx: Int, splitIdx: Int): Int {
            return if (input[srcIdx + featureIds[splitIdx]] < nodeSplitValues[splitIdx]) 1 else 0
        }
    }

    class LEQTreeSplitter(featureIds: IntArray, nodeSplitValues: FloatArray): TreeSplitter(featureIds, nodeSplitValues) {
        override fun split(input: FloatArray, srcIdx: Int, splitIdx: Int): Int {
            return if (input[srcIdx + featureIds[splitIdx]] <= nodeSplitValues[splitIdx]) 1 else 0
        }
    }
    
    companion object {
        fun get(splitType: TreeSplitType, featureIds: IntArray, nodeSplitValues: FloatArray): TreeSplitter {
            return when (splitType) {
                TreeSplitType.BRANCH_GT -> GTTreeSplitter(featureIds, nodeSplitValues)
                TreeSplitType.BRANCH_GTE -> GTETreeSplitter(featureIds, nodeSplitValues)
                TreeSplitType.BRANCH_LT -> LTTreeSplitter(featureIds, nodeSplitValues)
                TreeSplitType.BRANCH_LEQ -> LEQTreeSplitter(featureIds, nodeSplitValues)
            }
        }
    }
}
