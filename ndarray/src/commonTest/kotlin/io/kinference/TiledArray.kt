package io.kinference

import io.kinference.ndarray.Strides

class TiledArray(val strides: Strides, init: () -> Float) {
    companion object {
        const val MIN_BLOCK_SIZE = 1024
    }

    val size: Int = strides.linearSize
    val blockSize: Int
    val blocksNum: Int
    val blocksInRow: Int
    val blocks: Array<FloatArray>

    init {
        val rowSize = strides.shape.last()
        blockSize = if (rowSize < MIN_BLOCK_SIZE) rowSize else {
            var num = rowSize / MIN_BLOCK_SIZE
            while (rowSize % num != 0) num--
            rowSize / num
        }

        blocksInRow = rowSize / blockSize
        blocksNum = size / blockSize
        blocks = Array(blocksNum) { FloatArray(blockSize) }

        val colSize = strides.shape[strides.shape.lastIndex - 1]
        for (row in 0 until colSize) {
            for (col in 0 until blocksInRow) {
                val block = blocks[row + col * colSize]
                for (idx in block.indices) {
                    block[idx] = init()
                }
            }
        }
    }

    fun toArray(): FloatArray {
        val rowSize = strides.shape[strides.shape.lastIndex]
        val colSize = strides.shape[strides.shape.lastIndex - 1]

        val array = FloatArray(rowSize * colSize)

        var counter = 0
        for (row in 0 until colSize) {
            for (col in 0 until blocksInRow) {
                val block = blocks[row + col * colSize]
                for (idx in block.indices) {
                    array[counter] = block[idx]
                    counter++
                }
            }
        }

        return array
    }
}
