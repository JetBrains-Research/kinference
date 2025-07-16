@file:Suppress("UnusedImport")

package io.kinference.ndarray

import io.kinference.ndarray.arrays.DoubleNDArray
import io.kinference.ndarray.arrays.DoubleLNDArray
import io.kinference.ndarray.arrays.MutableDoubleNDArray
import io.kinference.ndarray.arrays.MutableDoubleLNDArray
import io.kinference.ndarray.arrays.FloatNDArray
import io.kinference.ndarray.arrays.FloatLNDArray
import io.kinference.ndarray.arrays.MutableFloatNDArray
import io.kinference.ndarray.arrays.MutableFloatLNDArray
import io.kinference.ndarray.arrays.MutablePrimitiveNDArray
import io.kinference.ndarray.extensions.MIN_VALUE_FOR_MAX
import io.kinference.ndarray.math.FastMath
import io.kinference.ndarray.math.exp
import io.kinference.ndarray.stubs.MIN_VALUE_FOR_MAX
import io.kinference.ndarray.stubs.max
import io.kinference.primitives.types.PrimitiveType
import jdk.incubator.vector.*
import kotlin.math.min
import io.kinference.ndarray.arrays.applyInPlace
import io.kinference.ndarray.arrays.BinaryVecOp
import io.kinference.ndarray.arrays.reduce

internal suspend fun dotVectorParallel(left: DoubleNDArray, right: DoubleNDArray, dest: MutableDoubleNDArray): MutableDoubleNDArray {
    val n = left.shape[0]
    val t = left.shape[1]
    val m = right.shape[1]

    val lBlocksInRow = left.blocksInRow
    val rdBlocksInRow = right.blocksInRow

    val leftBlocks = left.array.blocks
    val rightBlocks = right.array.blocks
    val destBlocks = dest.array.blocks

    val lBlockSize = left.array.blockSize

    val nRowFlop = t * m
    val spec = DoubleVector.SPECIES_PREFERRED
    val vecLen = spec.length()

    // Constant 261120 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByRows(nRowFlop, n, 500000) { nStart, nEnd, _ ->

        for (i in 0 until n) {
            val leftBlockOffset = i * lBlocksInRow
            val destBlockOffset = i * rdBlocksInRow
            var rightBlockIndex = 0

            for (lCol in 0 until lBlocksInRow) {
                val leftBlock = leftBlocks[leftBlockOffset + lCol]

                for (k in 0 until lBlockSize) {
                    val temp = leftBlock[k]

                    for (rdCol in 0 until rdBlocksInRow) {
                        val destBlock = destBlocks[destBlockOffset + rdCol]
                        val rightBlock = rightBlocks[rightBlockIndex++]
                        val vS = destBlock.size - destBlock.size % vecLen
                        for (idx in 0 until vS step vecLen) {
                            DoubleVector.fromArray(spec, rightBlock, idx)
                                .mul(temp).add(DoubleVector.fromArray(spec, destBlock, idx))
                                .intoArray(destBlock, idx)
                        }
                        for (j in vS until destBlock.size) {
                            destBlock[j] = (destBlock[j] + temp * rightBlock[j])
                        }

                        //for (j in destBlock.indices) {
                        //    destBlock[j] = (destBlock[j] + temp * rightBlock[j]).toDouble()
                        //}
                    }
                }
            }
        }
    }

    return dest
}

suspend fun dotLV(left: DoubleLNDArray, right: DoubleLNDArray, dest: MutableDoubleLNDArray): MutableDoubleLNDArray {

    val n = left.shape[0]
    val t = left.shape[1]
    val m = right.shape[1]
    val nRowFlop = t * m

    val destArray = dest.array
    val leftArray = left.array
    val rightArray = right.array

    val spec = DoubleVector.SPECIES_PREFERRED
    val vecLen = spec.length()
    val vecN = m - (m % vecLen)

    parallelizeByRows(nRowFlop, n, 1500000) { rowStart, rowEnd, _ ->
        for (rowIdx in rowStart until rowEnd) {
            val rowOffset = rowIdx * t
            val destOffset = rowIdx * m
            val row = leftArray.slice(rowOffset until (rowOffset + t))
            for (colIdx in 0 until vecN step vecLen) {
                var dVec = DoubleVector.broadcast(spec, 0.0)
                for (i in 0 until t) {
                    val tmp = row[i]
                    dVec = dVec.add(DoubleVector.fromArray(spec, rightArray, i * m + colIdx).mul(tmp))
                }
                dVec.intoArray(destArray, destOffset + colIdx)
            }
            for (colIdx in vecN until m) {
                for (i in 0 until t) {
                    destArray[destOffset + colIdx] += row[i] * rightArray[i * m + colIdx]
                }
            }
        }
    }

    return dest
}

suspend fun dotVectorChunked(left: DoubleLNDArray, right: DoubleLNDArray, dest: MutableDoubleLNDArray): MutableDoubleLNDArray {
    val n = left.shape[0]
    val t = left.shape[1]
    val m = right.shape[1]
    val numChunks = 20
    val lChunkSize = (n + numChunks - 1) / numChunks
    val rChunkSize = (m + numChunks - 1) / numChunks
    val leftChunks: Array<DoubleArray> = Array(numChunks) { i: Int ->
        left.array.sliceArray(i * lChunkSize * t until t * min((i + 1) * lChunkSize, n))
    }
    val rightChunks: Array<DoubleArray> = Array(numChunks) { i: Int ->
        val colStart = rChunkSize * i
        val colEnd = min(colStart + rChunkSize, m)
        DoubleArray((colEnd - colStart) * t) { j: Int ->
            right.array[(j % t) * m + j / t]
        }
    }

    val spec = DoubleVector.SPECIES_PREFERRED
    val vecLen = spec.length()
    val vecT = t - (t % vecLen)

    parallelizeByRows(n, numChunks, 20) { chunkIdx, chunkEnd, _ ->
        for (shift in 0 until numChunks) {
            val lChunk = leftChunks[chunkIdx]
            val rChunkId = (chunkIdx + shift) % numChunks
            val rChunk = rightChunks[rChunkId]
            for (i in 0 until lChunk.size step t) {
                for (j in 0 until rChunk.size step t) {
                    val destIdx = (chunkIdx * lChunkSize + i / t) * m + rChunkId * rChunkSize + j / t
                    var vecSum = DoubleVector.broadcast(spec, 0.0)
                    for (k in 0 until vecT step vecLen) {
                        vecSum = vecSum.add(
                            DoubleVector.fromArray(spec, lChunk, i + k)
                                .mul(DoubleVector.fromArray(spec, rChunk, j + k))
                        )
                    }
                    for (k in vecT until t) {
                        dest.array[destIdx] += lChunk[i + k] * rChunk[j + k]
                    }
                }
            }

        }
    }

    return dest
}

suspend fun dotVectorChunked(left: FloatLNDArray, right: FloatLNDArray, dest: MutableFloatLNDArray): MutableFloatLNDArray {
    val n = left.shape[0]
    val t = left.shape[1]
    val m = right.shape[1]
    val numChunks = 20
    val lChunkSize = (n + numChunks - 1) / numChunks
    val rChunkSize = (m + numChunks - 1) / numChunks
    val leftChunks: Array<FloatArray> = Array(numChunks) { i: Int ->
        left.array.sliceArray(i * t * lChunkSize until t * min((i + 1) * lChunkSize, n))
    }
    val rightChunks: Array<FloatArray> = Array(numChunks) { i: Int ->
        val colStart = rChunkSize * i
        val colEnd = min(colStart + rChunkSize, m)
        FloatArray((colEnd - colStart) * t) { j: Int ->
            right.array[(j % t) * m + j / t]
        }
    }

    val spec = FloatVector.SPECIES_PREFERRED
    val vecLen = spec.length()
    val vecT = t - (t % vecLen)

    parallelizeByRows(n, numChunks, 20) { chunkIdx, chunkEnd, _ ->
        for (shift in 0 until numChunks) {
            val lChunk = leftChunks[chunkIdx]
            val rChunkId = (chunkIdx + shift) % numChunks
            val rChunk = rightChunks[rChunkId]
            for (i in 0 until lChunk.size step t) {
                for (j in 0 until rChunk.size step t) {
                    val destIdx = (chunkIdx * lChunkSize + i / t) * m + rChunkId * rChunkSize + j / t
                    var vecSum = FloatVector.broadcast(spec, 0F)
                    for (k in 0 until vecT step vecLen) {
                        vecSum = vecSum.add(
                            FloatVector.fromArray(spec, lChunk, i + k)
                                .mul(FloatVector.fromArray(spec, rChunk, j + k))
                        )
                    }
                    for (k in vecT until t) {
                        dest.array[destIdx] += lChunk[i + k] * rChunk[j + k]
                    }
                }
            }
        }
    }

    return dest
}

internal suspend fun dotVectorParallel(left: FloatNDArray, right: FloatNDArray, dest: MutableFloatNDArray): MutableFloatNDArray {
    val n = left.shape[0]
    val t = left.shape[1]
    val m = right.shape[1]

    val lBlocksInRow = left.blocksInRow
    val rdBlocksInRow = right.blocksInRow

    val leftBlocks = left.array.blocks
    val rightBlocks = right.array.blocks
    val destBlocks = dest.array.blocks

    val lBlockSize = left.array.blockSize

    val nRowFlop = t * m
    val spec = FloatVector.SPECIES_PREFERRED
    val vecLen = spec.length()

    // Constant 261120 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByRows(nRowFlop, n, 500000) { nStart, nEnd, _ ->

        for (i in 0 until n) {
            val leftBlockOffset = i * lBlocksInRow
            val destBlockOffset = i * rdBlocksInRow
            var rightBlockIndex = 0

            for (lCol in 0 until lBlocksInRow) {
                val leftBlock = leftBlocks[leftBlockOffset + lCol]

                for (k in 0 until lBlockSize) {
                    val temp = leftBlock[k]

                    for (rdCol in 0 until rdBlocksInRow) {
                        val destBlock = destBlocks[destBlockOffset + rdCol]
                        val rightBlock = rightBlocks[rightBlockIndex++]
                        val vS = destBlock.size - destBlock.size % vecLen
                        for (idx in 0 until vS step vecLen) {
                            FloatVector.fromArray(spec, rightBlock, idx)
                                .mul(temp).add(FloatVector.fromArray(spec, destBlock, idx))
                                .intoArray(destBlock, idx)
                        }
                        for (j in vS until destBlock.size) {
                            destBlock[j] = (destBlock[j] + temp * rightBlock[j])
                        }

                        //for (j in destBlock.indices) {
                        //    destBlock[j] = (destBlock[j] + temp * rightBlock[j]).toFloat()
                        //}
                    }
                }
            }
        }
    }

    return dest
}

suspend fun dotLV(left: FloatLNDArray, right: FloatLNDArray, dest: MutableFloatLNDArray): MutableFloatLNDArray {

    val n = left.shape[0]
    val t = left.shape[1]
    val m = right.shape[1]
    val nRowFlop = t * m

    val destArray = dest.array
    val leftArray = left.array
    val rightArray = right.array

    val spec = FloatVector.SPECIES_PREFERRED
    val vecLen = spec.length()
    val vecN = m - (m % vecLen)

    parallelizeByRows(nRowFlop, n, 500000) { rowStart, rowEnd, _ ->
        for (rowIdx in rowStart until rowEnd) {
            val rowOffset = rowIdx * t
            val destOffset = rowIdx * m
            for (i in 0 until t) {
                val tmp = leftArray[rowOffset + i]
                val rightOffset = i * m
                for (colIdx in 0 until vecN step vecLen) {
                    FloatVector.fromArray(spec, rightArray, rightOffset + colIdx)
                        .mul(tmp)
                        .add(FloatVector.fromArray(spec, destArray, destOffset + colIdx))
                        .intoArray(destArray, destOffset + colIdx)
                }
                for (colIdx in vecN until m) {
                    destArray[destOffset + colIdx] = (destArray[destOffset + colIdx] + tmp * rightArray[rightOffset + colIdx])
                }
            }
        }
    }

    return dest
}

suspend fun vecSoftmax(input: FloatLNDArray, dest: MutableFloatLNDArray, rows: Int, columns: Int): MutableFloatLNDArray {
    val outputArray = dest.array
    val inputArray = input.array

    //Finding Max for each block
    // Constant 65536 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    val species = FloatVector.SPECIES_PREFERRED
    val vecLen = species.length()
    val vecN = columns - (columns % vecLen)
    parallelizeByRows(columns, rows, 65536) { rowStart, rowEnd, _ ->
        for (rowIdx in rowStart until rowEnd) {
            var rowMax = Float.MIN_VALUE
            val rowOffset = rowIdx * columns
            var vecMax = FloatVector.broadcast(species, Float.MIN_VALUE)
            for (i in 0 until vecN step vecLen) {
                vecMax = vecMax.max(FloatVector.fromArray(species, inputArray, rowOffset + i))
            }
            rowMax = vecMax.reduceLanes(VectorOperators.MAX)
            for (i in vecN until columns) rowMax = maxOf(inputArray[rowOffset + i], rowMax)

            for (i in 0 until vecN step vecLen) {
                FloatVector.fromArray(species, inputArray, rowOffset + i)
                    .sub(rowMax)
                    .intoArray(outputArray, rowOffset + i)
            }
            for (i in vecN until columns) outputArray[rowOffset + i] = (inputArray[rowOffset + i] - rowMax)
        }
    }

    // Apply exp for output array
    // Constant 2048 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByRows(columns, rows, 65536) { rowStart, rowEnd, _ ->
        for (rowIdx in rowStart until rowEnd) {
            val rowOffset = rowIdx * columns
            for (i in 0 until vecN step vecLen) {
                FloatVector.fromArray(species, outputArray, rowOffset + i)
                    .lanewise(VectorOperators.EXP)
                    .intoArray(outputArray, rowOffset + i)
            }
            for (i in vecN until columns) {
                outputArray[rowOffset + i] = FastMath.exp(outputArray[rowOffset + i])
            }
        }
    }

    parallelizeByRows(columns, rows, 65536) { rowStart, rowEnd, _ ->
        for (rowIdx in rowStart until rowEnd) {
            val rowOffset = rowIdx * columns
            var rowSum = FloatVector.broadcast(species, 0F)
            for (i in 0 until vecN step vecLen) {
                rowSum = rowSum.add(FloatVector.fromArray(species, outputArray, rowOffset + i))
            }
            var sum = rowSum.reduceLanes(VectorOperators.ADD)
            for (i in vecN until columns) {
                sum += outputArray[rowOffset + i]
            }
            for (i in 0 until vecN step vecLen) {
                FloatVector.fromArray(species, outputArray, rowOffset + i)
                    .div(rowSum)
                    .intoArray(outputArray, rowOffset + i)
            }
            for (i in vecN until columns) {
                outputArray[rowOffset + i] /= sum
            }
        }
    }
    return dest
}

internal suspend fun vecBlkSoftmax(input: FloatNDArray, dest: MutableFloatNDArray, rows: Int, columns: Int): MutableFloatNDArray {
    val inputBlockSize = input.array.blockSize
    val inputBlocks = input.array.blocks

    val outputArray = dest.array
    val maxesArray = FloatArray(inputBlocks.size)

    //Finding Max for each block
    // Constant 65536 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    val spec = FloatVector.SPECIES_PREFERRED
    val vecLen = spec.length()
    val vecN = inputBlockSize - (inputBlockSize % vecLen)
    parallelizeByBlocks(inputBlockSize, inputBlocks.size, 65536) { blockStart, blockEnd, _ ->
        for (blockNum in blockStart until blockEnd) {
            val inputBlock = inputBlocks[blockNum]
            maxesArray[blockNum] = reduce(inputBlock, 0, VectorOperators.MAX, inputBlockSize)
        }
    }

    val blocksInRow = columns / inputBlockSize

    //Minus maximum from input and store in output
    // Constant 1048576 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByRows(columns, rows, 1048576) { rowStart, rowEnd, _ ->
        for (rowNum in rowStart until rowEnd) {
            val rowBlockStart = rowNum * blocksInRow
            var localMax: Float = Float.MIN_VALUE_FOR_MAX.toFloat()
            for (rowBlockIdx in rowBlockStart until rowBlockStart + blocksInRow) {
                localMax = maxOf(localMax, maxesArray[rowBlockIdx])
            }

            for (rowBlockIdx in rowBlockStart until rowBlockStart + blocksInRow) {
                val inputBlock = inputBlocks[rowBlockIdx]
                val outputBlock = outputArray.blocks[rowBlockIdx]
                for (j in 0 until vecN step vecLen) {
                    FloatVector.fromArray(spec, inputBlock, j).sub(localMax).intoArray(outputBlock, j)
                }
                for (j in vecN until inputBlockSize) {
                    outputBlock[j] = inputBlock[j] - localMax
                }
            }
        }
    }

    // Apply exp for output array
    // Constant 2048 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByBlocks(inputBlockSize, inputBlocks.size, 2048) { blockStart, blockEnd, _ ->
        for (blockNum in blockStart until blockEnd) {
            val outputBlock = outputArray.blocks[blockNum]
            for (j in 0 until vecN step vecLen) {
                FloatVector.fromArray(spec, outputBlock, j).lanewise(VectorOperators.EXP).intoArray(outputBlock, j)
            }
            for (j in vecN until inputBlockSize) {
                outputBlock[j] = FastMath.exp(outputBlock[j])
            }
        }
    }

    val sumsArray = FloatArray(outputArray.blocks.size)

    // Calculate sum for each block
    // Constant 131072 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByBlocks(inputBlockSize, inputBlocks.size, 131072) { blockStart, blockEnd, _ ->
        for (blockNum in blockStart until blockEnd) {
            var cumSum = FloatVector.broadcast(spec, 0F)
            for (j in 0 until vecN step vecLen) {
                cumSum = cumSum.add(FloatVector.fromArray(spec, outputArray.blocks[blockNum], j))
            }
            sumsArray[blockNum] = cumSum.reduceLanes(VectorOperators.ADD)
            for (j in vecN until inputBlockSize) {
                sumsArray[blockNum] += outputArray.blocks[blockNum][j]
            }
        }
    }

    // Div by sum in output array
    // Constant 1048576 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByRows(columns, rows, 1048576) { rowStart, rowEnd, _ ->
        for (rowNum in rowStart until rowEnd) {
            val rowBlockStart = rowNum * blocksInRow
            var localSum = (0).toFloat()
            for (rowBlockIdx in rowBlockStart until rowBlockStart + blocksInRow) {
                localSum += sumsArray[rowBlockIdx]
            }

            for (rowBlockIdx in rowBlockStart until rowBlockStart + blocksInRow) {
                val outputBlock = outputArray.blocks[rowBlockIdx]
                for (j in 0 until vecN step vecLen) {
                    FloatVector.fromArray(spec, outputBlock, j).div(localSum).intoArray(outputBlock, j)
                }
                for (j in vecN until inputBlockSize) {
                    outputBlock[j] /= localSum
                }
            }
        }
    }

    return dest
}


suspend fun vecSoftmax(input: DoubleLNDArray, dest: MutableDoubleLNDArray, rows: Int, columns: Int): MutableDoubleLNDArray {
    val outputArray = dest.array
    val inputArray = input.array

    //Finding Max for each block
    // Constant 65536 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    val species = DoubleVector.SPECIES_PREFERRED
    val vecLen = species.length()
    val vecN = columns - (columns % vecLen)
    parallelizeByRows(columns, rows, 65536) { rowStart, rowEnd, _ ->
        for (rowIdx in rowStart until rowEnd) {
            var rowMax = Double.MIN_VALUE
            val rowOffset = rowIdx * columns
            var vecMax = DoubleVector.broadcast(species, Double.MIN_VALUE)
            for (i in 0 until vecN step vecLen) {
                vecMax = vecMax.max(DoubleVector.fromArray(species, inputArray, rowOffset + i))
            }
            rowMax = vecMax.reduceLanes(VectorOperators.MAX)
            for (i in vecN until columns) rowMax = maxOf(inputArray[rowOffset + i], rowMax)

            for (i in 0 until vecN step vecLen) {
                DoubleVector.fromArray(species, inputArray, rowOffset + i)
                    .sub(rowMax)
                    .intoArray(outputArray, rowOffset + i)
            }
            for (i in vecN until columns) outputArray[rowOffset + i] = (inputArray[rowOffset + i] - rowMax)
        }
    }

    // Apply exp for output array
    // Constant 2048 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByRows(columns, rows, 65536) { rowStart, rowEnd, _ ->
        for (rowIdx in rowStart until rowEnd) {
            val rowOffset = rowIdx * columns
            for (i in 0 until vecN step vecLen) {
                DoubleVector.fromArray(species, outputArray, rowOffset + i)
                    .lanewise(VectorOperators.EXP)
                    .intoArray(outputArray, rowOffset + i)
            }
            for (i in vecN until columns) {
                outputArray[rowOffset + i] = FastMath.exp(outputArray[rowOffset + i])
            }
        }
    }

    parallelizeByRows(columns, rows, 65536) { rowStart, rowEnd, _ ->
        for (rowIdx in rowStart until rowEnd) {
            val rowOffset = rowIdx * columns
            var rowSum = DoubleVector.broadcast(species, 0.0)
            for (i in 0 until vecN step vecLen) {
                rowSum = rowSum.add(DoubleVector.fromArray(species, outputArray, rowOffset + i))
            }
            var sum = rowSum.reduceLanes(VectorOperators.ADD)
            for (i in vecN until columns) {
                sum += outputArray[rowOffset + i]
            }
            for (i in 0 until vecN step vecLen) {
                DoubleVector.fromArray(species, outputArray, rowOffset + i)
                    .div(rowSum)
                    .intoArray(outputArray, rowOffset + i)
            }
            for (i in vecN until columns) {
                outputArray[rowOffset + i] /= sum
            }
        }
    }
    return dest
}

suspend fun vecSoftmaxHelper(input: FloatLNDArray, dest: MutableFloatLNDArray, rows: Int, columns: Int): MutableFloatLNDArray {
    val outputArray = dest.array
    val inputArray = input.array

    //Finding Max for each block
    // Constant 65536 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    val species = FloatVector.SPECIES_PREFERRED
    val vecLen = species.length()
    val vecN = columns - (columns % vecLen)
    parallelizeByRows(columns, rows, 65536) { rowStart, rowEnd, _ ->
        for (rowIdx in rowStart until rowEnd) {
            val rowOffset = rowIdx * columns
            val rowMax = reduce(inputArray, rowOffset, VectorOperators.MAX, columns)
            for (i in 0 until columns) outputArray[rowOffset + i] = inputArray[rowOffset + i] - rowMax
            //applyInPlace(outputArray, rowOffset, VectorOperators.SUB, rowMax, columns)
        }
    }

    // Apply exp for output array
    // Constant 2048 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByRows(columns, rows, 65536) { rowStart, rowEnd, _ ->
        for (rowIdx in rowStart until rowEnd) {
            val rowOffset = rowIdx * columns
            for (i in 0 until vecN step vecLen) {
                FloatVector.fromArray(species, outputArray, rowOffset + i)
                    .lanewise(VectorOperators.EXP)
                    .intoArray(outputArray, rowOffset + i)
            }
            for (i in vecN until columns) {
                outputArray[rowOffset + i] = FastMath.exp(outputArray[rowOffset + i])
            }
        }
    }

    parallelizeByRows(columns, rows, 65536) { rowStart, rowEnd, _ ->
        for (rowIdx in rowStart until rowEnd) {
            val rowOffset = rowIdx * columns
            val sum = reduce(outputArray, rowOffset, VectorOperators.ADD, columns)
            applyInPlace(outputArray, rowOffset, VectorOperators.DIV, sum, columns)
        }
    }
    return dest
}

suspend fun vecSoftmaxClass(input: FloatLNDArray, dest: MutableFloatLNDArray, rows: Int, columns: Int): MutableFloatLNDArray {
    val outputArray = dest.array
    val inputArray = input.array

    //Finding Max for each block
    // Constant 65536 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    val species = FloatVector.SPECIES_PREFERRED
    val vecLen = species.length()
    val vecN = columns - (columns % vecLen)
    parallelizeByRows(columns, rows, 65536) { rowStart, rowEnd, _ ->
        for (rowIdx in rowStart until rowEnd) {
            val rowOffset = rowIdx * columns
            val rowMax = reduce(inputArray, rowOffset, VectorOperators.MAX, columns)
            Sub(FloatSlice(outputArray, rowOffset), Constant(rowMax))
                .resolveTo(outputArray, rowOffset, columns)
            //applyInPlace(outputArray, rowOffset, VectorOperators.SUB, rowMax, columns)
        }
    }

    // Apply exp for output array
    // Constant 2048 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByRows(columns, rows, 65536) { rowStart, rowEnd, _ ->
        for (rowIdx in rowStart until rowEnd) {
            val rowOffset = rowIdx * columns
            Exp(FloatSlice(outputArray, rowOffset))
                .resolveTo(outputArray, rowOffset, columns)
        }
    }

    parallelizeByRows(columns, rows, 65536) { rowStart, rowEnd, _ ->
        for (rowIdx in rowStart until rowEnd) {
            val rowOffset = rowIdx * columns
            val sum = reduce(outputArray, rowOffset, VectorOperators.ADD, columns)
            Add(FloatSlice(outputArray, rowOffset), Constant(sum))
                .resolveTo(outputArray, rowOffset, columns)
        }
    }
    return dest
}

internal suspend fun vecBlkSoftmax(input: DoubleNDArray, dest: MutableDoubleNDArray, rows: Int, columns: Int): MutableDoubleNDArray {
    val inputBlockSize = input.array.blockSize
    val inputBlocks = input.array.blocks

    val outputArray = dest.array
    val maxesArray = DoubleArray(inputBlocks.size)

    //Finding Max for each block
    // Constant 65536 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    val spec = DoubleVector.SPECIES_PREFERRED
    val vecLen = spec.length()
    val vecN = inputBlockSize - (inputBlockSize % vecLen)
    parallelizeByBlocks(inputBlockSize, inputBlocks.size, 65536) { blockStart, blockEnd, _ ->
        for (blockNum in blockStart until blockEnd) {
            val inputBlock = inputBlocks[blockNum]
            var maxVec = DoubleVector.broadcast(spec, Double.MIN_VALUE_FOR_MAX.toDouble())
            for (idx in 0 until vecN step vecLen) {
                maxVec = maxVec.max(DoubleVector.fromArray(spec, inputBlock, idx))
            }
            maxesArray[blockNum] = maxVec.reduceLanes(VectorOperators.MAX)
            for (idx in vecN until inputBlockSize) {
                maxesArray[blockNum] = maxOf(maxesArray[blockNum], inputBlock[idx])
            }
        }
    }

    val blocksInRow = columns / inputBlockSize

    //Minus maximum from input and store in output
    // Constant 1048576 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByRows(columns, rows, 128000) { rowStart, rowEnd, _ ->
        for (rowNum in rowStart until rowEnd) {
            val rowBlockStart = rowNum * blocksInRow
            var localMax: Double = Double.MIN_VALUE_FOR_MAX.toDouble()
            for (rowBlockIdx in rowBlockStart until rowBlockStart + blocksInRow) {
                localMax = maxOf(localMax, maxesArray[rowBlockIdx])
            }

            for (rowBlockIdx in rowBlockStart until rowBlockStart + blocksInRow) {
                val inputBlock = inputBlocks[rowBlockIdx]
                val outputBlock = outputArray.blocks[rowBlockIdx]
                for (j in 0 until vecN step vecLen) {
                    DoubleVector.fromArray(spec, inputBlock, j).sub(localMax).intoArray(outputBlock, j)
                }
                for (j in vecN until inputBlockSize) {
                    outputBlock[j] = inputBlock[j] - localMax
                }
            }
        }
    }

    // Apply exp for output array
    // Constant 2048 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByBlocks(inputBlockSize, inputBlocks.size, 2048) { blockStart, blockEnd, _ ->
        for (blockNum in blockStart until blockEnd) {
            val outputBlock = outputArray.blocks[blockNum]
            for (j in 0 until vecN step vecLen) {
                DoubleVector.fromArray(spec, outputBlock, j).lanewise(VectorOperators.EXP).intoArray(outputBlock, j)
            }
            for (j in vecN until inputBlockSize) {
                outputBlock[j] = FastMath.exp(outputBlock[j])
            }
        }
    }

    val sumsArray = DoubleArray(outputArray.blocks.size)

    // Calculate sum for each block
    // Constant 131072 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByBlocks(inputBlockSize, inputBlocks.size, 65536) { blockStart, blockEnd, _ ->
        for (blockNum in blockStart until blockEnd) {
            var cumSum = DoubleVector.broadcast(spec, 0.0)
            for (j in 0 until vecN step vecLen) {
                cumSum = cumSum.add(DoubleVector.fromArray(spec, outputArray.blocks[blockNum], j))
            }
            sumsArray[blockNum] = cumSum.reduceLanes(VectorOperators.ADD)
            for (j in vecN until inputBlockSize) {
                sumsArray[blockNum] += outputArray.blocks[blockNum][j]
            }
        }
    }

    // Div by sum in output array
    // Constant 1048576 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByRows(columns, rows, 1048576) { rowStart, rowEnd, _ ->
        for (rowNum in rowStart until rowEnd) {
            val rowBlockStart = rowNum * blocksInRow
            var localSum = (0).toDouble()
            for (rowBlockIdx in rowBlockStart until rowBlockStart + blocksInRow) {
                localSum += sumsArray[rowBlockIdx]
            }

            for (rowBlockIdx in rowBlockStart until rowBlockStart + blocksInRow) {
                val outputBlock = outputArray.blocks[rowBlockIdx]
                for (j in 0 until vecN step vecLen) {
                    DoubleVector.fromArray(spec, outputBlock, j).div(localSum).intoArray(outputBlock, j)
                }
                for (j in vecN until inputBlockSize) {
                    outputBlock[j] /= localSum
                }
            }
        }
    }

    return dest
}

suspend fun vecSoftmaxGenerated(input: FloatLNDArray, dest: MutableFloatLNDArray, rows: Int, columns: Int): MutableFloatLNDArray {
    val outputArray = dest.array
    val inputArray = input.array

    //Finding Max for each block
    // Constant 65536 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    val species = FloatVector.SPECIES_PREFERRED
    val vecLen = species.length()
    val vecN = columns - (columns % vecLen)
    parallelizeByRows(columns, rows, 65536) { rowStart, rowEnd, _ ->
        for (rowIdx in rowStart until rowEnd) {
            val rowOffset = rowIdx * columns
            val rowMax = reduce(inputArray, rowOffset, VectorOperators.MAX, columns)
            val end = columns - (columns % 8)
            for (idx in 0 until end step 8) {
                FloatVector.fromArray(species, outputArray, rowOffset + idx)
                    .lanewise(VectorOperators.SUB, rowMax)
                    .intoArray(outputArray, rowOffset + idx)
            }
            for (idx in end until columns) {
                outputArray[rowOffset + idx] = (outputArray[rowOffset + idx] - rowMax)
            }
        }
    }

    // Apply exp for output array
    // Constant 2048 was precomputed on M1 Max processor
    // With this constant two launches work faster than single thread without launches
    // TODO: (cupertank) Remove constants
    parallelizeByRows(columns, rows, 65536) { rowStart, rowEnd, _ ->
        for (rowIdx in rowStart until rowEnd) {
            val rowOffset = rowIdx * columns
            for (i in 0 until vecN step vecLen) {
                FloatVector.fromArray(species, outputArray, rowOffset + i)
                    .lanewise(VectorOperators.EXP)
                    .intoArray(outputArray, rowOffset + i)
            }
            for (i in vecN until columns) {
                outputArray[rowOffset + i] = FastMath.exp(outputArray[rowOffset + i])
            }
        }
    }

    parallelizeByRows(columns, rows, 65536) { rowStart, rowEnd, _ ->
        for (rowIdx in rowStart until rowEnd) {
            val rowOffset = rowIdx * columns
            val sum = reduce(outputArray, rowOffset, VectorOperators.ADD, columns)
            val end = columns - (columns % 8)
            for (idx in 0 until end step 8) {
                FloatVector.fromArray(species, outputArray, rowOffset + idx)
                    .lanewise(VectorOperators.DIV, sum)
                    .intoArray(outputArray, rowOffset + idx)
            }
            for (idx in end until columns) {
                outputArray[rowOffset + idx] = (outputArray[rowOffset + idx] / sum)
            }
        }
    }
    return dest
}
