package io.kinference.benchmark

import io.kinference.ndarray.Strides
import org.junit.jupiter.api.Tag
import org.junit.jupiter.api.Test
import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole
import org.openjdk.jmh.runner.Runner
import org.openjdk.jmh.runner.options.OptionsBuilder
import java.nio.FloatBuffer
import java.util.concurrent.TimeUnit
import kotlin.random.Random


@State(Scope.Benchmark)
@Fork(value = 1, warmups = 0, jvmArgsAppend = [
//    "-XX:CompileThreshold=100",
//    "-XX:+UnlockDiagnosticVMOptions"
//    "-XX:CompileCommand=print,\"io.kinference/benchmark/DotBenchmark.baseline\""
])
@Warmup(iterations = 3)
@BenchmarkMode(Mode.SingleShotTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Measurement(iterations = 100)
open class DotBenchmark {
    @Param("4096", "1024", "128", "16")
    var n = 0

    @Param("4096", "1024", "128", "16")
    var m = 0

    @Param("4096", "1024", "128", "16")
    var t = 0

    lateinit var left: FloatArray
    lateinit var right: FloatArray
    lateinit var dest: FloatArray

    lateinit var array: FloatArray

    var leftSize: Int = 0
    var rightSize: Int = 0
    var destSize: Int = 0

    @Setup(Level.Iteration)
    fun setup() {
        leftSize = n * t
        rightSize = t * m
        destSize = n * m

        left = FloatArray(leftSize) { Random.nextFloat() }
        right = FloatArray(rightSize) { Random.nextFloat() }
        dest = FloatArray(destSize)

        array = FloatArray(destSize)
    }

    @TearDown(Level.Iteration)
    fun teardown() {
        for (i in 0 until destSize) {
            dest[i] = 0f
        }
    }

    @Benchmark
    fun baseline(blackhole: Blackhole) {
        dotBaseline(left, right, dest, m, n, t)
        blackhole.consume(dest)
    }

    @Benchmark
    fun baselineWithBuffer(blackhole: Blackhole) {
        val buffer = FloatBuffer.wrap(array)

        for (i in 0 until n) {
            val dInd = i * m
            val lInd = i * t
            for (k in 0 until t) {
                val temp = left[lInd + k]

                buffer.position(dInd)
                buffer.put(right, k * m, m)

                for (j in dInd until dInd + m) {
                    dest[j] += temp * array[j]
                }
            }
        }

        blackhole.consume(dest)
    }
}

@State(Scope.Benchmark)
@Fork(value = 1, warmups = 0)
@Warmup(iterations = 3)
@BenchmarkMode(Mode.SingleShotTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Measurement(iterations = 100)
open class DotBenchmarkComposite {
    @Param("4096", "1024", "128", "16")
    var n = 0

    @Param("4096", "1024", "128", "16")
    var m = 0

    @Param("4096", "1024", "128", "16")
    var t = 0

    lateinit var left: CompositeArray
    lateinit var right: CompositeArray
    lateinit var dest: CompositeArray

    @Setup(Level.Iteration)
    fun setup() {
        left = CompositeArray(Strides(intArrayOf(n, t))) { Random.nextFloat() }
        right = CompositeArray(Strides(intArrayOf(t, m))) { Random.nextFloat() }
        dest = CompositeArray(Strides(intArrayOf(n, m))) { 0f }
    }

    @TearDown(Level.Iteration)
    fun teardown() {
        for (i in 0 until dest.blocksNum) {
            for (j in 0 until dest.blockSize) {
                dest.blocks[i][j] = 0f
            }
        }
    }

    @Benchmark
    fun baseline(blackhole: Blackhole) {
        for (i in 0 until n) {
            val destBlock = dest.blocks[i]
            val leftBlock = left.blocks[i]
            for (k in 0 until t) {
                val temp = leftBlock[k]
                val rightBlock = right.blocks[k]
                for (j in 0 until m) {
                    destBlock[j] += temp * rightBlock[j]
                }
            }
        }

        blackhole.consume(dest)
    }
}

@State(Scope.Benchmark)
@Fork(value = 1, warmups = 0)
@Warmup(iterations = 3)
@BenchmarkMode(Mode.SingleShotTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Measurement(iterations = 100)
open class DotBenchmarkCopy {
    @Param("4096", "1024", "128", "16")
    var n = 0

    @Param("4096", "1024", "128", "16")
    var m = 0

    @Param("4096", "1024", "128", "16")
    var t = 0

    lateinit var left: FloatArray
    lateinit var right: FloatArray
    lateinit var dest: FloatArray

    var leftSize = 0
    var rightSize = 0
    var destSize = 0

    @Setup(Level.Iteration)
    fun setup() {
        leftSize = n * t
        rightSize = t * m
        destSize = n * m

        left = FloatArray(leftSize) { Random.nextFloat() }
        right = FloatArray(rightSize) { Random.nextFloat() }
        dest = FloatArray(destSize)
    }

    @TearDown(Level.Iteration)
    fun teardown() {
        for (i in 0 until destSize) {
            dest[i] = 0f
        }
    }

    @Benchmark
    fun baselineCopy(blackhole: Blackhole) {
        dotBaselineCopy(left, right, dest, m, n, t)
        blackhole.consume(dest)
    }
}

@State(Scope.Benchmark)
@Fork(value = 1, warmups = 0)
@Warmup(iterations = 3)
@BenchmarkMode(Mode.SingleShotTime)
@OutputTimeUnit(TimeUnit.MICROSECONDS)
@Measurement(iterations = 100)
open class DotBenchmarkTiled {
    @Param("4096", "1024", "128", "16")
    var n = 0

    @Param("4096", "1024", "128", "16")
    var m = 0

    @Param("4096", "1024", "128", "16")
    var t = 0

    lateinit var left: TiledArray
    lateinit var right: TiledArray
    lateinit var dest: TiledArray

    @Setup(Level.Iteration)
    fun setup() {
        left = TiledArray(Strides(intArrayOf(n, t))) { Random.nextFloat() }
        right = TiledArray(Strides(intArrayOf(t, m))) { Random.nextFloat() }
        dest = TiledArray(Strides(intArrayOf(n, m))) { 0f }
    }

    @TearDown(Level.Iteration)
    fun teardown() {
        for (i in 0 until dest.blocksNum) {
            for (j in 0 until dest.blockSize) {
                dest.blocks[i][j] = 0f
            }
        }
    }

    @Benchmark
    fun baseline(blackhole: Blackhole) {
        dotTiled(left, right, dest, m, n, t)
        blackhole.consume(dest)
    }
}

fun dotBaseline(left: FloatArray, right: FloatArray, dest: FloatArray, m: Int, n: Int, t: Int) {
    for (i in 0 until n) {
        val dInd = i * m
        val lInd = i * t
        for (k in 0 until t) {
            val temp = left[lInd + k]
            val rInd = k * m
            for (j in 0 until m) {
                dest[dInd + j] += temp * right[rInd + j]
            }
        }
    }
}

const val min_block_size = 1024

fun dotBaselineCopy(left: FloatArray, right: FloatArray, dest: FloatArray, m: Int, n: Int, t: Int) {
    val leftBlockSize = if (t < min_block_size) t else {
        var num = t / min_block_size
        while (t % num != 0) num--
        t / num
    }

    val rdBlockSize = if (m < min_block_size) m else {
        var num = m / min_block_size
        while (m % num != 0) num--
        m / num
    }

    val leftBlock = FloatArray(leftBlockSize)
    val rightBlock = FloatArray(rdBlockSize)
    val destBlock = FloatArray(rdBlockSize)

    val rdBlocs = m / rdBlockSize
    val lBlocks = t / leftBlockSize

    for (rdCol in 0 until rdBlocs) {
        val rdTemp = rdCol * rdBlockSize
        val rdTempEnd = (rdCol + 1) * rdBlockSize

        for (i in 0 until n) {
            val destIdx = i * m
            val leftIdx = i * t

            var kTemp = 0
            dest.copyInto(destBlock, 0, destIdx + rdTemp, destIdx + rdTempEnd)
            for (lCol in 0 until lBlocks) {
                val leftTemp = lCol * leftBlockSize
                val startTemp = leftIdx + leftTemp
                val endTemp = leftIdx + leftTemp + leftBlockSize
                left.copyInto(leftBlock, 0, startTemp, endTemp)
                for (k in 0 until leftBlockSize) {
                    val rightIdx = kTemp * m
                    val temp = leftBlock[k]

                    right.copyInto(rightBlock, 0, rightIdx + rdTemp, rightIdx + rdTempEnd)

                    for (j in 0 until rdBlockSize) {
                        destBlock[j] += temp * rightBlock[j]
                    }

                    kTemp++
                }
            }

            /*for (k in 0 until t) {
                val temp = left[leftIdx + k]
                val rightIdx = k * m

                right.copyInto(rightBlock, 0, rightIdx + rdTemp, rightIdx + rdTempEnd)

                for (j in 0 until min_block_size) {
                    destBlock[j] += temp * rightBlock[j]
                }
            }*/

            destBlock.copyInto(dest, i * m + rdCol * min_block_size)
        }
    }
}

fun dotTiled(left: TiledArray, right: TiledArray, dest: TiledArray, m: Int, n: Int, t: Int) {
    val rdBlockSize = dest.blockSize
    for (rdCol in 0 until right.blocksInRow) {
        val rightIdx = rdCol * t
        val destIdx = rdCol * n

        for (i in 0 until n) {
            val destBlock = dest.blocks[destIdx + i]

            for (lCol in 0 until left.blocksInRow) {
                val leftBlock = left.blocks[i + lCol * n]
                val rightIdxOffset = rightIdx + left.blockSize * lCol

                for (k in 0 until left.blockSize) {
                    val temp = leftBlock[k]
                    val rightBlock = right.blocks[rightIdxOffset + k]

                    for (j in 0 until rdBlockSize) {
                        destBlock[j] += temp * rightBlock[j]
                    }
                }
            }
        }
    }
}

class BenchmarkTest {
    @Test
    @Tag("heavy")
    fun `test tiled dot`() {
        val r1 = Random(100)
        val r2 = Random(100)

        val n = 374
        val m = 16384
        val t = 4096

        val leftTiled = TiledArray(Strides(intArrayOf(n, t))) { r1.nextFloat() }
        val rightTiled = TiledArray(Strides(intArrayOf(t, m))) { r1.nextFloat() }
        val destTiled = TiledArray(Strides(intArrayOf(n, m))) { 0f }

        dotTiled(leftTiled, rightTiled, destTiled, m, n, t)

        val left = FloatArray(n * t) { r2.nextFloat() }
        val right = FloatArray(t * m) { r2.nextFloat() }
        val dest = FloatArray(n * m)

        dotBaseline(left, right, dest, m, n, t)

        val tiledResult = destTiled.toArray()
        assert(dest.contentEquals(tiledResult))
    }

    @Test
    @Tag("heavy")
    fun `test copy dot`() {
        val r1 = Random(100)

        val n = 1024
        val m = 1024
        val t = 1024

        val left = FloatArray(n * t) { r1.nextFloat() }
        val right = FloatArray(t * m) { r1.nextFloat() }
        val dest = FloatArray(n * m)

        val leftCopy = left.copyOf()
        val rightCopy = right.copyOf()
        val destCopy = dest.copyOf()

        dotBaseline(left, right, dest, m, n, t)
        dotBaselineCopy(leftCopy, rightCopy, destCopy, m, n, t)

        assert(dest.contentEquals(destCopy))
    }

    @Test
    @Tag("benchmark")
    fun `test dot performance`() {
        val opts = OptionsBuilder()
            .include("DotBenchmark")
//            .addProfiler("org.openjdk.jmh.profile.WinPerfAsmProfiler")
            .build()
        val results = Runner(opts).run().toTypedArray()

//        assert(results[0].primaryResult.getScore() < 500)
//        assert(results[1].primaryResult.getScore() < 150)
    }
}
