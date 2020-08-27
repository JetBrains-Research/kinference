package org.jetbrains.research.kotlin.inference.benchmark.math

import org.jetbrains.research.kotlin.inference.extensions.primitives.dotInto
import org.openjdk.jmh.annotations.*
import org.openjdk.jmh.infra.Blackhole
import kotlin.random.Random

@State(Scope.Benchmark)
@Warmup(iterations = 3)
@Measurement(iterations = 15)
open class DotBenchmark {
    @Param("374")
    var m = 0

    @Param("50257")
    var n = 0

    @Param("1024")
    var t = 0

    lateinit var leftArray: FloatArray
    lateinit var rightArray: FloatArray

    lateinit var leftShape: IntArray
    lateinit var rightShape: IntArray

    lateinit var destinationArray: FloatArray

    @Setup
    fun setup() {

        leftShape = intArrayOf(m, t)
        rightShape = intArrayOf(t, n)

        leftArray = FloatArray(m * t) { Random.nextFloat() }
        rightArray = FloatArray(n * t) { Random.nextFloat() }

        destinationArray = FloatArray(m * n)
    }

    @Benchmark
    fun benchmark(blackhole: Blackhole) {
        dotInto(leftArray, 0, leftShape, rightArray, 0, rightShape, destinationArray, 0)
        blackhole.consume(destinationArray)
    }
}
