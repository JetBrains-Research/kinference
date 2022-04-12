package io.kinference.utils

import io.kinference.TestLoggerFactory
import io.kinference.ndarray.arrays.tiled.*
import kotlin.math.*
import kotlin.test.assertEquals
import kotlin.test.assertTrue

object ArrayAssertions {
    private val logger = TestLoggerFactory.create("io.kinference.utils.ArrayAssertions")

    fun assertEquals(expect: FloatArray, actual: FloatArray, delta: Double, tensorName: String) {
        val errorsArray = FloatArray(expect.size) { i ->
            abs(expect[i] - actual[i])
        }

        val averageError = if (errorsArray.isNotEmpty()) errorsArray.sum() / errorsArray.size else 0f
        val standardDeviation =
            if (errorsArray.size > 1)
                sqrt(errorsArray.sumOf { (it - averageError).pow(2).toDouble() } / (errorsArray.size - 1))
            else
                0f

        val sortedErrorsArray = errorsArray.sorted()

        val percentile50 = sortedErrorsArray.getOrElse(floor(0.5 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile95 = sortedErrorsArray.getOrElse(floor(0.95 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile99 = sortedErrorsArray.getOrElse(floor(0.99 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile999 = sortedErrorsArray.getOrElse(floor(0.999 * sortedErrorsArray.size).toInt()) { 0f }

        logger.info { "Average error '${tensorName}' = $averageError" }
        logger.info { "Standard deviation '${tensorName}' = $standardDeviation" }
        if (sortedErrorsArray.isNotEmpty()) logger.info { "Max error '${tensorName}' = ${sortedErrorsArray.last()}" }
        logger.info { "Percentile 50 '${tensorName}' = $percentile50" }
        logger.info { "Percentile 95 '${tensorName}' = $percentile95" }
        logger.info { "Percentile 99 '${tensorName}' = $percentile99" }
        logger.info { "Percentile 99.9 '${tensorName}' = $percentile999\n" }

        assertArrayEquals(expect, actual, { l, r -> abs(l - r).toDouble() }, delta, "Tensor $tensorName does not match")
    }

    fun assertEquals(expect: FloatTiledArray, actual: FloatTiledArray, delta: Double, tensorName: String) {
        val errorsArray = FloatArray(expect.size) { i ->
            abs(expect[i] - actual[i])
        }

        val averageError = if (errorsArray.isNotEmpty()) errorsArray.sum() / errorsArray.size else 0f
        val standardDeviation =
            if (errorsArray.isNotEmpty())
                sqrt(errorsArray.sumOf { (it - averageError).pow(2).toDouble() } / (errorsArray.size - 1))
            else
                0f

        val sortedErrorsArray = errorsArray.sorted()

        val percentile50 = sortedErrorsArray.getOrElse(floor(0.5 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile95 = sortedErrorsArray.getOrElse(floor(0.95 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile99 = sortedErrorsArray.getOrElse(floor(0.99 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile999 = sortedErrorsArray.getOrElse(floor(0.999 * sortedErrorsArray.size).toInt()) { 0f }

        logger.info { "Average error '${tensorName}' = $averageError" }
        logger.info { "Standard deviation '${tensorName}' = $standardDeviation" }
        if (sortedErrorsArray.isNotEmpty()) logger.info { "Max error '${tensorName}' = ${sortedErrorsArray.last()}" }
        logger.info { "Percentile 50 '${tensorName}' = $percentile50" }
        logger.info { "Percentile 95 '${tensorName}' = $percentile95" }
        logger.info { "Percentile 99 '${tensorName}' = $percentile99" }
        logger.info { "Percentile 99.9 '${tensorName}' = $percentile999\n" }

        assertArrayEquals(expect, actual, { l, r -> abs(l - r).toDouble() }, delta, "Tensor $tensorName does not match")
    }

    fun assertEquals(expect: DoubleArray, actual: DoubleArray, delta: Double, tensorName: String) {
        val errorsArray = DoubleArray(expect.size) { i ->
            abs(expect[i] - actual[i])
        }

        val averageError = if (errorsArray.isNotEmpty()) errorsArray.sum() / errorsArray.size else 0.0
        val standardDeviation =
            if (errorsArray.isNotEmpty())
                sqrt(errorsArray.sumOf { (it - averageError).pow(2) } / (errorsArray.size - 1))
            else
                0f

        val sortedErrorsArray = errorsArray.sorted()

        val percentile50 = sortedErrorsArray.getOrElse(floor(0.5 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile95 = sortedErrorsArray.getOrElse(floor(0.95 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile99 = sortedErrorsArray.getOrElse(floor(0.99 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile999 = sortedErrorsArray.getOrElse(floor(0.999 * sortedErrorsArray.size).toInt()) { 0f }

        logger.info { "Average error '${tensorName}' = $averageError" }
        logger.info { "Standard deviation '${tensorName}' = $standardDeviation" }
        if (sortedErrorsArray.isNotEmpty()) logger.info { "Max error '${tensorName}' = ${sortedErrorsArray.last()}" }
        logger.info { "Percentile 50 '${tensorName}' = $percentile50" }
        logger.info { "Percentile 95 '${tensorName}' = $percentile95" }
        logger.info { "Percentile 99 '${tensorName}' = $percentile99" }
        logger.info { "Percentile 99.9 '${tensorName}' = $percentile999\n" }

        assertArrayEquals(expect, actual, { l, r -> abs(l - r) }, delta, "Tensor $tensorName does not match")
    }

    fun assertEquals(expect: DoubleTiledArray, actual: DoubleTiledArray, delta: Double, tensorName: String) {
        val errorsArray = DoubleArray(expect.size) { i ->
            abs(expect[i] - actual[i])
        }

        val averageError = if (errorsArray.isNotEmpty()) errorsArray.sum() / errorsArray.size else 0.0
        val standardDeviation =
            if (errorsArray.isNotEmpty())
                sqrt(errorsArray.sumOf { (it - averageError).pow(2) } / (errorsArray.size - 1))
            else
                0f

        val sortedErrorsArray = errorsArray.sorted()

        val percentile50 = sortedErrorsArray.getOrElse(floor(0.5 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile95 = sortedErrorsArray.getOrElse(floor(0.95 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile99 = sortedErrorsArray.getOrElse(floor(0.99 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile999 = sortedErrorsArray.getOrElse(floor(0.999 * sortedErrorsArray.size).toInt()) { 0f }

        logger.info { "Average error '${tensorName}' = $averageError" }
        logger.info { "Standard deviation '${tensorName}' = $standardDeviation" }
        if (sortedErrorsArray.isNotEmpty()) logger.info { "Max error '${tensorName}' = ${sortedErrorsArray.last()}" }
        logger.info { "Percentile 50 '${tensorName}' = $percentile50" }
        logger.info { "Percentile 95 '${tensorName}' = $percentile95" }
        logger.info { "Percentile 99 '${tensorName}' = $percentile99" }
        logger.info { "Percentile 99.9 '${tensorName}' = $percentile999\n" }

        assertArrayEquals(expect, actual, { l, r -> abs(l - r) }, delta, "Tensor $tensorName does not match")
    }

    fun assertEquals(expect: LongArray, actual: LongArray, delta: Double, tensorName: String) {
        val errorsArray = LongArray(expect.size) { i ->
            abs(expect[i] - actual[i])
        }

        val averageError = if (errorsArray.isNotEmpty()) errorsArray.sum().toDouble() / errorsArray.size else 0.0
        val standardDeviation =
            if (errorsArray.isNotEmpty())
                sqrt(errorsArray.sumOf { (it - averageError).pow(2) } / (errorsArray.size - 1))
            else
                0.0

        val sortedErrorsArray = errorsArray.sorted()

        val percentile50 = sortedErrorsArray.getOrElse(floor(0.5 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile95 = sortedErrorsArray.getOrElse(floor(0.95 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile99 = sortedErrorsArray.getOrElse(floor(0.99 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile999 = sortedErrorsArray.getOrElse(floor(0.999 * sortedErrorsArray.size).toInt()) { 0f }

        logger.info { "Average error '${tensorName}' = $averageError" }
        logger.info { "Standard deviation '${tensorName}' = $standardDeviation" }
        if (sortedErrorsArray.isNotEmpty()) logger.info { "Max error '${tensorName}' = ${sortedErrorsArray.last()}" }
        logger.info { "Percentile 50 '${tensorName}' = $percentile50" }
        logger.info { "Percentile 95 '${tensorName}' = $percentile95" }
        logger.info { "Percentile 99 '${tensorName}' = $percentile99" }
        logger.info { "Percentile 99.9 '${tensorName}' = $percentile999\n" }

        assertArrayEquals(expect, actual, { l, r -> abs(l - r).toDouble() }, delta, "Tensor $tensorName does not match")
    }

    fun assertEquals(expect: LongTiledArray, actual: LongTiledArray, delta: Double, tensorName: String) {
        val errorsArray = LongArray(expect.size) { i ->
            abs(expect[i] - actual[i])
        }

        val averageError = if (errorsArray.isNotEmpty()) errorsArray.sum().toDouble() / errorsArray.size else 0.0
        val standardDeviation =
            if (errorsArray.isNotEmpty())
                sqrt(errorsArray.sumOf { (it - averageError).pow(2) } / (errorsArray.size - 1))
            else
                0.0

        val sortedErrorsArray = errorsArray.sorted()

        val percentile50 = sortedErrorsArray.getOrElse(floor(0.5 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile95 = sortedErrorsArray.getOrElse(floor(0.95 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile99 = sortedErrorsArray.getOrElse(floor(0.99 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile999 = sortedErrorsArray.getOrElse(floor(0.999 * sortedErrorsArray.size).toInt()) { 0f }

        logger.info { "Average error '${tensorName}' = $averageError" }
        logger.info { "Standard deviation '${tensorName}' = $standardDeviation" }
        if (sortedErrorsArray.isNotEmpty()) logger.info { "Max error '${tensorName}' = ${sortedErrorsArray.last()}" }
        logger.info { "Percentile 50 '${tensorName}' = $percentile50" }
        logger.info { "Percentile 95 '${tensorName}' = $percentile95" }
        logger.info { "Percentile 99 '${tensorName}' = $percentile99" }
        logger.info { "Percentile 99.9 '${tensorName}' = $percentile999\n" }

        assertArrayEquals(expect, actual, { l, r -> abs(l - r).toDouble() }, delta, "Tensor $tensorName does not match")
    }

    fun assertEquals(expect: IntArray, actual: IntArray, delta: Double, tensorName: String) {
        val errorsArray = IntArray(expect.size) { i ->
            abs(expect[i] - actual[i])
        }

        val averageError = if (errorsArray.isNotEmpty()) errorsArray.sum().toDouble() / errorsArray.size else 0.0
        val standardDeviation =
            if (errorsArray.isNotEmpty())
                sqrt(errorsArray.sumOf { (it - averageError).pow(2) } / (errorsArray.size - 1))
            else
                0.0

        val sortedErrorsArray = errorsArray.sorted()

        val percentile50 = sortedErrorsArray.getOrElse(floor(0.5 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile95 = sortedErrorsArray.getOrElse(floor(0.95 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile99 = sortedErrorsArray.getOrElse(floor(0.99 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile999 = sortedErrorsArray.getOrElse(floor(0.999 * sortedErrorsArray.size).toInt()) { 0f }

        logger.info { "Average error '${tensorName}' = $averageError" }
        logger.info { "Standard deviation '${tensorName}' = $standardDeviation" }
        if (sortedErrorsArray.isNotEmpty()) logger.info { "Max error '${tensorName}' = ${sortedErrorsArray.last()}" }
        logger.info { "Percentile 50 '${tensorName}' = $percentile50" }
        logger.info { "Percentile 95 '${tensorName}' = $percentile95" }
        logger.info { "Percentile 99 '${tensorName}' = $percentile99" }
        logger.info { "Percentile 99.9 '${tensorName}' = $percentile999\n" }

        assertArrayEquals(expect, actual, { l, r -> abs(l - r).toDouble() }, delta, "Tensor $tensorName does not match")
    }

    fun assertEquals(expect: IntTiledArray, actual: IntTiledArray, delta: Double, tensorName: String) {
        val errorsArray = IntArray(expect.size) { i ->
            abs(expect[i] - actual[i])
        }

        val averageError = if (errorsArray.isNotEmpty()) errorsArray.sum().toDouble() / errorsArray.size else 0.0
        val standardDeviation =
            if (errorsArray.isNotEmpty())
                sqrt(errorsArray.sumOf { (it - averageError).pow(2) } / (errorsArray.size - 1))
            else
                0.0

        val sortedErrorsArray = errorsArray.sorted()

        val percentile50 = sortedErrorsArray.getOrElse(floor(0.5 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile95 = sortedErrorsArray.getOrElse(floor(0.95 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile99 = sortedErrorsArray.getOrElse(floor(0.99 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile999 = sortedErrorsArray.getOrElse(floor(0.999 * sortedErrorsArray.size).toInt()) { 0f }

        logger.info { "Average error '${tensorName}' = $averageError" }
        logger.info { "Standard deviation '${tensorName}' = $standardDeviation" }
        if (sortedErrorsArray.isNotEmpty()) logger.info { "Max error '${tensorName}' = ${sortedErrorsArray.last()}" }
        logger.info { "Percentile 50 '${tensorName}' = $percentile50" }
        logger.info { "Percentile 95 '${tensorName}' = $percentile95" }
        logger.info { "Percentile 99 '${tensorName}' = $percentile99" }
        logger.info { "Percentile 99.9 '${tensorName}' = $percentile999\n" }

        assertArrayEquals(expect, actual, { l, r -> abs(l - r).toDouble() }, delta, "Tensor $tensorName does not match")
    }

    fun assertEquals(expect: UByteArray, actual: UByteArray, delta: Double, tensorName: String) {
        val errorsArray = IntArray(expect.size) { i ->
            abs(expect[i].toInt() - actual[i].toInt())
        }

        val averageError = if (errorsArray.isNotEmpty()) errorsArray.sum().toDouble() / errorsArray.size else 0.0
        val standardDeviation =
            if (errorsArray.isNotEmpty())
                sqrt(errorsArray.sumOf { (it - averageError).pow(2) } / (errorsArray.size - 1))
            else
                0.0

        val sortedErrorsArray = errorsArray.sorted()

        val percentile50 = sortedErrorsArray.getOrElse(floor(0.5 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile95 = sortedErrorsArray.getOrElse(floor(0.95 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile99 = sortedErrorsArray.getOrElse(floor(0.99 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile999 = sortedErrorsArray.getOrElse(floor(0.999 * sortedErrorsArray.size).toInt()) { 0f }

        logger.info { "Average error '${tensorName}' = $averageError" }
        logger.info { "Standard deviation '${tensorName}' = $standardDeviation" }
        if (sortedErrorsArray.isNotEmpty()) logger.info { "Max error '${tensorName}' = ${sortedErrorsArray.last()}" }
        logger.info { "Percentile 50 '${tensorName}' = $percentile50" }
        logger.info { "Percentile 95 '${tensorName}' = $percentile95" }
        logger.info { "Percentile 99 '${tensorName}' = $percentile99" }
        logger.info { "Percentile 99.9 '${tensorName}' = $percentile999\n" }

        assertArrayEquals(expect, actual, { l, r -> abs(l.toInt() - r.toInt()).toDouble() }, delta, "Tensor $tensorName does not match")
    }

    fun assertEquals(expect: UByteTiledArray, actual: UByteTiledArray, delta: Double, tensorName: String) {
        val errorsArray = IntArray(expect.size) { i ->
            abs(expect[i].toInt() - actual[i].toInt())
        }

        val averageError = if (errorsArray.isNotEmpty()) errorsArray.sum().toDouble() / errorsArray.size else 0.0
        val standardDeviation =
            if (errorsArray.isNotEmpty())
                sqrt(errorsArray.sumOf { (it - averageError).pow(2) } / (errorsArray.size - 1))
            else
                0.0

        val sortedErrorsArray = errorsArray.sorted()

        val percentile50 = sortedErrorsArray.getOrElse(floor(0.5 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile95 = sortedErrorsArray.getOrElse(floor(0.95 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile99 = sortedErrorsArray.getOrElse(floor(0.99 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile999 = sortedErrorsArray.getOrElse(floor(0.999 * sortedErrorsArray.size).toInt()) { 0f }

        logger.info { "Average error '${tensorName}' = $averageError" }
        logger.info { "Standard deviation '${tensorName}' = $standardDeviation" }
        if (sortedErrorsArray.isNotEmpty()) logger.info { "Max error '${tensorName}' = ${sortedErrorsArray.last()}" }
        logger.info { "Percentile 50 '${tensorName}' = $percentile50" }
        logger.info { "Percentile 95 '${tensorName}' = $percentile95" }
        logger.info { "Percentile 99 '${tensorName}' = $percentile99" }
        logger.info { "Percentile 99.9 '${tensorName}' = $percentile999\n" }

        assertArrayEquals(expect, actual, { l, r -> abs(l.toInt() - r.toInt()).toDouble() }, delta, "Tensor $tensorName does not match")
    }

    fun assertEquals(expect: ShortArray, actual: ShortArray, delta: Double, tensorName: String) {
        val errorsArray = IntArray(expect.size) { i ->
            abs(expect[i] - actual[i])
        }

        val averageError = if (errorsArray.isNotEmpty()) errorsArray.sum().toDouble() / errorsArray.size else 0.0
        val standardDeviation =
            if (errorsArray.isNotEmpty())
                sqrt(errorsArray.sumOf { (it - averageError).pow(2) } / (errorsArray.size - 1))
            else
                0.0

        val sortedErrorsArray = errorsArray.sorted()

        val percentile50 = sortedErrorsArray.getOrElse(floor(0.5 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile95 = sortedErrorsArray.getOrElse(floor(0.95 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile99 = sortedErrorsArray.getOrElse(floor(0.99 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile999 = sortedErrorsArray.getOrElse(floor(0.999 * sortedErrorsArray.size).toInt()) { 0f }

        logger.info { "Average error '${tensorName}' = $averageError" }
        logger.info { "Standard deviation '${tensorName}' = $standardDeviation" }
        if (sortedErrorsArray.isNotEmpty()) logger.info { "Max error '${tensorName}' = ${sortedErrorsArray.last()}" }
        logger.info { "Percentile 50 '${tensorName}' = $percentile50" }
        logger.info { "Percentile 95 '${tensorName}' = $percentile95" }
        logger.info { "Percentile 99 '${tensorName}' = $percentile99" }
        logger.info { "Percentile 99.9 '${tensorName}' = $percentile999\n" }

        assertArrayEquals(expect, actual, { l, r -> abs(l - r).toDouble() }, delta, "Tensor $tensorName does not match")
    }

    fun assertEquals(expect: ByteArray, actual: ByteArray, delta: Double, tensorName: String) {
        val errorsArray = IntArray(expect.size) { i ->
            abs(expect[i] - actual[i])
        }

        val averageError = if (errorsArray.isNotEmpty()) errorsArray.sum().toDouble() / errorsArray.size else 0.0
        val standardDeviation =
            if (errorsArray.isNotEmpty())
                sqrt(errorsArray.sumOf { (it - averageError).pow(2) } / (errorsArray.size - 1))
            else
                0.0

        val sortedErrorsArray = errorsArray.sorted()

        val percentile50 = sortedErrorsArray.getOrElse(floor(0.5 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile95 = sortedErrorsArray.getOrElse(floor(0.95 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile99 = sortedErrorsArray.getOrElse(floor(0.99 * sortedErrorsArray.size).toInt()) { 0f }
        val percentile999 = sortedErrorsArray.getOrElse(floor(0.999 * sortedErrorsArray.size).toInt()) { 0f }

        logger.info { "Average error '${tensorName}' = $averageError" }
        logger.info { "Standard deviation '${tensorName}' = $standardDeviation" }
        if (sortedErrorsArray.isNotEmpty()) logger.info { "Max error '${tensorName}' = ${sortedErrorsArray.last()}" }
        logger.info { "Percentile 50 '${tensorName}' = $percentile50" }
        logger.info { "Percentile 95 '${tensorName}' = $percentile95" }
        logger.info { "Percentile 99 '${tensorName}' = $percentile99" }
        logger.info { "Percentile 99.9 '${tensorName}' = $percentile999\n" }

        assertArrayEquals(expect, actual, { l, r -> abs(l - r).toDouble() }, delta, "Tensor $tensorName does not match")
    }

    fun assertArrayEquals(left: FloatTiledArray, right: FloatTiledArray, diff: (Float, Float) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for (i in 0 until left.size) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: FloatArray, right: FloatArray, diff: (Float, Float) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for (i in left.indices) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: DoubleTiledArray, right: DoubleTiledArray, diff: (Double, Double) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for (i in 0 until left.size) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: DoubleArray, right: DoubleArray, diff: (Double, Double) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for (i in left.indices) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: LongTiledArray, right: LongTiledArray, diff: (Long, Long) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for (i in 0 until left.size) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: LongArray, right: LongArray, diff: (Long, Long) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for (i in left.indices) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: IntTiledArray, right: IntTiledArray, diff: (Int, Int) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for (i in 0 until left.size) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: IntArray, right: IntArray, diff: (Int, Int) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for (i in left.indices) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: UIntArray, right: UIntArray, diff: (UInt, UInt) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for (i in left.indices) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: UByteTiledArray, right: UByteTiledArray, diff: (UByte, UByte) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for (i in 0 until left.size) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: UByteArray, right: UByteArray, diff: (UByte, UByte) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for (i in left.indices) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: ShortArray, right: ShortArray, diff: (Short, Short) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for (i in left.indices) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun assertArrayEquals(left: ByteArray, right: ByteArray, diff: (Byte, Byte) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for (i in left.indices) {
            val l = left[i]
            val r = right[i]

            assertTrue(diff(l, r) <= delta, message)
        }
    }

    fun <T : Comparable<T>> assertArrayEquals(left: Array<T>, right: Array<T>, diff: (T, T) -> Double, delta: Double, message: String = "") {
        assertEquals(left.size, right.size, message)
        for ((l, r) in left.zip(right)) {
            assertTrue(diff(l, r) <= delta, message)
        }
    }


    fun <T> assertArrayEquals(left: Array<T>, right: Array<T>, message: String = "") {
        assertEquals(left.size, right.size, message)
        for ((l, r) in left.zip(right)) {
            assertEquals(l, r, message)
        }
    }
}
