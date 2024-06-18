package io.kinference.utils

import kotlin.math.*

object Errors {
    data class ErrorsData(
        val avgError: Double,
        val standardDeviation: Double,
        val maxError: Double,
        val p50: Double,
        val p95: Double,
        val p99: Double,
        val p999: Double
    ) {
        fun print(name: String, logger: KILogger) {
            logger.info { "Average error '$name' = $avgError" }
            logger.info { "Standard deviation '$name' = $standardDeviation" }
            logger.info { "Max error '$name' = $maxError" }
            logger.info { "Percentile 50 '$name' = $p50" }
            logger.info { "Percentile 95 '$name' = $p95" }
            logger.info { "Percentile 99 '$name' = $p99" }
            logger.info { "Percentile 99.9 '$name' = $p999\n" }
        }

        companion object {
            val EMPTY = ErrorsData(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

        }
    }

    private fun percentile(percentile: Double, size: Int) = floor(percentile * size).toInt()


    fun computeErrors(errorsArray: DoubleArray): ErrorsData {
        val averageError = if (errorsArray.isNotEmpty()) errorsArray.sum() / errorsArray.size else 0.0

        val standardDeviation =
            if (errorsArray.size > 1)
                sqrt(errorsArray.sumOf { (it - averageError).pow(2) } / (errorsArray.size - 1))
            else
                0.0

        val sortedErrorsArray = errorsArray.sorted()

        val percentile50 = sortedErrorsArray.getOrElse(percentile(0.5, sortedErrorsArray.size)) { 0.0 }
        val percentile95 = sortedErrorsArray.getOrElse(percentile(0.95, sortedErrorsArray.size)) { 0.0 }
        val percentile99 = sortedErrorsArray.getOrElse(percentile(0.99, sortedErrorsArray.size)) { 0.0 }
        val percentile999 = sortedErrorsArray.getOrElse(percentile(0.999, sortedErrorsArray.size)) { 0.0 }

        val maxError = if (sortedErrorsArray.isNotEmpty()) sortedErrorsArray.last() else 0.0

        return ErrorsData(
            averageError,
            standardDeviation,
            maxError,
            percentile50,
            percentile95,
            percentile99,
            percentile999
        )
    }
}
