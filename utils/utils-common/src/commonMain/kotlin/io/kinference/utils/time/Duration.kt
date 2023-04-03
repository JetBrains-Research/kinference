package io.kinference.utils.time

import kotlin.math.abs
import kotlin.math.roundToInt

@Suppress("unused")
data class Duration internal constructor(val millis: Long) {
    val isZero: Boolean
        get() = this.millis == 0L
    val seconds: Int
        get() = (millis.toDouble() / 1000).roundToInt()
    val minutes: Int
        get() = (seconds.toDouble() / 60).roundToInt()
    val hours: Int
        get() = (minutes.toDouble() / 60).roundToInt()
    val days: Int
        get() = (hours.toDouble() / 24).roundToInt()

    companion object {
        val ZERO = Duration(0L)
        fun seconds(seconds: Int): Duration {
            return Duration(seconds.toLong() * 1000)
        }

        fun minutes(minutes: Int): Duration {
            return seconds(minutes * 60)
        }

        fun hours(hours: Int): Duration {
            return minutes(hours * 60)
        }

        fun days(days: Int): Duration {
            return hours(days * 24)
        }
    }

    operator fun compareTo(other: Duration): Int {
        return this.millis.compareTo(other.millis)
    }

    fun less(other: Duration): Boolean {
        return this < other
    }

    fun lessOrEqual(other: Duration): Boolean {
        return this <= other
    }

    fun greater(other: Duration): Boolean {
        return this > other
    }

    fun greaterOrEqual(other: Duration): Boolean {
        return this >= other
    }


    /**
     * It would subtract one duration from another apply abs
     */
    operator fun minus(other: Duration): Duration {
        return Duration(abs(this.millis - other.millis))
    }

    operator fun plus(other: Duration): Duration {
        return Duration(this.millis + other.millis)
    }
}
