package io.kinference.utils.time

import kotlin.js.JsName
import kotlin.math.*


/**
 * Timestamp MPP wrapper that is represented as seconds/millis since epoch start
 */
@Suppress("unused")
data class Timestamp constructor(@Suppress("NON_EXPORTABLE_TYPE") val millis: Long) {
    companion object {
        fun now() = Timestamp(Time.epochMillis())

        fun epoch() = Timestamp(0L)
    }

    override fun toString(): String {
        return millis.toString()
    }

    val seconds: Int
        get() = (millis.toDouble() / 1000).roundToInt()

    val minutes: Int
        get() = (seconds.toDouble() / 60).roundToInt()

    val hours: Int
        get() = (minutes.toDouble() / 60).roundToInt()

    val days: Int
        get() = (hours.toDouble() / 24).roundToInt()

    operator fun plus(duration: Duration): Timestamp {
        return Timestamp(this.millis + duration.millis)
    }

    operator fun minus(duration: Duration): Timestamp {
        return Timestamp(max(0, this.millis - duration.millis))
    }

    @JsName("minusPeriod")
    operator fun compareTo(other: Timestamp): Int {
        return this.millis.compareTo(other.millis)
    }

    /**
     * Duration between this and other time point
     *
     * It is calculated with "abs", so there is no difference
     * from what you substract what
     */
    fun between(other: Timestamp): Duration {
        return Duration(abs(this.millis - other.millis));
    }

    fun before(other: Timestamp): Boolean {
        return this < other;
    }

    fun beforeOrEqual(other: Timestamp): Boolean {
        return this <= other
    }

    fun after(other: Timestamp): Boolean {
        return this >= other;
    }

    fun afterOrEqual(other: Timestamp): Boolean {
        return this.millis >= other.millis
    }

}

