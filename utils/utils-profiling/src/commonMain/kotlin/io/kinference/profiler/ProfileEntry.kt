package io.kinference.profiler

import io.kinference.utils.time.Duration
import io.kinference.utils.time.average
import kotlin.math.floor
import kotlin.math.roundToInt

data class ProfileEntry(val name: String, val duration: Duration, val children: List<ProfileEntry>) {
    fun writeToStringBuilder(builder: StringBuilder, indent: Int = 0) {
        builder.append("${" ".repeat(indent)}$name - ${duration.millis} ms\n")
        for (entry in children) entry.writeToStringBuilder(builder, indent + 1)
    }
}

data class ProfileAnalysisEntry(
    val name: String,

    val avg: Duration,
    val max: Duration,
    val min: Duration,
    val p50: Duration,
    val p90: Duration,

    val children: List<ProfileAnalysisEntry>
) {
    private fun Double.toRoundString(): String {
        val rounded = this.toInt()
        return "${rounded}.${((this - rounded) * 100.0).roundToInt().toString().padStart(2, '0')}".padStart(7)
    }

    fun writeToStringBuilder(builder: StringBuilder, indent: Int = 0) {
        builder.append("\n${("  ".repeat(indent) + name).padEnd(40)} AVG: ${avg.millis} ms | MAX: ${max.millis} ms | MIN: ${min.millis} ms | Percentile 50: ${p50.millis} ms | Percentile 90: ${p90.millis} ms")
        for (entry in children) {
            entry.writeToStringBuilder(builder, indent + 1)
        }
    }

    fun getInfo(): String = buildString { writeToStringBuilder(this) }
}

internal fun ProfileEntry.writeToAnalysisBuilder(builder: ProfileAnalysisBuilder) {
    // TODO: support conditional/loop operators
    require(this.name == builder.name) { "Entries must be from the same profiling context" }

    builder.times.add(this.duration)

    if (builder.children.size == 0 && children.isNotEmpty()) {
        for (child in children) {
            builder.children.add(ProfileAnalysisBuilder(child.name))
        }
    }

    require(children.size == builder.children.size) { "Entries must be from the same profiling context" }

    for ((index, child) in children.withIndex()) {
        child.writeToAnalysisBuilder(builder.children[index])
    }
}

internal data class ProfileAnalysisBuilder(val name: String,
                                          val times: MutableList<Duration> = ArrayList(),
                                          val children: MutableList<ProfileAnalysisBuilder> = ArrayList()) {
    fun analyze(): ProfileAnalysisEntry {
        require(times.isNotEmpty()) { "Nothing to analyze" }

        val analyzedChildren: MutableList<ProfileAnalysisEntry> = ArrayList()
        for (child in children) {
            analyzedChildren.add(child.analyze())
        }

        times.sortBy { it.millis }
        return ProfileAnalysisEntry(
            name,
            times.average(),
            times.last(),
            times.first(),
            times[floor(0.5 * times.size).toInt()],
            times[floor(0.9 * times.size).toInt()],
            analyzedChildren
        )
    }
}
