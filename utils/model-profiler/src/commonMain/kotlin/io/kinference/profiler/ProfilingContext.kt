package io.kinference.profiler

import kotlin.time.ExperimentalTime
import kotlin.time.measureTime


@ExperimentalTime
class ProfilingContext(val name: String) {
    private val entries: MutableList<ProfileEntry> = ArrayList()
    private var time: Long = 0

    fun profile(name: String, block: (context: ProfilingContext?) -> Unit) {
        val context = ProfilingContext(name)
        val time = measureTime {
            block(context)
        }

        this.time += time.toLongNanoseconds()
        entries.add(ProfileEntry(name, time.toLongNanoseconds(), context.entries))
    }

    fun entry(): ProfileEntry = ProfileEntry(name, time, entries.toMutableList())
}

@ExperimentalTime
fun ProfilingContext?.profile(name: String, block: (context: ProfilingContext?) -> Unit) =
    this?.profile(name, block) ?: block(null)

@ExperimentalTime
fun Collection<ProfilingContext>.analyze(name: String): ProfileAnalysisEntry {
    val builder = ProfileAnalysisBuilder(name)
    for (context in this) {
        val profile = context.entry()
        profile.writeToAnalysisBuilder(builder)
    }

    return builder.analyze()
}
