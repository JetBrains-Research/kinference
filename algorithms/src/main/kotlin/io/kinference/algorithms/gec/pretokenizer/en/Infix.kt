package io.kinference.algorithms.gec.pretokenizer.en

import io.kinference.algorithms.gec.pretokenizer.CharClasses

/**
 * Infix class which contains information about english infixes
 * @param infixes list of String which contains icons, ellipses and several regex String which can search infixes in text
 * @param infixesRegex Regex for searching infixes based on [infixes]
 */
class Infix {

    private val infixes = CharClasses.ListEllipses + CharClasses.ListIcons +
        listOf("""(?<=[0-9])[+\-\*^](?=[0-9-])""",
            """(?<=[${CharClasses.AlphaLower}}${CharClasses.ConcatQuotes}])\.(?=[${CharClasses.AlphaUpper}${CharClasses.ConcatQuotes}])""",
            """(?<=[${CharClasses.Alpha}]),(?=[${CharClasses.Alpha}])""",
            """(?<=[${CharClasses.Alpha}])(?:${CharClasses.Hyphens})(?=[${CharClasses.Alpha}])""",
            """(?<=[${CharClasses.Alpha}0-9])[:<>=/](?=[${CharClasses.Alpha}])""")

    val infixesRegex: Regex?

    init {
        infixesRegex = compileSuffix()
    }

    fun compileSuffix() : Regex? {
        if (infixes.isNotEmpty()){
            return infixes.filter { it.trim() != "" }.joinToString("|") { it }.toRegex()
        }
        return null
    }
}
