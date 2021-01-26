package io.kinference.algorithms.gec.pretokenizer.en

import io.kinference.algorithms.gec.pretokenizer.CharClasses

/**
 * SpecialCases is class-container for exceptions and url searching string
 * @param baseExceptions basic exceptions for tokenizer
 * @param tokenizerExceptions tokenizer exceptions for tokenizer
 * @param specialCases hash map for matching special token and information about this token
 * @param urls string which generates url-matching regex
 * @param urlRegex regex which match urls
 */
class SpecialCases {
    private val baseExceptions = BaseExceptions()
    private val tokenizerExceptions = TokenizerExceptions()

    var specialCases: HashMap<String, List<TokenInfo>>

    private var urlRegex: Regex

    val urls = """(?u)""" +
    // fmt: off
    """^""" +
    // protocol identifier (mods: make optional and expand schemes)
    // (see: https://www.iana.org/assignments/uri-schemes/uri-schemes.xhtml)
    """(?:(?:[\w\+\-\.]{2,})://)?""" +
    // mailto:user or user:pass authentication
    """(?:\S+(?::\S*)?@)?""" +
    """(?:""" +
    // IP address exclusion
    // private & local networks
    """(?!(?:10|127)(?:\.\d{1,3}){3})""" +
    """(?!(?:169\.254|192\.168)(?:\.\d{1,3}){2})""" +
    """(?!172\.(?:1[6-9]|2\d|3[0-1])(?:\.\d{1,3}){2})""" +
    // IP address dotted notation octets
    // excludes loopback network 0.0.0.0
    // excludes reserved space >= 224.0.0.0
    // excludes network & broadcast addresses
    // (first & last IP address of each class)
    // MH: Do we really need this? Seems excessive, and seems to have caused
    // Issue #957
    """(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])""" +
    """(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}""" +
    """(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))""" +
    """|""" +
    // host & domain names
    // mods: match is case-sensitive, so include [A-Z]
    """(?:""" +  // noqa
      """(?:""" +
        """[A-Za-z0-9\u00a1-\uffff]""" +
        """[A-Za-z0-9\u00a1-\uffff_-]{0,62}""" +
      """)?""" +
      """[A-Za-z0-9\u00a1-\uffff]\.""" +
    """)+""" +
    // TLD identifier
    // mods: use ALPHA_LOWER instead of a wider range so that this doesn't match
    // strings like "lower.Upper", which can be split on "." by infixes in some
    // languages
    """(?:[""" + CharClasses.AlphaLower + """]{2,63})""" +
    """)""" +
    // port number
    """(?::\d{2,5})?""" +
    // resource path
    """(?:[/?#]\S*)?""" +
    """$"""
    // fmt: on

    init {
        specialCases = tokenizerExceptions.exceptions
        specialCases.putAll(baseExceptions.exceptions)
        expandExceptions(search = "'", replace = "’")
        urlRegex = urls.toRegex()
    }


    private fun expandExceptions(search: String, replace: String){
        fun fixToken(token: TokenInfo, search: String, replace: String): TokenInfo{
            return TokenInfo(orth = token.orth.replace(search, replace), norm = token.norm, tag = token.tag, lemma = token.lemma)
        }
        val newItems = HashMap<String, List<TokenInfo>>()
        for ((tokenString, tokens) in specialCases){
            if (search in tokenString){
                val newKey = tokenString.replace(search, replace)
                val newValue = tokens.map { fixToken(it, search, replace) }
                newItems[newKey] = newValue
            }
        }
        specialCases.putAll(newItems)
    }

    fun get(token: String): List<TokenInfo>?{
        return if (specialCases.containsKey(token)){
            specialCases[token]
        }
        else{
            null
        }
    }

    fun urlMatch(token: String): Boolean{
        val matches = urlRegex.find(token)
        if (matches != null){
            return true
        }
        return false
    }
}
