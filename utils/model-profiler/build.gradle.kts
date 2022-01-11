group = rootProject.group
version = rootProject.version


kotlin {
    jvm()

    js(BOTH) {
        browser()
    }
}
