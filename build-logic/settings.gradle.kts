rootProject.name = "build-logic"

// Include the plugin project
include("test-plugin")

// Configure the version catalog to use the one from the parent project
dependencyResolutionManagement {
    repositories {
        mavenCentral()
        gradlePluginPortal()
    }
    
    // Use the version catalog from the parent project
    versionCatalogs {
        create("libs") {
            from(files("../gradle/libs.versions.toml"))
        }
    }
}
