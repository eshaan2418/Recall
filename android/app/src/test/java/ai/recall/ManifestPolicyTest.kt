package ai.recall

import org.junit.Assert.assertEquals
import org.junit.Assert.assertTrue
import org.junit.Test
import java.io.File
import javax.xml.parsers.DocumentBuilderFactory

/**
 * Privacy policy: the merged RELEASE manifest (app + every library) must not request network access.
 * The path is injected by app/build.gradle.kts (it also makes this test depend on processReleaseMainManifest).
 */
class ManifestPolicyTest {
    private val manifest = File(requireNotNull(System.getProperty("recall.mergedReleaseManifest")) {
        "recall.mergedReleaseManifest not set — run via Gradle (testDebugUnitTest)"
    })
    private val forbidden = System.getProperty("recall.forbiddenPermissions")!!.split(",")

    private fun permissions(): List<String> {
        val doc = DocumentBuilderFactory.newInstance().apply { isNamespaceAware = true }.newDocumentBuilder().parse(manifest)
        val ns = "http://schemas.android.com/apk/res/android"
        return listOf("uses-permission", "uses-permission-sdk-23").flatMap { tag ->
            val nl = doc.getElementsByTagName(tag)
            (0 until nl.length).map { nl.item(it).attributes.getNamedItemNS(ns, "name").nodeValue }
        }
    }

    @Test fun mergedReleaseManifestHasNoNetworkPermission() {
        assertTrue("merged manifest missing: $manifest", manifest.isFile)
        val perms = permissions()
        assertTrue("sanity: camera permission present", "android.permission.CAMERA" in perms)
        assertEquals("forbidden permissions in merged release manifest", emptyList<String>(), perms.filter { it in forbidden })
        assertTrue(forbidden.contains("android.permission.INTERNET"))
    }

    @Test fun releaseServiceIsNotExported() {
        val text = manifest.readText()
        val svc = Regex("<service[^>]*CaptureService[^>]*>", RegexOption.DOT_MATCHES_ALL).find(text)!!.value
        assertTrue("release CaptureService must not be exported: $svc", svc.contains("android:exported=\"false\""))
    }

    @Test fun noBackup() {
        val app = Regex("<application[^>]*>", RegexOption.DOT_MATCHES_ALL).find(manifest.readText())!!.value
        assertTrue(app.contains("android:allowBackup=\"false\""))
    }
}
