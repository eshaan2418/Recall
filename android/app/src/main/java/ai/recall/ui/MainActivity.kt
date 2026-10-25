package ai.recall.ui

import ai.recall.service.CaptureService
import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.speech.RecognitionListener
import android.speech.RecognizerIntent
import android.speech.SpeechRecognizer
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle

class MainActivity : ComponentActivity() {
    private val vm: RecallViewModel by viewModels()
    private var recognizer: SpeechRecognizer? = null
    private var afterPerms: (() -> Unit)? = null
    private val askPerms = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { res ->
        if (res.values.all { it }) afterPerms?.invoke()
        afterPerms = null
    }

    private val actions = object : UiActions {
        override fun toggleRecording() {
            if (vm.state.value.recording) {
                startService(Intent(this@MainActivity, CaptureService::class.java).setAction(CaptureService.ACTION_PAUSE))
            } else withPerms(recordPerms()) { startForegroundService(CaptureService.startIntent(this@MainActivity)) }
        }
        override fun setTab(i: Int) = vm.setTab(i)
        override fun setQuery(q: String) = vm.setQuery(q)
        override fun ask() = vm.ask()
        override fun voice() = withPerms(listOf(Manifest.permission.RECORD_AUDIO)) { listen() }
        override fun showMoment(t: Double) = vm.showMoment(t)
        override fun forget(id: Long) = vm.forget(id)
        override fun neverTrack(cls: Int) = vm.neverTrack(cls)
        override fun neverTrackByName(name: String) = vm.neverTrackByName(name)
        override fun unblockObject(cls: Int) = vm.unblockObject(cls)
        override fun setRetention(days: Int) = vm.setRetention(days)
        override fun setPlaceBlocked(id: Int, blocked: Boolean) = vm.setPlaceBlocked(id, blocked)
        override fun blockCurrentPlace() = vm.blockCurrentPlace()
        override fun wipe() = vm.wipe()
        override fun dismissMessage() = vm.dismissMessage()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            val s by vm.state.collectAsStateWithLifecycle()
            MaterialTheme { RecallApp(s, actions) }
        }
    }

    override fun onDestroy() {
        recognizer?.destroy()
        super.onDestroy()
    }

    private fun recordPerms() = buildList {
        add(Manifest.permission.CAMERA)
        if (Build.VERSION.SDK_INT >= 33) add(Manifest.permission.POST_NOTIFICATIONS)
    }

    private fun withPerms(perms: List<String>, then: () -> Unit) {
        val need = perms.filter { checkSelfPermission(it) != PackageManager.PERMISSION_GRANTED }
        if (need.isEmpty()) then() else { afterPerms = then; askPerms.launch(need.toTypedArray()) }
    }

    /**
     * On-device speech only: SpeechRecognizer.createOnDeviceSpeechRecognizer (API 31+) with EXTRA_PREFER_OFFLINE.
     * We deliberately do not fall back to the default (possibly cloud) recognition service.
     */
    private fun listen() {
        if (Build.VERSION.SDK_INT < 31 || !SpeechRecognizer.isOnDeviceRecognitionAvailable(this)) {
            vm.setListening(false, "On-device speech recognition isn't available on this phone — please type your question.")
            return
        }
        recognizer?.destroy()
        val sr = SpeechRecognizer.createOnDeviceSpeechRecognizer(this).also { recognizer = it }
        vm.setListening(true)
        sr.setRecognitionListener(object : RecognitionListener {
            override fun onResults(b: Bundle?) {
                val text = b?.getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION)?.firstOrNull().orEmpty()
                vm.setListening(false)
                if (text.isNotBlank()) { vm.setQuery(text); vm.ask() }
            }
            override fun onError(e: Int) = vm.setListening(false, "Didn't catch that (speech error $e).")
            override fun onReadyForSpeech(p: Bundle?) {}
            override fun onBeginningOfSpeech() {}
            override fun onRmsChanged(v: Float) {}
            override fun onBufferReceived(b: ByteArray?) {}
            override fun onEndOfSpeech() {}
            override fun onPartialResults(b: Bundle?) {}
            override fun onEvent(t: Int, b: Bundle?) {}
        })
        sr.startListening(Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH)
            .putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL, RecognizerIntent.LANGUAGE_MODEL_FREE_FORM)
            .putExtra(RecognizerIntent.EXTRA_PREFER_OFFLINE, true))
    }
}
