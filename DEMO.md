# Recall — 2-minute live demo script

Setup (before the audience arrives): emulator or phone with the debug build installed and the model bundle copied in
(`make export && python -m recall.compress.export_android && make android && make demo` once to warm up). Airplane mode
on — the point of the demo is that nothing needs the network.

| Time | Do | Say |
|---|---|---|
| 0:00 | Show the home screen, then the system settings page for the app: *no network permission listed*. | "Recall is a visual memory that lives entirely on the phone. It has no network permission at all — it can't upload anything even if it wanted to." |
| 0:15 | Tap **Start**. Point at the recording indicator in the status bar and the one-tap **Pause** in the notification. | "When it's on, you always know, and one tap stops it." |
| 0:25 | (Phone) Put your keys on the counter next to the fruit bowl, walk away. (Emulator) `ANDROID_SERIAL=emulator-5554 ./scripts/demo_e2e.sh` injects the synthetic *keys-on-table* clip. | "A tiny always-on gate — tens of kilobytes — watches for hands putting things down. Only those moments wake up the bigger models." |
| 0:50 | Open **Timeline**. | "This isn't video. Each row is one event: object, place, time, one redacted thumbnail. That's why a week fits in megabytes, not gigabytes." |
| 1:05 | Ask by voice or text: *"Where are my keys?"* | — |
| 1:10 | Read the answer card aloud (place, time, confidence, thumbnail). Tap **Show me the moment**. | "It tells me where, when, and how sure it is, and shows the evidence frame — I don't have to trust it blindly." |
| 1:25 | Ask *"Where is my passport?"* (never shown). | "And when it hasn't seen something, it says *I don't know* instead of making something up — the large vision-language baseline we compared against invents an answer most of the time." |
| 1:40 | Long-press the event → **Forget this**. Ask about the keys again → it no longer knows. | "Forgetting is one tap, and it's real deletion from the encrypted store." |
| 1:50 | Close. | "Honest caveat: on public kitchen video, object recognition from tiny crops is still the weak link — the paper shows exactly where it loses to baselines. What works is the system shape: sparse triggering, event memory, calibrated, grounded answers." |

Fallbacks: if the camera path fails, run the scripted e2e (`make demo`) which uses the same pipeline on the
committed synthetic clip. Never demo on footage of people who have not consented.
