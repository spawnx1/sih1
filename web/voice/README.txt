JARVIS custom voice — how to drop in a cloned voice
====================================================

The console plays a bundled audio clip as JARVIS's voice when you open a case.
It looks for, in order:
    web/voice/<ack_no>.mp3     (a clip specific to that complaint)
    web/voice/default.mp3      (a generic clip played for any case)
    ...otherwise it uses the built-in system voice.

So you have two easy options:

OPTION 1 — one generic line (fastest)
  Put a single file here named:  default.mp3
  Suggested script to synthesise in your cloned voice:
    "Cash-out forecast assistant online. I have reviewed the complaint —
     the situation and recommended actions are on your screen."
  This plays in your voice for EVERY case.

OPTION 2 — per-case clips (most convincing for the demo)
  For each case you'll demo, open it in the console, copy the exact sentence
  JARVIS shows in the SITUATION panel, synthesise it in your cloned voice, and
  save it here named after that case's full 14-digit ack number, e.g.:
    24866488770974.mp3
  (The card shows only the first 6 digits; get the full number from the URL
   /predict/<ack_no> or ask for it — I can print it for you.)

HOW TO MAKE THE CLIP FROM hairshvoice.mp3
  - Use any voice-cloning tool (e.g. ElevenLabs free tier, Instant Voice Clone):
      1. Upload hairshvoice.mp3 to create the cloned voice.
      2. Paste the script text.
      3. Download the generated MP3.
      4. Rename it as above and drop it in this folder.
  - Only use a voice you are allowed to use (your own / a teammate's with consent).

No rebuild needed — the server serves this folder; just refresh the page.
