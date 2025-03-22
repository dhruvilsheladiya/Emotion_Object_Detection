import pyttsx3

engine = pyttsx3.init()
voices = engine.getProperty('voices')

print(f"Available voices: {len(voices)}")
for idx, voice in enumerate(voices):
    print(f"Voice {idx}: {voice.name} - {voice.id}")
